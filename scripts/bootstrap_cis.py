"""Compute proper bootstrap CIs on raw cosines (not approximate-from-quartiles).

Re-embeds the held-out corpus with each encoder, samples pairs, computes
cosines, then bootstrap-resamples 1000 times to produce 95% CIs on the
median for each (encoder, bucket).

Output: data/codeswitch_pairs_lase_heldout/bootstrap_cis.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

random.seed(1337)
np.random.seed(1337)

ROOT = Path(__file__).resolve().parents[2]
LASE_CKPT = ROOT / "data/codeswitch_pairs_lase/r1_last.pt"
SAMPLE_PAIRS = 200
N_BOOTSTRAP = 1000


def _wavlm_sv_embedder():
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    proc = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv"); model.eval()
    def embed(p):
        import torchaudio
        wav, sr = torchaudio.load(str(p))
        if sr != 16_000: wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        with torch.inference_mode():
            out = model(**proc(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt"))
        return out.embeddings[0].numpy()
    return embed


def _ecapa_embedder():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                        savedir="/tmp/spkrec-ecapa-voxceleb")
    def embed(p):
        import torchaudio
        wav, sr = torchaudio.load(str(p))
        if sr != 16_000: wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        with torch.inference_mode():
            return m.encode_batch(wav).squeeze().numpy()
    return embed


def _lase_embedder():
    import torch
    from models.novel.lase import LambdaSchedule, LASE, WavLMSpeakerEncoder
    backbone = WavLMSpeakerEncoder(model_name="microsoft/wavlm-base-plus",
                                    embedding_dim=256, freeze_backbone=True)
    schedule = LambdaSchedule(warmup_steps=200, ramp_steps=500, peak=0.1)
    model = LASE(backbone=backbone, embedding_dim=256, n_languages=4,
                 lambda_schedule=schedule)
    state = torch.load(LASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False); model.eval()
    def embed(p):
        import torchaudio
        wav, sr = torchaudio.load(str(p))
        if sr != 16_000: wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        wav = wav[0][:32_000]
        if wav.shape[0] < 32_000:
            wav = torch.nn.functional.pad(wav, (0, 32_000 - wav.shape[0]))
        with torch.inference_mode():
            return model(wav.unsqueeze(0))["embedding"][0].cpu().numpy()
    return embed


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _bootstrap_ci(cosines: list[float], n_boot=N_BOOTSTRAP) -> tuple[float, float, float]:
    arr = np.array(cosines)
    medians = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        medians.append(np.median(sample))
    medians = np.array(medians)
    return float(np.median(arr)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _eval_corpus(name: str, manifest_path: Path):
    print(f"\n=== {name} ===")
    rows = [json.loads(ln) for ln in manifest_path.read_text().splitlines() if ln.strip()]
    print(f"  {len(rows)} pairs")

    def sample(pred):
        N = len(rows); out = []; seen = set(); attempts = 0
        while len(out) < SAMPLE_PAIRS and attempts < SAMPLE_PAIRS * 50:
            attempts += 1
            i, j = random.sample(range(N), 2)
            if (i, j) in seen or (j, i) in seen: continue
            a, b = rows[i], rows[j]
            if pred(a, b):
                seen.add((i, j)); out.append((a, b))
        return out
    same_voice_same_lang = sample(lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] == b["lang"])
    same_voice_diff_lang = sample(lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] != b["lang"])
    diff_voice_same_lang = sample(lambda a, b: a["voice_id"] != b["voice_id"] and a["lang"] == b["lang"])

    needed = sorted({r["wav_path"] for pair in (same_voice_same_lang + same_voice_diff_lang + diff_voice_same_lang) for r in pair})
    print(f"  {len(needed)} unique wavs needed")

    out = {}
    for enc_name, factory in [("wavlm_sv", _wavlm_sv_embedder),
                               ("ecapa_tdnn", _ecapa_embedder),
                               ("lase_r1", _lase_embedder)]:
        print(f"  embedding via {enc_name}...")
        embedder = factory()
        cache = {}
        t0 = time.time()
        for i, p in enumerate(needed):
            full = ROOT / p
            if not full.exists(): continue
            try:
                cache[p] = embedder(full)
            except Exception as e:
                print(f"    [{i}] FAILED {p}: {e}")
        print(f"  embedded {len(cache)}/{len(needed)} in {time.time()-t0:.0f}s")

        bucket_stats = {}
        for bucket_name, pairs in [("within_script", same_voice_same_lang),
                                    ("cross_script", same_voice_diff_lang),
                                    ("across_voice", diff_voice_same_lang)]:
            cosines = []
            for a, b in pairs:
                ea = cache.get(a["wav_path"]); eb = cache.get(b["wav_path"])
                if ea is None or eb is None: continue
                cosines.append(_cosine(ea, eb))
            if cosines:
                med, lo, hi = _bootstrap_ci(cosines)
                bucket_stats[bucket_name] = {
                    "n": len(cosines),
                    "median": round(med, 4),
                    "ci95_lo": round(lo, 4),
                    "ci95_hi": round(hi, 4),
                }
        # Compute gap and margin with their own bootstrap
        if all(b in bucket_stats for b in ("within_script", "cross_script", "across_voice")):
            # Recompute bootstrap on differences
            w_cos = [_cosine(cache[a["wav_path"]], cache[b["wav_path"]]) for a, b in same_voice_same_lang if cache.get(a["wav_path"]) is not None and cache.get(b["wav_path"]) is not None]
            c_cos = [_cosine(cache[a["wav_path"]], cache[b["wav_path"]]) for a, b in same_voice_diff_lang if cache.get(a["wav_path"]) is not None and cache.get(b["wav_path"]) is not None]
            f_cos = [_cosine(cache[a["wav_path"]], cache[b["wav_path"]]) for a, b in diff_voice_same_lang if cache.get(a["wav_path"]) is not None and cache.get(b["wav_path"]) is not None]
            gap_boot, margin_boot = [], []
            for _ in range(N_BOOTSTRAP):
                ws = np.random.choice(w_cos, size=len(w_cos), replace=True)
                cs = np.random.choice(c_cos, size=len(c_cos), replace=True)
                fs = np.random.choice(f_cos, size=len(f_cos), replace=True)
                gap_boot.append(np.median(ws) - np.median(cs))
                margin_boot.append(np.median(cs) - np.median(fs))
            bucket_stats["gap"] = {
                "median": round(float(np.median(gap_boot)), 4),
                "ci95_lo": round(float(np.percentile(gap_boot, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(gap_boot, 97.5)), 4),
            }
            bucket_stats["margin"] = {
                "median": round(float(np.median(margin_boot)), 4),
                "ci95_lo": round(float(np.percentile(margin_boot, 2.5)), 4),
                "ci95_hi": round(float(np.percentile(margin_boot, 97.5)), 4),
            }
        out[enc_name] = bucket_stats
        print(f"    {enc_name}: gap={bucket_stats.get('gap', {}).get('median', '?')} "
              f"[{bucket_stats.get('gap', {}).get('ci95_lo', '?')}, "
              f"{bucket_stats.get('gap', {}).get('ci95_hi', '?')}]  "
              f"margin={bucket_stats.get('margin', {}).get('median', '?')} "
              f"[{bucket_stats.get('margin', {}).get('ci95_lo', '?')}, "
              f"{bucket_stats.get('margin', {}).get('ci95_hi', '?')}]")
    return out


def main():
    summary = {}
    for name, path in [
        ("western", ROOT / "data/codeswitch_pairs_lase_heldout/manifest.jsonl"),
        ("indian",  ROOT / "data/codeswitch_pairs_lase_v2_indian/manifest.jsonl"),
    ]:
        summary[name] = _eval_corpus(name, path)

    out_path = ROOT / "data/codeswitch_pairs_lase_heldout/bootstrap_cis.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
