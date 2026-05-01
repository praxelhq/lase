"""LASE multi-encoder eval: compare LASE r1 vs SOTA speaker encoders on the
held-out cross-script identity test.

Encoders measured:
    1. WavLM-base-plus-sv  (our LASE backbone in eval-mode — baseline)
    2. ECAPA-TDNN          (SpeechBrain — industry-standard speaker verification)
    3. LASE r1             (our trained model)
    4. (optional) OpenVoice tone-color extractor — added later if available

For each encoder, runs the same 3-distribution analysis on the held-out
1043-pair manifest and reports:
    within-script median (upper bound)
    cross-script median  (the test)
    across-voice median  (noise floor)
    gap (within - cross)
    cross-vs-floor margin

Output: data/codeswitch_pairs_lase_heldout/multi_encoder_secs.json
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/codeswitch_pairs_lase_heldout/manifest.jsonl"
OUT = ROOT / "data/codeswitch_pairs_lase_heldout/multi_encoder_secs.json"
LASE_CKPT = ROOT / "data/codeswitch_pairs_lase/r1_last.pt"
SAMPLE_PAIRS = 200

random.seed(1337)


def _load_manifest():
    rows = []
    for ln in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rows.append(json.loads(ln))
    return rows


def _sample_pairs(rows, pred, max_pairs):
    n = len(rows)
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 50:
        attempts += 1
        i, j = random.sample(range(n), 2)
        if (i, j) in seen or (j, i) in seen:
            continue
        a, b = rows[i], rows[j]
        if pred(a, b):
            seen.add((i, j))
            pairs.append((a, b))
    return pairs


def _stats(label, pairs, cache, cosine_fn):
    cosines = []
    for a, b in pairs:
        ea = cache.get(a["wav_path"])
        eb = cache.get(b["wav_path"])
        if ea is None or eb is None:
            continue
        cosines.append(cosine_fn(ea, eb))
    if not cosines:
        return {"label": label, "n": 0}
    cosines.sort()
    return {
        "label": label,
        "n": len(cosines),
        "median": round(float(np.median(cosines)), 4),
        "mean": round(float(np.mean(cosines)), 4),
        "p25": round(cosines[len(cosines)//4], 4),
        "p75": round(cosines[len(cosines)*3//4], 4),
        "min": round(min(cosines), 4),
        "max": round(max(cosines), 4),
    }


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _wavlm_sv_embedder():
    """Off-the-shelf WavLM-base-plus-sv embedding (the LASE backbone in eval mode)."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    proc = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    model.eval()

    def embed(audio_path):
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        inputs = proc(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            out = model(**inputs)
        return out.embeddings[0].numpy()
    return embed


def _ecapa_embedder():
    """SpeechBrain ECAPA-TDNN speaker embedding."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa-voxceleb",
    )

    def embed(audio_path):
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        with torch.inference_mode():
            emb = model.encode_batch(wav)
        return emb.squeeze().numpy()
    return embed


def _lase_embedder(ckpt_path):
    """LASE r1 trained encoder."""
    import torch
    from models.novel.lase import LambdaSchedule, LASE, WavLMSpeakerEncoder
    backbone = WavLMSpeakerEncoder(model_name="microsoft/wavlm-base-plus",
                                    embedding_dim=256, freeze_backbone=True)
    schedule = LambdaSchedule(warmup_steps=200, ramp_steps=500, peak=0.1)
    model = LASE(backbone=backbone, embedding_dim=256, n_languages=4,
                 lambda_schedule=schedule)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    def embed(audio_path):
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav[0][:32_000]
        if wav.shape[0] < 32_000:
            wav = torch.nn.functional.pad(wav, (0, 32_000 - wav.shape[0]))
        x = wav.unsqueeze(0)
        with torch.inference_mode():
            out = model(x)
        return out["embedding"][0].cpu().numpy()
    return embed


def _embed_set(embedder_fn, wav_paths, label):
    cache = {}
    t0 = time.time()
    for i, p in enumerate(sorted(wav_paths)):
        full = ROOT / p
        if not full.exists():
            continue
        try:
            cache[p] = embedder_fn(full)
        except Exception as e:
            print(f"  [{label}][{i}] FAILED {p}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  [{label}][{i+1}/{len(wav_paths)}] elapsed {time.time()-t0:.0f}s")
    print(f"[{label}] embedded {len(cache)} wavs in {time.time()-t0:.0f}s")
    return cache


def main():
    rows = _load_manifest()
    print(f"loaded {len(rows)} pairs")

    same_voice_same_lang = _sample_pairs(
        rows,
        lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] == b["lang"],
        SAMPLE_PAIRS,
    )
    same_voice_diff_lang = _sample_pairs(
        rows,
        lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] != b["lang"],
        SAMPLE_PAIRS,
    )
    diff_voice_same_lang = _sample_pairs(
        rows,
        lambda a, b: a["voice_id"] != b["voice_id"] and a["lang"] == b["lang"],
        SAMPLE_PAIRS,
    )
    needed = set()
    for a, b in same_voice_same_lang + same_voice_diff_lang + diff_voice_same_lang:
        needed.add(a["wav_path"])
        needed.add(b["wav_path"])
    print(f"{len(needed)} unique wavs needed")

    ENCODERS = [
        ("wavlm_sv",   _wavlm_sv_embedder),
        ("ecapa_tdnn", _ecapa_embedder),
        ("lase_r1",    lambda: _lase_embedder(LASE_CKPT)),
    ]

    summary = {"n_rows": len(rows), "n_unique_wavs": len(needed), "encoders": {}}
    for name, factory in ENCODERS:
        print(f"\n=== {name} ===")
        try:
            embedder = factory()
            cache = _embed_set(embedder, needed, name)
            stats = {
                "within_script": _stats("within", same_voice_same_lang, cache, _cosine),
                "cross_script":  _stats("cross", same_voice_diff_lang, cache, _cosine),
                "across_voice":  _stats("across", diff_voice_same_lang, cache, _cosine),
            }
            w = stats["within_script"].get("median", 0)
            c = stats["cross_script"].get("median", 0)
            a_ = stats["across_voice"].get("median", 0)
            stats["gap_within_minus_cross"] = round(w - c, 4)
            stats["cross_vs_floor_margin"] = round(c - a_, 4)
            print(f"  within={w}  cross={c}  floor={a_}  gap={w-c:+.4f}  margin={c-a_:+.4f}")
            summary["encoders"][name] = stats
        except Exception as e:
            print(f"  FAILED for {name}: {e}")
            summary["encoders"][name] = {"error": str(e)}

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    print("\n=== TABLE ===")
    print(f"{'encoder':15s}  {'within':>8s}  {'cross':>8s}  {'floor':>8s}  {'gap':>8s}  {'margin':>8s}")
    for name in ["wavlm_sv", "ecapa_tdnn", "lase_r1"]:
        s = summary["encoders"].get(name, {})
        if "error" in s:
            print(f"{name:15s}  ERROR: {s['error']}")
            continue
        w = s.get("within_script", {}).get("median", "?")
        c = s.get("cross_script", {}).get("median", "?")
        f = s.get("across_voice", {}).get("median", "?")
        g = s.get("gap_within_minus_cross", "?")
        m = s.get("cross_vs_floor_margin", "?")
        print(f"{name:15s}  {w:>8}  {c:>8}  {f:>8}  {g:>8}  {m:>8}")


if __name__ == "__main__":
    main()
