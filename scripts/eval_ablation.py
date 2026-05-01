"""ECAPA+GRL ablation eval — load the ablation checkpoint, run the same
3-distribution analysis on both held-out corpora, write a 4th-encoder column
into the paper headline data.

The ablation result tells us whether LASE's improvement came from the GRL
training objective (in which case ECAPA+GRL should also improve over
ECAPA-vanilla) or from the WavLM backbone choice (in which case ECAPA+GRL
should look like ECAPA-vanilla).

Output: data/codeswitch_pairs_lase_heldout/ablation_ecapa_secs.json
        + data/codeswitch_pairs_lase_v2_indian/ablation_ecapa_secs.json
"""
from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABLATION_CKPT_LOCAL = ROOT / "data/codeswitch_pairs_lase/ablation_ecapa_last.pt"
SAMPLE_PAIRS = 200

random.seed(1337)
np.random.seed(1337)


def _download_ablation_ckpt():
    if ABLATION_CKPT_LOCAL.exists():
        return ABLATION_CKPT_LOCAL
    ABLATION_CKPT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ablation-eval] downloading ablation_ecapa/last.pt from praxy-lase-runs volume…")
    res = subprocess.run(
        ["modal", "volume", "get", "praxy-lase-runs",
         "/ablation_ecapa/last.pt", str(ABLATION_CKPT_LOCAL), "--force"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"modal volume get failed: {res.stderr}")
    print(f"[ablation-eval] downloaded to {ABLATION_CKPT_LOCAL}")
    return ABLATION_CKPT_LOCAL


def _build_ablation_encoder(ckpt_path):
    import torch
    from models.novel.lase import LambdaSchedule, LASE, EcapaSpeakerEncoder
    backbone = EcapaSpeakerEncoder(embedding_dim=256, freeze_backbone=True)
    schedule = LambdaSchedule(warmup_steps=200, ramp_steps=500, peak=0.1)
    model = LASE(backbone=backbone, embedding_dim=256, n_languages=4,
                 lambda_schedule=schedule)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _embed(audio_path, model):
    import torch
    import torchaudio
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav[0][:32_000]
    if wav.shape[0] < 32_000:
        wav = torch.nn.functional.pad(wav, (0, 32_000 - wav.shape[0]))
    with torch.inference_mode():
        return model(wav.unsqueeze(0))["embedding"][0].cpu().numpy()


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _sample_pairs(rows, pred):
    n = len(rows); out = []; seen = set(); attempts = 0
    while len(out) < SAMPLE_PAIRS and attempts < SAMPLE_PAIRS * 50:
        attempts += 1
        i, j = random.sample(range(n), 2)
        if (i, j) in seen or (j, i) in seen: continue
        a, b = rows[i], rows[j]
        if pred(a, b):
            seen.add((i, j)); out.append((a, b))
    return out


def _stats(label, pairs, cache):
    cosines = []
    for a, b in pairs:
        ea = cache.get(a["wav_path"]); eb = cache.get(b["wav_path"])
        if ea is None or eb is None: continue
        cosines.append(_cosine(ea, eb))
    if not cosines:
        return {"label": label, "n": 0}
    cosines.sort()
    return {
        "label": label, "n": len(cosines),
        "median": round(float(np.median(cosines)), 4),
        "p25": round(cosines[len(cosines)//4], 4),
        "p75": round(cosines[len(cosines)*3//4], 4),
        "min": round(min(cosines), 4),
        "max": round(max(cosines), 4),
    }


def _eval(corpus_name: str, manifest_path: Path, model):
    print(f"\n=== {corpus_name} ===")
    rows = [json.loads(ln) for ln in manifest_path.read_text().splitlines() if ln.strip()]
    same_voice_same_lang = _sample_pairs(rows, lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] == b["lang"])
    same_voice_diff_lang = _sample_pairs(rows, lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] != b["lang"])
    diff_voice_same_lang = _sample_pairs(rows, lambda a, b: a["voice_id"] != b["voice_id"] and a["lang"] == b["lang"])
    needed = sorted({r["wav_path"] for pair in (same_voice_same_lang + same_voice_diff_lang + diff_voice_same_lang) for r in pair})
    print(f"  {len(rows)} pairs, {len(needed)} unique wavs to embed")

    cache = {}
    t0 = time.time()
    for i, p in enumerate(needed):
        full = ROOT / p
        if not full.exists(): continue
        try:
            cache[p] = _embed(full, model)
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")
    print(f"  embedded {len(cache)}/{len(needed)} in {time.time()-t0:.0f}s")

    s = {
        "within_script": _stats("within", same_voice_same_lang, cache),
        "cross_script":  _stats("cross", same_voice_diff_lang, cache),
        "across_voice":  _stats("across", diff_voice_same_lang, cache),
    }
    w = s["within_script"].get("median", 0)
    c = s["cross_script"].get("median", 0)
    f = s["across_voice"].get("median", 0)
    s["gap"] = round(w - c, 4)
    s["margin"] = round(c - f, 4)
    print(f"  within={w}  cross={c}  floor={f}  gap={s['gap']:+.4f}  margin={s['margin']:+.4f}")
    return s


def main():
    ckpt = _download_ablation_ckpt()
    model = _build_ablation_encoder(ckpt)

    summary = {}
    for name, path in [
        ("western", ROOT / "data/codeswitch_pairs_lase_heldout/manifest.jsonl"),
        ("indian",  ROOT / "data/codeswitch_pairs_lase_v2_indian/manifest.jsonl"),
    ]:
        summary[name] = _eval(name, path, model)

    out = ROOT / "data/codeswitch_pairs_lase_heldout/ablation_ecapa_secs.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")

    print("\n=== ABLATION TABLE: ECAPA+GRL on both corpora ===")
    print(f"{'corpus':10s} {'within':>8s} {'cross':>8s} {'floor':>8s} {'gap':>8s} {'margin':>8s}")
    for k in ("western", "indian"):
        s = summary[k]
        print(f"{k:10s} {s['within_script']['median']:>8} {s['cross_script']['median']:>8} "
              f"{s['across_voice']['median']:>8} {s['gap']:>8} {s['margin']:>8}")


if __name__ == "__main__":
    main()
