"""LASE post-training eval: re-measure the cross-script SECS gap with the
trained encoder, compare to the pre-flight baseline.

Pre-flight (preflight_secs_gap.py) measured three distributions using
microsoft/wavlm-base-plus-sv (off-the-shelf):
    within-script  median 0.928 (upper bound)
    cross-script   median 0.829 (the test)
    across-voice   median 0.642 (noise floor)

This script does the same three-distribution measurement but uses the
LASE-trained encoder instead. The win condition is:
    - cross-script median climbs toward within-script (gap closed)
    - across-voice stays low (speaker discrimination preserved)

Loads checkpoint from data/codeswitch_pairs_lase/r1_last.pt (downloaded
from the praxy-lase-runs Modal volume) or any path passed via --ckpt.

Usage::

    # Auto-download r1/last.pt from praxy-lase-runs volume + run eval
    uv run python -m paper.lase.eval_secs_gap

    # Specify a different ckpt
    uv run python -m paper.lase.eval_secs_gap \\
        --ckpt /path/to/step_500.pt

Output: data/codeswitch_pairs_lase/post_train_secs_gap.json
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/codeswitch_pairs_lase/manifest.jsonl"
DEFAULT_CKPT_LOCAL = ROOT / "data/codeswitch_pairs_lase/r1_last.pt"
OUT = ROOT / "data/codeswitch_pairs_lase/post_train_secs_gap.json"
PREFLIGHT = ROOT / "data/codeswitch_pairs_lase/preflight_secs_gap.json"
SAMPLE_PAIRS_PER_BUCKET = 200

random.seed(1337)


def _download_ckpt_if_missing(local_path: Path) -> Path:
    if local_path.exists():
        return local_path
    print(f"[lase-eval] ckpt not at {local_path}, downloading from Modal volume…")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["modal", "volume", "get", "praxy-lase-runs",
         "/r1/last.pt", str(local_path), "--force"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"modal volume get failed: {res.stderr}")
    print(f"[lase-eval] downloaded to {local_path}")
    return local_path


def _load_manifest() -> list[dict]:
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


def _build_lase_encoder(ckpt_path: Path):
    """Construct the LASE wrapper, load the r1 weights, return a callable
    that maps a single waveform tensor (B, T) → speaker embedding (B, 256)."""
    import torch
    from models.novel.lase import LambdaSchedule, LASE, WavLMSpeakerEncoder

    print(f"[lase-eval] building LASE wrapper, loading ckpt {ckpt_path}")
    backbone = WavLMSpeakerEncoder(
        model_name="microsoft/wavlm-base-plus",
        embedding_dim=256,
        freeze_backbone=True,
    )
    schedule = LambdaSchedule(warmup_steps=200, ramp_steps=500, peak=0.1)
    model = LASE(backbone=backbone, embedding_dim=256, n_languages=4,
                 lambda_schedule=schedule)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _embed_one(wav_path: Path, model):
    import torch
    import torchaudio
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    # Crop / pad to 2 s like training
    wav = wav[0][:32_000]
    if wav.shape[0] < 32_000:
        wav = torch.nn.functional.pad(wav, (0, 32_000 - wav.shape[0]))
    x = wav.unsqueeze(0)  # (1, T)
    with torch.inference_mode():
        out = model(x)
    return out["embedding"][0].cpu().numpy()


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _stats(label, pairs, cache):
    cosines = []
    for a, b in pairs:
        ea = cache.get(a["wav_path"])
        eb = cache.get(b["wav_path"])
        if ea is None or eb is None:
            continue
        cosines.append(_cosine(ea, eb))
    if not cosines:
        return {"label": label, "n": 0}
    cosines.sort()
    return {
        "label": label,
        "n": len(cosines),
        "median": round(float(np.median(cosines)), 4),
        "mean": round(float(np.mean(cosines)), 4),
        "p10": round(cosines[len(cosines)//10], 4),
        "p25": round(cosines[len(cosines)//4], 4),
        "p75": round(cosines[len(cosines)*3//4], 4),
        "p90": round(cosines[len(cosines)*9//10], 4),
        "min": round(min(cosines), 4),
        "max": round(max(cosines), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_LOCAL)
    args = ap.parse_args()

    ckpt_path = _download_ckpt_if_missing(args.ckpt)

    rows = _load_manifest()
    print(f"[lase-eval] {len(rows)} pairs in manifest")

    same_voice_same_lang = _sample_pairs(
        rows,
        lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] == b["lang"],
        SAMPLE_PAIRS_PER_BUCKET,
    )
    same_voice_diff_lang = _sample_pairs(
        rows,
        lambda a, b: a["voice_id"] == b["voice_id"] and a["lang"] != b["lang"],
        SAMPLE_PAIRS_PER_BUCKET,
    )
    diff_voice_same_lang = _sample_pairs(
        rows,
        lambda a, b: a["voice_id"] != b["voice_id"] and a["lang"] == b["lang"],
        SAMPLE_PAIRS_PER_BUCKET,
    )

    needed = set()
    for a, b in same_voice_same_lang + same_voice_diff_lang + diff_voice_same_lang:
        needed.add(a["wav_path"])
        needed.add(b["wav_path"])
    print(f"[lase-eval] {len(needed)} unique wavs to embed")

    model = _build_lase_encoder(ckpt_path)

    cache = {}
    t0 = time.time()
    for i, p in enumerate(sorted(needed)):
        full = ROOT / p
        if not full.exists():
            continue
        try:
            cache[p] = _embed_one(full, model)
        except Exception as e:
            print(f"  [{i}] FAILED {p}: {e}")
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(needed)}] {rate:.1f}/s  eta {(len(needed)-i-1)/rate/60:.1f} min")
    print(f"[lase-eval] embedded {len(cache)} wavs in {time.time()-t0:.0f}s")

    summary = {
        "ckpt": str(ckpt_path),
        "n_rows": len(rows),
        "post_train": {
            "within_script": _stats("within-speaker within-script", same_voice_same_lang, cache),
            "cross_script":  _stats("within-speaker CROSS-script", same_voice_diff_lang, cache),
            "across_voice":  _stats("across-speaker within-script", diff_voice_same_lang, cache),
        },
    }

    # Compare to pre-flight if available
    if PREFLIGHT.exists():
        pf = json.loads(PREFLIGHT.read_text())
        pre = pf.get("sampled", {})
        summary["pre_train"] = {
            "within_script": pre.get("within_script", {}).get("median"),
            "cross_script":  pre.get("cross_script", {}).get("median"),
            "across_voice":  pre.get("across_voice", {}).get("median"),
        }
        post_w = summary["post_train"]["within_script"]["median"]
        post_c = summary["post_train"]["cross_script"]["median"]
        post_a = summary["post_train"]["across_voice"]["median"]
        summary["delta_cross_minus_pre"] = round(
            post_c - (pre.get("cross_script", {}).get("median") or 0), 4
        )
        summary["post_gap_within_minus_cross"] = round(post_w - post_c, 4)
        summary["pre_gap_within_minus_cross"] = round(
            (pre.get("within_script", {}).get("median") or 0)
            - (pre.get("cross_script", {}).get("median") or 0), 4
        )

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== POST-TRAINING SECS DISTRIBUTION ===")
    print(f"within-script median: {summary['post_train']['within_script'].get('median')}")
    print(f"cross-script median:  {summary['post_train']['cross_script'].get('median')}")
    print(f"across-voice median:  {summary['post_train']['across_voice'].get('median')}")
    if "pre_gap_within_minus_cross" in summary:
        print(f"\npre-train  gap (within - cross): {summary['pre_gap_within_minus_cross']:+.4f}")
        print(f"post-train gap (within - cross): {summary['post_gap_within_minus_cross']:+.4f}")
        gap_closed = summary['pre_gap_within_minus_cross'] - summary['post_gap_within_minus_cross']
        print(f"gap closed: {gap_closed:+.4f}  ({gap_closed/summary['pre_gap_within_minus_cross']*100:+.1f}% relative)")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
