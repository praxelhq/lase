"""LASE pre-flight: prove the cross-script identity-drift gap exists.

Computes three SECS distributions on the 1118-pair corpus we just generated:
  1. WITHIN-SPEAKER WITHIN-SCRIPT (same voice, same lang, ≠ sentences)
  2. WITHIN-SPEAKER CROSS-SCRIPT  (same voice, ≠ lang)              ← test
  3. ACROSS-SPEAKER WITHIN-SCRIPT (≠ voice, same lang)              ← lower bound

Verdict logic:
  - within-script median ≈ 1 = embedder is consistent for same-speaker
  - across-speaker median ≈ 0.3-0.6 = embedder discriminates speakers
  - cross-script must land **strictly between** the two for LASE to have a story:
      * if cross-script ≈ within-script → no drift, no paper
      * if cross-script ≈ across-speaker → entanglement extreme, easier paper
      * if cross-script in between → expected, paper has legs

Runs on local CPU using cached WavLM-base-plus-sv (~95M params, ~30 min for the
sampled pair set). Reuses paper/lase/codeswitch_pairs.py:_embed.

Output: data/codeswitch_pairs_lase/preflight_secs_gap.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/codeswitch_pairs_lase/manifest.jsonl"
OUT = ROOT / "data/codeswitch_pairs_lase/preflight_secs_gap.json"
SAMPLE_PAIRS_PER_BUCKET = 200  # cap pair count per bucket to keep runtime sane

random.seed(1337)


def _load_manifest() -> list[dict]:
    rows = []
    for ln in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rows.append(json.loads(ln))
    return rows


def _sample_pairs(rows: list[dict], pred, max_pairs: int) -> list[tuple[dict, dict]]:
    """Yield up to max_pairs random (a, b) pairs from rows where pred(a, b) is True."""
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


def _embed_one(audio_path: Path, model, processor, device):
    """Lazy-import torchaudio + run WavLM forward → 1024-d speaker embedding."""
    import torch
    import torchaudio
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    inputs = processor(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
    with torch.inference_mode():
        out = model(**{k: v.to(device) for k, v in inputs.items()})
    return out.embeddings[0].cpu().numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    print(f"[lase-preflight] loading manifest from {MANIFEST.relative_to(ROOT)}")
    rows = _load_manifest()
    print(f"[lase-preflight] {len(rows)} pairs loaded")

    # Group rows by (voice_id, lang) for sanity
    by_voice_lang: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_voice_lang[(r["voice_id"], r["lang"])].append(r)
    print(f"[lase-preflight] (voice × lang) buckets: {len(by_voice_lang)}")
    voices = sorted({r["voice_id"] for r in rows})
    langs = sorted({r["lang"] for r in rows})
    print(f"[lase-preflight] voices={len(voices)} langs={langs}")

    # Sample pair lists
    print("[lase-preflight] sampling pair triplets...")
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
    print(f"  within-script:  {len(same_voice_same_lang)} pairs")
    print(f"  cross-script:   {len(same_voice_diff_lang)} pairs")
    print(f"  across-speaker: {len(diff_voice_same_lang)} pairs")

    # Load WavLM
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lase-preflight] loading microsoft/wavlm-base-plus-sv on {device}")
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
    model.eval()

    # Embed each unique audio path once
    print("[lase-preflight] embedding unique audios...")
    unique_paths = sorted({r["wav_path"] for r in rows
                           if any(r in (a, b) for a, b in
                                  same_voice_same_lang + same_voice_diff_lang + diff_voice_same_lang)})
    # Fallback: just embed every path that appears in any sampled pair
    needed = set()
    for a, b in same_voice_same_lang + same_voice_diff_lang + diff_voice_same_lang:
        needed.add(a["wav_path"])
        needed.add(b["wav_path"])
    print(f"  {len(needed)} unique wavs to embed")
    cache: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, p in enumerate(sorted(needed)):
        full = ROOT / p
        if not full.exists():
            continue
        try:
            cache[p] = _embed_one(full, model, processor, device)
        except Exception as e:
            print(f"  [{i}] FAILED {p}: {e}")
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(needed)}] {rate:.1f}/s, eta {(len(needed)-i-1)/rate/60:.1f} min")
    print(f"[lase-preflight] embedded {len(cache)} wavs in {time.time()-t0:.0f}s")

    def _stats(label: str, pairs: list[tuple[dict, dict]]) -> dict:
        cosines = []
        for a, b in pairs:
            ea, eb = cache.get(a["wav_path"]), cache.get(b["wav_path"])
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

    summary = {
        "n_rows": len(rows),
        "n_voices": len(voices),
        "langs": langs,
        "sampled": {
            "within_script": _stats("within-speaker within-script", same_voice_same_lang),
            "cross_script":  _stats("within-speaker CROSS-script", same_voice_diff_lang),
            "across_voice":  _stats("across-speaker within-script", diff_voice_same_lang),
        },
    }

    # Verdict
    w = summary["sampled"]["within_script"]
    c = summary["sampled"]["cross_script"]
    a = summary["sampled"]["across_voice"]
    if w.get("n") and c.get("n") and a.get("n"):
        gap_within = w["median"] - c["median"]      # how much identity drops cross-script
        gap_floor = c["median"] - a["median"]      # how much room above noise floor
        verdict = ""
        if gap_within < 0.01:
            verdict = "NO_DRIFT — no story; cross-script identity already preserved by embedder. LASE has no headline."
        elif gap_within > 0.30:
            verdict = "SEVERE_DRIFT — strong story; cross-script collapses near speaker noise floor."
        elif gap_within >= 0.05 and gap_floor >= 0.10:
            verdict = "MODERATE_DRIFT — paper has legs; LASE Phase 2 worth $30 spend."
        else:
            verdict = "MARGINAL — gap exists but small; consider scoping LASE as a v2 followup."
        summary["verdict"] = verdict
        summary["gap_within_minus_cross"] = round(gap_within, 4)
        summary["gap_cross_minus_floor"] = round(gap_floor, 4)
        print(f"\n=== VERDICT ===")
        print(f"within-script median: {w['median']}")
        print(f"cross-script median:  {c['median']}  (drop: {gap_within:+.4f})")
        print(f"across-voice median:  {a['median']}  (floor)")
        print(f"-> {verdict}")
    else:
        summary["verdict"] = "INSUFFICIENT_DATA"

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[lase-preflight] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
