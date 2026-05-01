"""Synthesize the v2 voice-held-out corpus using 8 fresh Indian-accented
ElevenLabs voices. Same pipeline as paper/lase/codeswitch_pairs.py but with
a different voice catalog patched in.

Run::

    uv run python -m paper.lase.run_indian_voice_holdout --execute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 8 distinct Indian-accented EL voices (verified 2026-05-01 via /v1/voices).
# Different speakers from the 8 we used in v1 (Rachel, Drew, etc.).
INDIAN_VOICES: dict[str, str] = {
    "MonikaSogam":  "2zRM7PkgwBPiau2jvVXc",   # F, deep & natural, indian accent
    "Kartik":       "XPqjYvTqfyUQr09yCpCY",   # M, warm & expressive, indian
    "Chinmay":      "xnx6sPTtvU635ocDt2j7",   # M, calm energetic, indian formal
    "Raju":         "pzxut4zZz4GImZNlqQ3H",   # M, customer care, indian
    "Sagar":        "Qc0h5B5Mqs8oaH4sFZ9X",   # M, formal/polite/elegant, indian
    "Ziina":        "FaqthkZu1EWxXxUFbAfb",   # F, confident/clear, indian pro
    "NehaP":        "QTKSa2Iyv0yoxvXY2V8a",   # F, casual customer care, indian
    "Vishesh":      "Hq6EwBRAX1WbS8MuCZtT",   # M, calm, indian
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/codeswitch_pairs_lase_v2_indian")
    ap.add_argument("--source-manifest",
                    default="data/codeswitch_pairs_lase/source_heldout.jsonl",
                    help="Reuse the held-out transcripts (50/lang × 4 langs)")
    ap.add_argument("--pairs-per-voice", type=int, default=50)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    # Monkey-patch the voice catalog before importing the synth pipeline.
    import data.codeswitch_gen as cg
    original = cg.ELEVENLABS_MULTILINGUAL_VOICES.copy()
    cg.ELEVENLABS_MULTILINGUAL_VOICES.clear()
    cg.ELEVENLABS_MULTILINGUAL_VOICES.update(INDIAN_VOICES)
    print(f"[v2-indian] patched voice catalog: {list(cg.ELEVENLABS_MULTILINGUAL_VOICES.keys())}")

    # Now invoke the codeswitch_pairs runner with the patched catalog.
    sys.argv = [
        "codeswitch_pairs",
        "--out-dir", args.out_dir,
        "--n-voices", str(len(INDIAN_VOICES)),
        "--pairs-per-voice", str(args.pairs_per_voice),
        "--source-manifest", args.source_manifest,
    ]
    if args.execute:
        sys.argv.append("--execute")

    try:
        from paper.lase.codeswitch_pairs import run, _parse_args
        run(_parse_args())
    finally:
        cg.ELEVENLABS_MULTILINGUAL_VOICES.clear()
        cg.ELEVENLABS_MULTILINGUAL_VOICES.update(original)
    return 0


if __name__ == "__main__":
    sys.exit(main())
