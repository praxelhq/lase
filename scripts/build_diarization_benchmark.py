"""Build a synthetic multi-speaker code-switching diarization benchmark.

Uses the held-out 1043-pair LASE corpus (8 voices × 4 langs) to construct
multi-speaker conversations with KNOWN speaker boundaries:

  Conversation = N segments, each segment = [voice_id, lang, wav_path, start, end].

Each conversation contains 2-4 speakers and may switch languages within a
single speaker. This is exactly the failure mode LASE is designed to fix:
diarization that doesn't hallucinate a "new speaker" when the script changes.

Output: data/codeswitch_pairs_lase_heldout/diarization/
  conversations.jsonl  — 50 conversations, ~30-60s each
  audio/conv_NNN.wav  — concatenated audio
  rttm/conv_NNN.rttm  — ground-truth speaker labels (RTTM format for pyannote)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HELDOUT_MANIFEST = ROOT / "data/codeswitch_pairs_lase_heldout/manifest.jsonl"
OUT_DIR = ROOT / "data/codeswitch_pairs_lase_heldout/diarization"

random.seed(2026)


def _load_corpus():
    by_voice_lang: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ln in HELDOUT_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        r = json.loads(ln)
        by_voice_lang[(r["voice_id"], r["lang"])].append(r)
    return by_voice_lang


def _build_conversation(by_voice_lang, conv_id: int, n_speakers: int = 3,
                        n_segments: int = 8) -> dict:
    """Pick N speakers, generate a conversation with N_SEGMENTS turn-takings.
    Each speaker takes 2-3 segments, possibly in different languages."""
    voices = list({k[0] for k in by_voice_lang.keys()})
    speakers = random.sample(voices, n_speakers)
    langs = sorted({k[1] for k in by_voice_lang.keys()})

    segments = []
    last_speaker = None
    for _ in range(n_segments):
        # Pick a speaker different from the last (round-robin-ish)
        candidates = [s for s in speakers if s != last_speaker] or speakers
        speaker = random.choice(candidates)
        lang = random.choice(langs)
        candidates_clips = by_voice_lang.get((speaker, lang), [])
        if not candidates_clips:
            # Fall back to any lang for this speaker
            available_langs = [l for l in langs
                               if (speaker, l) in by_voice_lang and by_voice_lang[(speaker, l)]]
            if not available_langs:
                continue
            lang = random.choice(available_langs)
            candidates_clips = by_voice_lang[(speaker, lang)]
        clip = random.choice(candidates_clips)
        segments.append({
            "voice_id": speaker,
            "lang": lang,
            "wav_path": clip["wav_path"],
            "text": clip.get("text", ""),
        })
        last_speaker = speaker

    return {
        "conv_id": f"conv_{conv_id:03d}",
        "n_speakers": n_speakers,
        "segments": segments,
    }


def _concatenate_audio(segments: list[dict], out_audio: Path,
                        out_rttm: Path, gap_s: float = 0.3):
    """Concatenate segment wavs with small gaps between turns. Writes:
       - out_audio: 16 kHz mono WAV.
       - out_rttm: RTTM file with ground-truth speaker labels.
    """
    import soundfile as sf
    samples = []
    rttm_lines = []
    cursor = 0.0  # seconds
    sr = 16_000
    for seg in segments:
        wav_path = ROOT / seg["wav_path"]
        if not wav_path.exists():
            continue
        wav, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            # Quick resample via numpy (kept simple — pyannote's pipeline will
            # resample if needed)
            import librosa
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        seg_dur = len(wav) / sr
        # RTTM: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        speaker_id = seg["voice_id"][:8]  # short id
        rttm_lines.append(
            f"SPEAKER {out_audio.stem} 1 {cursor:.3f} {seg_dur:.3f} <NA> <NA> {speaker_id} <NA> <NA>"
        )
        samples.append(wav)
        # gap of silence
        samples.append(np.zeros(int(gap_s * sr), dtype="float32"))
        cursor += seg_dur + gap_s

    if not samples:
        return None
    audio = np.concatenate(samples)
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_audio), audio, sr, subtype="PCM_16")
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    out_rttm.write_text("\n".join(rttm_lines) + "\n")
    return cursor  # total duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conversations", type=int, default=50)
    ap.add_argument("--n-speakers-min", type=int, default=2)
    ap.add_argument("--n-speakers-max", type=int, default=4)
    ap.add_argument("--n-segments-min", type=int, default=6)
    ap.add_argument("--n-segments-max", type=int, default=10)
    args = ap.parse_args()

    by_voice_lang = _load_corpus()
    print(f"loaded held-out corpus: {len(by_voice_lang)} (voice, lang) buckets")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audio").mkdir(exist_ok=True)
    (OUT_DIR / "rttm").mkdir(exist_ok=True)

    convs = []
    for i in range(args.n_conversations):
        n_spk = random.randint(args.n_speakers_min, args.n_speakers_max)
        n_seg = random.randint(args.n_segments_min, args.n_segments_max)
        conv = _build_conversation(by_voice_lang, i, n_spk, n_seg)
        out_audio = OUT_DIR / "audio" / f"{conv['conv_id']}.wav"
        out_rttm = OUT_DIR / "rttm" / f"{conv['conv_id']}.rttm"
        dur = _concatenate_audio(conv["segments"], out_audio, out_rttm)
        if dur is None:
            print(f"  conv {i}: skipped (no segments)")
            continue
        conv["duration_s"] = round(dur, 2)
        conv["audio_path"] = str(out_audio.relative_to(ROOT))
        conv["rttm_path"] = str(out_rttm.relative_to(ROOT))
        convs.append(conv)
        if (i + 1) % 10 == 0:
            print(f"  built {i+1}/{args.n_conversations}")

    manifest_path = OUT_DIR / "conversations.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in convs) + "\n"
    )

    print(f"\nbuilt {len(convs)} conversations")
    print(f"total duration: {sum(c['duration_s'] for c in convs):.0f}s "
          f"({sum(c['duration_s'] for c in convs)/60:.1f} min)")
    print(f"avg n_speakers: {np.mean([c['n_speakers'] for c in convs]):.1f}")
    print(f"avg n_segments: {np.mean([len(c['segments']) for c in convs]):.1f}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
