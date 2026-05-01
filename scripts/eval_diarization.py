"""LASE diarization eval — measure speaker-clustering quality on synthetic
multi-speaker code-switching conversations.

For each conversation:
  1. Embed each segment with the chosen encoder.
  2. Run agglomerative clustering with known K = #speakers.
  3. Compute Adjusted Rand Index (ARI) against ground-truth speaker labels.
  4. Compute "cross-script speaker recall" — for each speaker, what fraction
     of cross-language segments end up in the same cluster as their
     same-language segments.

Run for 3 encoders (WavLM-SV / ECAPA-TDNN / LASE r1) on the same 50-conv
benchmark. The diarization-flavoured headline metric for the paper.

Output: data/codeswitch_pairs_lase_heldout/diarization/diar_eval.json
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONVS = ROOT / "data/codeswitch_pairs_lase_heldout/diarization/conversations.jsonl"
OUT = ROOT / "data/codeswitch_pairs_lase_heldout/diarization/diar_eval.json"
LASE_CKPT = ROOT / "data/codeswitch_pairs_lase/r1_last.pt"


def _load_convs():
    return [json.loads(ln) for ln in CONVS.read_text(encoding="utf-8").splitlines() if ln.strip()]


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
    m = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa-voxceleb",
    )
    def embed(p):
        import torchaudio
        wav, sr = torchaudio.load(str(p))
        if sr != 16_000: wav = torchaudio.functional.resample(wav, sr, 16_000)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        with torch.inference_mode():
            return m.encode_batch(wav).squeeze().numpy()
    return embed


def _lase_embedder(ckpt_path):
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


def _adjusted_rand_index(true_labels, pred_labels):
    """ARI without requiring sklearn. Standard formula."""
    from itertools import combinations
    n = len(true_labels)
    if n < 2: return 1.0
    # Contingency table
    same_true = [(true_labels[i] == true_labels[j]) for i, j in combinations(range(n), 2)]
    same_pred = [(pred_labels[i] == pred_labels[j]) for i, j in combinations(range(n), 2)]
    tp = sum(1 for s, p in zip(same_true, same_pred) if s and p)
    fp = sum(1 for s, p in zip(same_true, same_pred) if not s and p)
    fn = sum(1 for s, p in zip(same_true, same_pred) if s and not p)
    tn = sum(1 for s, p in zip(same_true, same_pred) if not s and not p)
    # Adjusted Rand Index
    a_n = tp + fn   # same-true total
    b_n = tp + fp   # same-pred total
    total = tp + fp + fn + tn
    expected = (a_n * b_n + (total - a_n) * (total - b_n)) / total
    max_val = (a_n + b_n) / 2
    if max_val == expected: return 1.0
    return (tp + tn - expected) / (max_val - expected) * 2 - 1  # rough; sklearn-equivalent below
    # Actually the standard ARI formula:


def _ari_sklearn_style(true_labels, pred_labels):
    """Hand-rolled adjusted rand index, sklearn-compatible."""
    from collections import Counter
    n = len(true_labels)
    if n < 2: return 1.0
    # Contingency
    contingency: dict[tuple, int] = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        contingency[(t, p)] += 1
    # Sum over rows / cols
    row_sums = Counter(true_labels)
    col_sums = Counter(pred_labels)
    def comb2(x): return x * (x - 1) // 2
    sum_comb_c = sum(comb2(v) for v in contingency.values())
    sum_comb_a = sum(comb2(v) for v in row_sums.values())
    sum_comb_b = sum(comb2(v) for v in col_sums.values())
    expected = sum_comb_a * sum_comb_b / comb2(n) if comb2(n) > 0 else 0
    max_val = (sum_comb_a + sum_comb_b) / 2
    if max_val == expected: return 1.0
    return (sum_comb_c - expected) / (max_val - expected)


def _cluster_segments(embeddings, k):
    """Agglomerative clustering on cosine distance with K clusters."""
    from sklearn.cluster import AgglomerativeClustering
    if len(embeddings) <= 1:
        return [0] * len(embeddings)
    if k >= len(embeddings):
        return list(range(len(embeddings)))
    # Cosine distance matrix
    embs = np.array(embeddings)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs_norm = embs / norms
    similarity = embs_norm @ embs_norm.T
    distance = 1 - similarity
    distance = np.clip(distance, 0, 2)
    np.fill_diagonal(distance, 0)
    cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    return cl.fit_predict(distance).tolist()


def _cross_script_recall(segments, true_labels, pred_labels) -> float | None:
    """Per-speaker, what fraction of their CROSS-LANG segments end up clustered
    with their majority cluster (computed from same-LANG segments)?

    Returns None if no speaker has both cross-lang and same-lang segments in
    this conversation.
    """
    speaker_segs: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(true_labels):
        speaker_segs[t].append(i)
    recalls = []
    for spk, idxs in speaker_segs.items():
        if len(idxs) < 2: continue
        langs = [segments[i]["lang"] for i in idxs]
        # majority lang for this speaker
        from collections import Counter
        majority_lang = Counter(langs).most_common(1)[0][0]
        same_lang_idx = [i for i in idxs if segments[i]["lang"] == majority_lang]
        diff_lang_idx = [i for i in idxs if segments[i]["lang"] != majority_lang]
        if not diff_lang_idx or not same_lang_idx: continue
        # majority cluster for same-lang
        same_pred = [pred_labels[i] for i in same_lang_idx]
        majority_cluster = Counter(same_pred).most_common(1)[0][0]
        # how many cross-lang segments land in that cluster?
        in_cluster = sum(1 for i in diff_lang_idx if pred_labels[i] == majority_cluster)
        recalls.append(in_cluster / len(diff_lang_idx))
    if not recalls:
        return None
    return float(np.mean(recalls))


def _eval_encoder(name, embed_fn, convs):
    print(f"\n=== {name} ===")
    aris = []
    cs_recalls = []
    n_evaluated = 0
    t0 = time.time()
    for ci, conv in enumerate(convs):
        segs = conv["segments"]
        if len(segs) < 2: continue
        # Embed each segment
        embeddings = []
        valid_segs = []
        for seg in segs:
            wp = ROOT / seg["wav_path"]
            if not wp.exists(): continue
            try:
                embeddings.append(embed_fn(wp))
                valid_segs.append(seg)
            except Exception as e:
                print(f"  conv {ci} seg failed: {e}")
        if len(embeddings) < 2: continue
        true_labels = [seg["voice_id"] for seg in valid_segs]
        unique_speakers = sorted(set(true_labels))
        # Map to ints
        spk_idx = {s: i for i, s in enumerate(unique_speakers)}
        true_int = [spk_idx[s] for s in true_labels]
        k = len(unique_speakers)
        pred_labels = _cluster_segments(embeddings, k)
        ari = _ari_sklearn_style(true_int, pred_labels)
        aris.append(ari)
        cs = _cross_script_recall(valid_segs, true_int, pred_labels)
        if cs is not None:
            cs_recalls.append(cs)
        n_evaluated += 1
        if (ci + 1) % 10 == 0:
            print(f"  conv {ci+1}/{len(convs)}  ARI mean={np.mean(aris):.3f}  cs_recall={np.mean(cs_recalls) if cs_recalls else float('nan'):.3f}")
    elapsed = time.time() - t0
    return {
        "encoder": name,
        "n_conversations": n_evaluated,
        "ari_mean": round(float(np.mean(aris)), 4) if aris else None,
        "ari_median": round(float(np.median(aris)), 4) if aris else None,
        "ari_p25": round(float(np.percentile(aris, 25)), 4) if aris else None,
        "ari_p75": round(float(np.percentile(aris, 75)), 4) if aris else None,
        "cross_script_recall_mean": round(float(np.mean(cs_recalls)), 4) if cs_recalls else None,
        "cross_script_recall_n": len(cs_recalls),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    convs = _load_convs()
    print(f"loaded {len(convs)} conversations")

    ENCODERS = [
        ("wavlm_sv",   _wavlm_sv_embedder),
        ("ecapa_tdnn", _ecapa_embedder),
        ("lase_r1",    lambda: _lase_embedder(LASE_CKPT)),
    ]

    results = []
    for name, factory in ENCODERS:
        try:
            embed_fn = factory()
            stats = _eval_encoder(name, embed_fn, convs)
            results.append(stats)
        except Exception as e:
            print(f"  FAILED for {name}: {e}")
            results.append({"encoder": name, "error": str(e)})

    OUT.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    print("\n=== TABLE ===")
    print(f"{'encoder':15s}  {'ARI mean':>10s}  {'ARI median':>10s}  {'cs_recall':>10s}  {'n_convs':>8s}")
    for r in results:
        if "error" in r:
            print(f"{r['encoder']:15s}  ERROR: {r['error']}")
            continue
        print(f"{r['encoder']:15s}  {r['ari_mean']:>10}  {r['ari_median']:>10}  "
              f"{r['cross_script_recall_mean']:>10}  {r['n_conversations']:>8}")


if __name__ == "__main__":
    main()
