# 🎙️ LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity

Reference implementation accompanying the paper *"LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity Preservation"* ([arXiv:TBD](https://arxiv.org/abs/TBD)).

> *A speaker encoder used in multilingual voice cloning should treat the same speaker identically regardless of which script the audio was uttered in. Off-the-shelf encoders do not, and the failure is accent-conditional.*

LASE is a small projection head (~170k trainable params) over a frozen WavLM-base-plus backbone, trained with two losses: a supervised contrastive loss over voice identity, and a gradient-reversal cross-entropy against a 4-language classifier. The result is a 256-dim speaker embedding that preserves speaker identity across Devanagari, Telugu, Tamil, and Latin scripts.

## Headline result

| Encoder | Western voices gap | Indian voices gap |
|---|---|---|
| WavLM-base-plus-sv (off-the-shelf) | 0.082 | 0.006 |
| ECAPA-TDNN (off-the-shelf) | 0.105 | 0.058 |
| ECAPA + GRL (ablation) | 0.027 | 0.037 |
| **LASE r1 (WavLM + GRL, ours)** | **0.013** | **−0.000** |

Lower is better. *gap* = within-script median minus cross-script median for the same speaker. LASE r1's bootstrap 95% CI on gap straddles zero on both held-out corpora; both baselines' CIs are bounded above zero.

In synthetic multi-speaker code-switching diarization, LASE matches ECAPA-TDNN on cross-script speaker recall (0.788 vs 0.789) using ~100× less training data (1118 cross-script pairs vs 1M+ VoxCeleb utterances).

## Assets

| Asset | Where | License |
|---|---|---|
| **Paper** | `paper/lase.pdf` (also tarball for arXiv) | CC-BY-4.0 |
| **LASE r1 checkpoint** | [`Praxel/lase-r1`](https://huggingface.co/Praxel/lase-r1) | MIT |
| **Training corpus** (1118 pairs) | [`Praxel/codeswitch-pairs-lase`](https://huggingface.co/datasets/Praxel/codeswitch-pairs-lase) | CC-BY-4.0 |
| **Western held-out** (1043 pairs) | [`Praxel/codeswitch-pairs-lase-heldout`](https://huggingface.co/datasets/Praxel/codeswitch-pairs-lase-heldout) | CC-BY-4.0 |
| **Indian held-out** (1369 pairs) | [`Praxel/codeswitch-pairs-lase-indian`](https://huggingface.co/datasets/Praxel/codeswitch-pairs-lase-indian) | CC-BY-4.0 |
| **Code** (this repo) | `github.com/praxelhq/lase` | MIT |

## Quick start

### Use a pretrained LASE encoder

```python
from huggingface_hub import hf_hub_download
import torch
from models.lase import LASE, LambdaSchedule, WavLMSpeakerEncoder

ckpt_path = hf_hub_download("Praxel/lase-r1", "last.pt")
backbone = WavLMSpeakerEncoder("microsoft/wavlm-base-plus", embedding_dim=256, freeze_backbone=True)
model = LASE(backbone, embedding_dim=256, n_languages=4,
             lambda_schedule=LambdaSchedule(200, 500, 0.1))
model.load_state_dict(torch.load(ckpt_path)["model"], strict=False)
model.eval()

# wav: (B, T) float32 at 16 kHz, ~2 seconds
embedding = model(wav)["embedding"]   # (B, 256)
```

### Reproduce the paper's headline numbers

```bash
# 1. Build the corpus + train (Modal A10G; ~$0.31 / round, ~17 min)
modal run scripts/modal_lase_train.py::train_round --round-id r1 --execute

# 2. Evaluate on held-out
python scripts/eval_secs_gap_multi_encoder.py     # 3-encoder comparison
python scripts/eval_ablation.py                    # ECAPA+GRL ablation eval
python scripts/eval_diarization.py                 # synthetic multi-speaker DER
python scripts/bootstrap_cis.py                    # 95% CIs on gap + margin
```

### Build a fresh cross-script corpus

```bash
python scripts/codeswitch_pairs.py \
    --out-dir data/your_corpus \
    --n-voices 8 --pairs-per-voice 50 \
    --source-manifest your_transcripts.jsonl \
    --execute
```

## Method (one paragraph)

LASE wraps a frozen [WavLM-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) backbone with a 2-layer projection MLP (768 → 512 → 256) and a gradient-reversal language classifier on the 256-d output. Training optimises a sum of two losses on each batch: supervised contrastive (SupCon) over voice identity, and cross-entropy against a 4-class language classifier post-GRL with scheduled λ (warmup 0 for 200 steps, ramp to 0.1 over 500 steps, hold). 1000 steps total at batch 16, AdamW LR 1e-4. Trained on 1118 same-voice cross-script pairs synthesised from 8 ElevenLabs Multilingual voices and gated through a WavLM-cosine ≥ 0.90 threshold.

## Citation

```bibtex
@misc{lase2026,
  title={{LASE}: Language-Adversarial Speaker Encoding for {Indic} Cross-Script Identity Preservation},
  author={Menta, Venkata Pushpak Teja},
  year={2026},
  eprint={TBD},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
}
```

## Companion papers

- **PSP** ([arXiv:2604.25476](https://arxiv.org/abs/2604.25476)) — interpretable per-dimension accent benchmark for Indic TTS. LASE addresses the orthogonal *identity* axis.
- **Praxy Voice** ([arXiv:2604.25441](https://arxiv.org/abs/2604.25441)) — open-source Indic TTS that LASE encodings can plug into for cross-script voice cloning.

## License

- Code (this repo): **MIT**
- Paper: **CC-BY-4.0**
- Released artefacts on HF: licensed per their model / dataset cards.

## Contact

Pushpak Teja Menta · Praxel Ventures · pushpak@praxel.in · [ORCID 0009-0003-2479-9208](https://orcid.org/0009-0003-2479-9208)
