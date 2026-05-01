"""LASE — Language-Adversarial Speaker Encoder.

**Goal.** Force a speaker embedding to be predictive of *who* is speaking but
*not* of *which language* they speak. Under cross-script voice cloning, a
conventional speaker encoder drifts — clone an English speaker, synthesize
in Telugu, and the voice shifts toward a "generic Telugu speaker" because
the encoder has entangled speaker identity with language statistics. LASE
breaks that entanglement.

**Mechanism (three parts):**

1. A **backbone speaker encoder** produces a dense embedding from a short
   reference clip. At initialization this can be a frozen pretrained encoder
   (Resemblyzer / WavLM / Chatterbox's internal speaker head) so we don't
   need to train from scratch.
2. A **gradient-reversal layer (GRL)** sits between the embedding and the
   language head. Forward pass is identity; backward pass flips the sign of
   the gradient and scales it by a schedulable λ.
3. A small **language classifier head** is trained to predict the language
   from the embedding. Because of the GRL, making the classifier's job
   easier hurts our speaker-encoder training — so the encoder learns to
   hide language information.

**Training loss** (to be wired in Sprint 5)::

    L_total = L_speaker  +  λ * L_lang_adv

``L_speaker`` is whatever objective drives the downstream TTS speaker signal
(contrastive loss against other speakers, or a reconstruction loss against
the cloned target). ``L_lang_adv`` is cross-entropy of the language
classifier — the GRL inverts its gradient so the encoder *minimises* the
classifier's accuracy while the classifier still tries to maximise it
(it's an adversarial game, but because the gradient is reversed we can
train end-to-end without alternating updates).

**λ schedule.** Starting λ at 0 and ramping up slowly is standard practice
(Ganin & Lempitsky 2015). Praxy default: 5k-step warmup at λ=0, then linear
ramp to peak λ=0.1 over 10k steps. Overly aggressive λ collapses the
embedding into noise; too small does nothing.

This file contains the module definitions and a shape test. Actual training
happens in Sprint 5 (``training/train_lase.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.autograd import Function


# -----------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin & Lempitsky 2015).
# -----------------------------------------------------------------------------

class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:  # type: ignore[override]
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: Tensor, lambda_: float = 1.0) -> Tensor:
    """Reverse gradient during backward. Forward is identity."""
    return _GradientReversal.apply(x, float(lambda_))


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return gradient_reverse(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda={self.lambda_}"


# -----------------------------------------------------------------------------
# λ schedule.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LambdaSchedule:
    """Linear ramp: 0 during warmup, then linearly rise to peak over ramp_steps."""

    warmup_steps: int = 5_000
    ramp_steps: int = 10_000
    peak: float = 0.1

    def value_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        t = step - self.warmup_steps
        if t >= self.ramp_steps:
            return self.peak
        return self.peak * (t / self.ramp_steps)


# -----------------------------------------------------------------------------
# Language classifier head.
# -----------------------------------------------------------------------------

class LanguageClassifier(nn.Module):
    """Small MLP that predicts language from a speaker embedding.

    The classifier's architecture is deliberately shallow — we want it to be
    easy to fool. A deep classifier would overpower the encoder in the
    adversarial game and make training unstable.
    """

    def __init__(self, embedding_dim: int, n_languages: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_languages),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.net(embedding)


# -----------------------------------------------------------------------------
# LASE wrapper.
# -----------------------------------------------------------------------------

class LASE(nn.Module):
    """Language-Adversarial Speaker Encoder.

    Wraps a speaker-encoder backbone (supplied as-is) with a GRL + language
    classifier. The backbone can be any nn.Module that returns a dense
    embedding — we keep the interface minimal so you can swap in WavLM,
    Resemblyzer, or Chatterbox's speaker head at will.

    Args:
        backbone: module taking (B, T) or (B, C, T) audio tensors and returning
            (B, embedding_dim) speaker embeddings.
        embedding_dim: size of the speaker embedding.
        n_languages: number of languages in the adversarial classifier (e.g.,
            4 for Telugu/Hindi/Tamil/English).
        lambda_schedule: how to ramp the GRL strength during training.

    Forward returns a dict with:
        - 'embedding': (B, embedding_dim) — the speaker embedding
        - 'lang_logits': (B, n_languages) — language classifier output
          (passed through GRL; use with cross-entropy against true language id)
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        n_languages: int,
        lambda_schedule: LambdaSchedule | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.n_languages = n_languages
        self.grl = GradientReversalLayer(lambda_=0.0)  # starts at 0, scheduler updates
        self.lang_classifier = LanguageClassifier(embedding_dim, n_languages)
        self.schedule = lambda_schedule or LambdaSchedule()
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def set_step(self, step: int) -> None:
        """Update the GRL lambda from the schedule. Call once per training step."""
        self._step.fill_(int(step))
        self.grl.lambda_ = self.schedule.value_at(step)

    @property
    def current_lambda(self) -> float:
        return self.grl.lambda_

    def forward(self, audio: Tensor) -> dict[str, Tensor]:
        embedding = self.backbone(audio)
        reversed_ = self.grl(embedding)
        lang_logits = self.lang_classifier(reversed_)
        return {
            "embedding": embedding,
            "lang_logits": lang_logits,
        }


# -----------------------------------------------------------------------------
# SupCon loss (Khosla et al. 2020). Used by training/lase_train.py as the
# speaker-side objective: same-voice clips pull together, different-voice push
# apart. Keep the implementation tight and stateless — no reason to carry this
# in a class.
# -----------------------------------------------------------------------------

def supcon_loss(
    embeddings: Tensor,
    labels: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Supervised contrastive loss over per-sample labels.

    Args:
        embeddings: (B, D) tensor. We l2-normalise internally.
        labels: (B,) integer tensor — e.g., voice_id hashes.
        temperature: SupCon temperature τ. 0.07 is the Khosla default.

    Returns:
        Scalar loss. If no sample has a positive pair in the batch, returns 0
        (upstream should ensure batches contain ≥2 same-label samples).
    """
    import torch.nn.functional as F  # noqa: PLC0415

    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T / temperature
    # Numerical stability: subtract row-max before exp.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    # Positive mask: same label, excluding the diagonal.
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    pos_mask = same & ~eye

    # Denominator mask: everything except the diagonal.
    denom_mask = ~eye

    exp_sim = torch.exp(sim) * denom_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count
    # Only anchors with ≥1 positive contribute.
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return -mean_log_prob_pos[has_pos].mean()


# -----------------------------------------------------------------------------
# Stand-in backbone for shape tests. Real backbones (WavLM, Resemblyzer) get
# plugged in at training time.
# -----------------------------------------------------------------------------

class DummySpeakerEncoder(nn.Module):
    """A trivial placeholder encoder — only for shape tests. Do not train."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(1, embedding_dim)

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, T) → (B, 1, T)
        pooled = self.pool(audio).squeeze(-1)  # (B, 1)
        return self.proj(pooled)


# -----------------------------------------------------------------------------
# WavLM-Base+ speaker encoder — the real backbone for LASE training.
#
# Pin chain (per memory project_indicf5_unblock_recipe_2026-04-27):
#     torch==2.4.0  +  transformers==4.49.0  +  huggingface_hub>=0.25
#
# Behaviour:
#   - Forward takes raw 16 kHz mono waveform (B, T) or (B, 1, T) and returns
#     a `embedding_dim`-d speaker embedding.
#   - Internally pulls the average of WavLM hidden states from layers 10–12
#     (the layers that VoxSim / WavLM-SV recipes find most speaker-rich) and
#     mean-pools over time to a 768-d vector, then projects to embedding_dim.
#   - R1: backbone frozen, only the projection MLP trains.
#   - R2: caller wraps the backbone with PEFT LoRA on attention projections
#     before constructing this encoder. Loading LoRA is out-of-band so this
#     class stays pin-light.
# -----------------------------------------------------------------------------


class WavLMSpeakerEncoder(nn.Module):
    """Wrap microsoft/wavlm-base-plus into a (B, T) → (B, embedding_dim) module.

    Args:
        model_name: HuggingFace repo id. Defaults to ``microsoft/wavlm-base-plus``;
            ``microsoft/wavlm-base-plus-sv`` is a drop-in alternative trained for
            speaker verification, but ships with a downstream head we strip.
        embedding_dim: projection output dim. 256 by default to match the
            Chatterbox T3Cond.speaker_emb interface used at inference.
        freeze_backbone: if True (R1 default), the WavLM weights have
            ``requires_grad=False`` and the encoder is in eval() mode for the
            backbone forward (BatchNorm/dropout disabled).
        layer_range: which transformer layers to mean over for the pooled
            representation. Default (10, 12) inclusive — the SV-rich band per
            Chen 2022.
        sampling_rate: expected input sample rate. WavLM was trained at 16 kHz;
            we error if the caller hands us anything else (caller is
            responsible for resampling).
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        layer_range: tuple[int, int] = (10, 12),
        sampling_rate: int = 16_000,
    ) -> None:
        super().__init__()
        try:
            from transformers import WavLMModel  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers must be installed to use WavLMSpeakerEncoder. "
                "Install with `uv pip install transformers==4.49.0`."
            ) from e

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.layer_range = layer_range
        self.sampling_rate = sampling_rate
        self.freeze_backbone = freeze_backbone

        self.backbone = WavLMModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        hidden_size: int = int(self.backbone.config.hidden_size)  # 768 for base+

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def forward(self, audio: Tensor) -> Tensor:
        """Compute (B, embedding_dim) speaker embeddings.

        Accepts (B, T) or (B, 1, T) float tensors at 16 kHz. Values should be
        in [-1, 1] (raw waveform). No further normalisation is performed —
        WavLM internally applies its own layer-norm.
        """
        if audio.dim() == 3:
            if audio.size(1) != 1:
                raise ValueError(
                    f"WavLMSpeakerEncoder expects mono audio; got C={audio.size(1)}"
                )
            audio = audio.squeeze(1)
        if audio.dim() != 2:
            raise ValueError(
                f"WavLMSpeakerEncoder expects (B, T) or (B, 1, T); got {audio.shape}"
            )

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(audio, output_hidden_states=True)
        else:
            outputs = self.backbone(audio, output_hidden_states=True)

        # outputs.hidden_states: tuple of (B, T_frames, H), one per layer + 1 input.
        hidden_states = outputs.hidden_states
        lo, hi = self.layer_range
        # Layer indices are inclusive on both ends.
        selected = torch.stack(hidden_states[lo:hi + 1], dim=0)  # (L_sel, B, T, H)
        pooled_layers = selected.mean(dim=0)                     # (B, T, H)
        pooled = pooled_layers.mean(dim=1)                       # (B, H)
        embedding = self.projection(pooled)                      # (B, D)
        return embedding

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


# -----------------------------------------------------------------------------
# ECAPA-TDNN speaker encoder — the alternative backbone for the GRL ablation.
#
# Same interface as WavLMSpeakerEncoder: (B, T) raw 16 kHz waveform →
# (B, embedding_dim) speaker embedding. Used to isolate whether LASE's
# improvement comes from the GRL training objective or from the WavLM
# backbone choice.
#
# ECAPA produces a 192-d embedding from its own pretrained head; we project
# that to embedding_dim with the same 2-layer MLP as the WavLM variant.
# Backbone is frozen by default (matches LASE r1 setup); only the projection
# trains.
# -----------------------------------------------------------------------------


class EcapaSpeakerEncoder(nn.Module):
    """Wrap SpeechBrain's spkrec-ecapa-voxceleb into a (B, T) → (B, embedding_dim) module.

    Args mirror WavLMSpeakerEncoder. embedding_dim is the projection output
    (256 by default for parity with the WavLM variant).
    """

    ECAPA_OUTPUT_DIM = 192  # SpeechBrain ECAPA-TDNN's pretrained head dim.

    def __init__(
        self,
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        sampling_rate: int = 16_000,
    ) -> None:
        super().__init__()
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "speechbrain must be installed to use EcapaSpeakerEncoder. "
                "Install with `uv add --optional ml speechbrain`."
            ) from e

        self.embedding_dim = embedding_dim
        self.sampling_rate = sampling_rate
        self.freeze_backbone = freeze_backbone

        self.backbone = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec-ecapa-voxceleb",
        )

        self.projection = nn.Sequential(
            nn.Linear(self.ECAPA_OUTPUT_DIM, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, embedding_dim),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, audio: Tensor) -> Tensor:
        """(B, T) or (B, 1, T) at 16 kHz → (B, embedding_dim)."""
        if audio.dim() == 3:
            if audio.size(1) != 1:
                raise ValueError(f"EcapaSpeakerEncoder expects mono audio; got C={audio.size(1)}")
            audio = audio.squeeze(1)
        if audio.dim() != 2:
            raise ValueError(f"EcapaSpeakerEncoder expects (B, T) or (B, 1, T); got {audio.shape}")

        # SpeechBrain encode_batch returns (B, 1, 192). It runs in eval mode by
        # default; in train mode the backbone would be doing dropout etc. — we
        # disable that explicitly.
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                emb = self.backbone.encode_batch(audio)
        else:
            emb = self.backbone.encode_batch(audio)

        # (B, 1, 192) → (B, 192)
        emb = emb.squeeze(1)
        return self.projection(emb)


# -----------------------------------------------------------------------------
# Shape tests — confirm the forward path and the GRL actually reverses gradient.
# -----------------------------------------------------------------------------

def _test_shapes() -> None:
    torch.manual_seed(1337)
    backbone = DummySpeakerEncoder(embedding_dim=256)
    lase = LASE(backbone=backbone, embedding_dim=256, n_languages=4)

    # Fake 2-second 16 kHz mono batch of 3 samples.
    audio = torch.randn(3, 32_000)

    out = lase(audio)
    assert out["embedding"].shape == (3, 256), out["embedding"].shape
    assert out["lang_logits"].shape == (3, 4), out["lang_logits"].shape
    print("  shapes OK:", {k: tuple(v.shape) for k, v in out.items()})


def _test_gradient_reversal() -> None:
    """Confirm GRL actually flips the sign of the gradient w.r.t. the input."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    y = gradient_reverse(x, lambda_=0.5)
    loss = y.sum()
    loss.backward()
    # Without GRL, dy/dx = 1 → x.grad would be all 1s (since sum).
    # With GRL λ=0.5, x.grad should be all -0.5.
    assert torch.allclose(x.grad, torch.full_like(x.grad, -0.5)), x.grad
    print("  gradient reversal verified: x.grad is uniformly -0.5 under λ=0.5")


def _test_lambda_schedule() -> None:
    sched = LambdaSchedule(warmup_steps=100, ramp_steps=200, peak=0.2)
    assert sched.value_at(0) == 0.0
    assert sched.value_at(99) == 0.0
    assert sched.value_at(100) == 0.0
    mid = sched.value_at(200)
    assert 0.09 < mid < 0.11, mid
    assert sched.value_at(400) == 0.2
    assert sched.value_at(10_000) == 0.2
    print("  λ schedule OK: 0 → 0 @ warmup → 0.1 @ mid → 0.2 @ peak")


def _test_end_to_end_backprop() -> None:
    """Fake a training step: loss on language classifier + dummy speaker loss."""
    torch.manual_seed(7)
    backbone = DummySpeakerEncoder(embedding_dim=128)
    lase = LASE(
        backbone=backbone,
        embedding_dim=128,
        n_languages=4,
        lambda_schedule=LambdaSchedule(warmup_steps=0, ramp_steps=1, peak=1.0),
    )
    lase.set_step(10)  # fully ramped
    assert lase.current_lambda == 1.0

    opt = torch.optim.SGD(lase.parameters(), lr=1e-3)
    audio = torch.randn(4, 16_000)
    lang_ids = torch.tensor([0, 1, 2, 3])

    for _ in range(3):
        out = lase(audio)
        lang_loss = torch.nn.functional.cross_entropy(out["lang_logits"], lang_ids)
        speaker_loss = out["embedding"].pow(2).mean()  # trivial "spread out" proxy
        total = speaker_loss + lang_loss
        opt.zero_grad()
        total.backward()
        opt.step()
    print(f"  end-to-end step OK: final total loss ≈ {total.item():.4f}")


def _test_wavlm_encoder_shape_and_freeze() -> None:
    """Shape + frozen/trainable param counts for the WavLM backbone.

    Skipped silently if (a) transformers is not installed, or (b) the
    microsoft/wavlm-base-plus weights are not yet cached locally — we never
    auto-download in a unit test (HF hits would surprise the caller).
    """
    import os  # noqa: PLC0415

    try:
        import transformers  # noqa: F401, PLC0415
    except ImportError:
        print("  WavLM test skipped: transformers not installed")
        return

    # Refuse to download in tests. Honor HF_HUB_OFFLINE.
    if os.environ.get("LASE_ALLOW_HF_DOWNLOAD") != "1":
        # Check the HF cache for the weights; if missing, skip.
        try:
            from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
            cached = try_to_load_from_cache(
                repo_id="microsoft/wavlm-base-plus", filename="config.json"
            )
        except Exception:
            cached = None
        if cached is None:
            print("  WavLM test skipped: weights not in HF cache "
                  "(set LASE_ALLOW_HF_DOWNLOAD=1 to enable)")
            return

    enc = WavLMSpeakerEncoder(embedding_dim=256, freeze_backbone=True)
    audio = torch.randn(2, 16_000)  # 1 second mono batch
    out = enc(audio)
    assert out.shape == (2, 256), out.shape

    n_train = enc.num_trainable_params()
    n_frozen = enc.num_frozen_params()
    # Frozen backbone (~94M) >> trainable projection (~0.5M).
    assert n_frozen > 50_000_000, f"backbone seems unfrozen: frozen={n_frozen}"
    assert 100_000 < n_train < 5_000_000, f"projection size off: trainable={n_train}"
    print(
        f"  WavLM encoder OK: out{tuple(out.shape)}; "
        f"frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.2f}M"
    )

    # Gradient flow: the projection params should receive gradient when wrapped
    # in LASE with GRL fully ramped.
    lase = LASE(
        backbone=enc, embedding_dim=256, n_languages=4,
        lambda_schedule=LambdaSchedule(warmup_steps=0, ramp_steps=1, peak=1.0),
    )
    lase.set_step(10)
    lang_ids = torch.tensor([0, 1])
    out_dict = lase(audio)
    loss = (
        out_dict["embedding"].pow(2).mean()
        + torch.nn.functional.cross_entropy(out_dict["lang_logits"], lang_ids)
    )
    loss.backward()
    proj_grad = enc.projection[0].weight.grad
    assert proj_grad is not None and proj_grad.abs().sum() > 0, \
        "projection MLP did not receive gradient through GRL path"
    # Frozen backbone params should have no grad.
    bb_param = next(enc.backbone.parameters())
    assert bb_param.grad is None, "frozen backbone unexpectedly received grad"
    print("  WavLM gradient flow verified: projection grads OK, backbone frozen")


def run_all_tests() -> None:
    _test_shapes()
    _test_gradient_reversal()
    _test_lambda_schedule()
    _test_end_to_end_backprop()
    _test_wavlm_encoder_shape_and_freeze()
    print("\nall LASE smoke tests passed")


if __name__ == "__main__":
    run_all_tests()
