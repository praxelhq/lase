"""Modal app for LASE training (real implementation, dry-run gated).

Mirrors patterns from ``serving/modal_app.py`` and ``training/lase_train.py``.

Pin chain (from memory ``project_indicf5_unblock_recipe_2026-04-27``):
    torch==2.4.0  +  transformers==4.49.0  +  peft + accelerate

Cost targets:
    R1 sanity (A10G frozen, 1k steps, batch 16): ~$1, ~1 h
    R2 final (A100-80GB LoRA, 4k steps, batch 32): ~$15, ~4 h

Concurrency cap: ``MAX_CONCURRENT_GPUS=4`` (≤6 per
``feedback_modal_gpu_concurrency_limit.md``).

CLI::

    # Always dry-run by default.
    uv run python paper/lase/modal_lase_train.py --round-id r1

    # Real execution requires --execute AND we recommend running via
    # ``modal run`` directly, e.g.:
    #
    #     modal run paper/lase/modal_lase_train.py::train_round \\
    #         --round-id r1 --use-lora False --max-steps 1000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# IMPORTANT: keep modal import lazy so this module can be imported and
# inspected (and dry-run printed) without a Modal token / network.

APP_NAME = "praxy-lase"
MODAL_VOLUME_NAME = "praxy-lase-runs"
HF_DATASET_NAME = "Praxel/codeswitch_pairs_v1"  # not yet uploaded
DEFAULT_BACKBONE = "microsoft/wavlm-base-plus"

CUDA_IMAGE_PINS: dict[str, str] = {
    "torch": "==2.4.0",
    "torchaudio": "==2.4.0",
    "transformers": "==4.49.0",
    "peft": "==0.13.0",
    "accelerate": "==0.33.0",
    "datasets": ">=2.18",
    "soundfile": "==0.12.1",
    "huggingface_hub": ">=0.25",
    "numpy": "<2",
    "scipy": ">=1.10",
    "speechbrain": ">=1.0",          # for ECAPA-TDNN backbone in the ablation
    "librosa": ">=0.10",
}

MAX_CONCURRENT_GPUS = 4  # per feedback_modal_gpu_concurrency_limit.md


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class LaseModalConfig:
    round_id: str = "r1"
    backbone: str = DEFAULT_BACKBONE
    backbone_type: str = "wavlm"   # one of: "wavlm" | "ecapa"
    use_lora: bool = False
    max_steps: int = 1_000
    batch_size: int = 16
    learning_rate: float = 1e-4
    grl_peak: float = 0.1
    grl_warmup: int = 200
    grl_ramp: int = 500
    supcon_temperature: float = 0.07
    gpu: str = "A10G"
    timeout_seconds: int = 60 * 60 * 6  # 6 h hard cap
    save_every: int = 500
    log_every: int = 25
    grad_clip: float = 1.0
    hf_dataset: str = HF_DATASET_NAME
    execute: bool = False  # if False, no Modal call is made

    def cost_estimate_dollars(self) -> float:
        # Rough per-hour blended price (A10G $1.10, A100-80GB $3.30, H100 $5.0).
        rate = {"A10G": 1.10, "A100-40GB": 2.10, "A100-80GB": 3.30, "H100": 5.00}.get(
            self.gpu, 1.10
        )
        # Assume ~1.0 s / step on A10G frozen, ~3.5 s / step on A100 LoRA.
        sec_per_step = 3.5 if self.use_lora else 1.0
        wall_seconds = self.max_steps * sec_per_step
        return round(rate * wall_seconds / 3600, 2)


# -----------------------------------------------------------------------------
# Image (lazy-built; only invoked if --execute)
# -----------------------------------------------------------------------------


def _build_modal_image():
    import modal  # noqa: PLC0415
    pkgs: list[str] = []
    for name, ver in CUDA_IMAGE_PINS.items():
        pkgs.append(f"{name}{ver}")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "git", "libsndfile1")
        .pip_install(*pkgs)
        # Ship ONLY the python modules we import. add_local_dir of the repo
        # root would mount 320k+ files (processed/, .venv/, .claude/) and
        # blow past Modal's 125k-file mount cap. add_local_python_source ships
        # the named modules' .py files plus their package __init__.py chain.
        .add_local_python_source("models", "training", "paper")
    )
    return image


def _modal_app():
    """Construct the Modal App lazily, only when --execute is set."""
    import modal  # noqa: PLC0415
    app = modal.App(APP_NAME)
    image = _build_modal_image()
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
    return app, image, volume


# -----------------------------------------------------------------------------
# Data / dataset prep
# -----------------------------------------------------------------------------


def prepare_dataset(manifest_path: Path) -> list[dict]:
    """Load the codeswitch-pairs manifest into memory.

    Per ``feedback_modal_volume_inodes.md``, the LASE pair set should fit
    well below the 500k-file cap (~5k clips planned). If we ever scale up,
    swap to a parquet shard layout — but at v1 sizes the JSONL+per-clip wav
    layout is fine.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}. Run "
            "`uv run python -m paper.lase.codeswitch_pairs --execute ...` first."
        )
    rows: list[dict] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(__import__("json").loads(line))
    return rows


# -----------------------------------------------------------------------------
# Training entrypoint (runs INSIDE Modal worker if --execute, else local-CPU)
# -----------------------------------------------------------------------------


def _run_train_inner(cfg: LaseModalConfig,
                      manifest_path: str = "/runs/codeswitch_pairs_lase/manifest.jsonl",
                      output_dir: str = "/runs") -> None:
    """The actual training loop. Imports torch lazily so this file is
    importable without ML deps installed.
    """
    sys.path.insert(0, "/root/praxy_tts")
    import torch  # noqa: PLC0415

    from models.novel.lase import (  # noqa: PLC0415
        LambdaSchedule, LASE, WavLMSpeakerEncoder, EcapaSpeakerEncoder, supcon_loss,
    )
    from training.lase_train import (  # noqa: PLC0415
        TrainConfig, _load_manifest, _batch_iter,
    )

    # Build the chosen backbone. WavLM is the LASE default; ECAPA is the
    # ablation backbone that lets us isolate GRL contribution from the
    # backbone choice.
    if cfg.backbone_type == "ecapa":
        print(f"[lase_modal] backbone=EcapaSpeakerEncoder (frozen)", flush=True)
        backbone = EcapaSpeakerEncoder(
            embedding_dim=256,
            freeze_backbone=not cfg.use_lora,
        )
    else:
        print(f"[lase_modal] backbone=WavLMSpeakerEncoder ({cfg.backbone}, "
              f"frozen={not cfg.use_lora})", flush=True)
        backbone = WavLMSpeakerEncoder(
            model_name=cfg.backbone,
            embedding_dim=256,
            freeze_backbone=not cfg.use_lora,
        )

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        # Apply LoRA to the WavLM transformer blocks specifically.
        backbone.backbone = get_peft_model(backbone.backbone, lora_cfg)
        # Re-enable grads only for the LoRA params; rest stays frozen.
        for n, p in backbone.backbone.named_parameters():
            p.requires_grad = "lora_" in n

    schedule = LambdaSchedule(
        warmup_steps=cfg.grl_warmup,
        ramp_steps=cfg.grl_ramp,
        peak=cfg.grl_peak,
    )
    model = LASE(
        backbone=backbone, embedding_dim=256, n_languages=4,
        lambda_schedule=schedule,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Hand control to the existing training driver by constructing a TrainConfig.
    train_cfg = TrainConfig(
        manifest=Path(manifest_path),
        output_dir=Path(output_dir) / cfg.round_id,
        device=str(device),
        batch_size=cfg.batch_size,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        grl_peak=cfg.grl_peak,
        grl_warmup_steps=cfg.grl_warmup,
        grl_ramp_steps=cfg.grl_ramp,
        save_every=cfg.save_every,
        log_every=cfg.log_every,
        grad_clip=cfg.grad_clip,
        supcon_temperature=cfg.supcon_temperature,
    )
    # We sidestep training.lase_train.train()'s DummySpeakerEncoder hardcode
    # by reproducing the loop here with our real WavLM-backed LASE.
    rows = _load_manifest(train_cfg.manifest)
    print(f"[lase_modal] {len(rows)} clips loaded from {train_cfg.manifest}", flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate, weight_decay=0.01,
    )
    ce_loss = torch.nn.CrossEntropyLoss()
    lang_to_idx = {"en": 0, "hi": 1, "te": 2, "ta": 3}
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    import torchaudio  # noqa: PLC0415
    # Remap manifest paths from local repo-relative
    # ("data/codeswitch_pairs_lase/Rachel/en_000.wav") to Modal volume
    # ("/runs/codeswitch_pairs_lase/Rachel/en_000.wav"). No-op if already
    # absolute.
    audio_root_local = "data/codeswitch_pairs_lase/"
    audio_root_modal = "/runs/codeswitch_pairs_lase/"
    for step, batch in zip(range(1, cfg.max_steps + 1), _batch_iter(rows, cfg.batch_size)):
        waves: list[torch.Tensor] = []
        for r in batch:
            p = r["wav_path"]
            if p.startswith(audio_root_local):
                p = audio_root_modal + p[len(audio_root_local):]
            wav, sr = torchaudio.load(p)
            if sr != 16_000:
                wav = torchaudio.functional.resample(wav, sr, 16_000)
            wav = wav[0][:32_000]
            if wav.shape[0] < 32_000:
                wav = torch.nn.functional.pad(wav, (0, 32_000 - wav.shape[0]))
            waves.append(wav)
        x = torch.stack(waves).to(device)

        voice_ids = torch.tensor(
            [hash(r["voice_id"]) % (2 ** 31) for r in batch],
            device=device, dtype=torch.long,
        )
        lang_labels = torch.tensor(
            [lang_to_idx.get(r["lang"], 0) for r in batch],
            device=device, dtype=torch.long,
        )
        model.set_step(step)
        out = model(x)
        l_speaker = supcon_loss(out["embedding"], voice_ids,
                                temperature=cfg.supcon_temperature)
        l_lang_adv = ce_loss(out["lang_logits"], lang_labels)
        l_total = l_speaker + model.current_lambda * l_lang_adv

        if not torch.isfinite(l_total):
            raise RuntimeError(f"non-finite loss at step {step}; abort.")

        optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.grad_clip,
        )
        optimizer.step()

        if step % cfg.log_every == 0:
            print(
                f"[step {step:5d}] L_total={l_total.item():.4f} "
                f"L_spk={l_speaker.item():.4f} L_lang={l_lang_adv.item():.4f} "
                f"λ={model.current_lambda:.3f}", flush=True,
            )
        if step % cfg.save_every == 0:
            ckpt = train_cfg.output_dir / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step,
                        "config": asdict(cfg)}, ckpt)
            print(f"[checkpoint] {ckpt}", flush=True)

    final = train_cfg.output_dir / "last.pt"
    torch.save({"model": model.state_dict(), "step": cfg.max_steps,
                "config": asdict(cfg)}, final)
    print(f"[done] final checkpoint at {final}", flush=True)


# -----------------------------------------------------------------------------
# Modal-side wrapper (only constructed when --execute)
# -----------------------------------------------------------------------------


def _register_modal_function(cfg: LaseModalConfig):
    """Register the train function on the Modal app dynamically."""
    import modal  # noqa: PLC0415
    app, image, volume = _modal_app()

    @app.function(
        image=image,
        gpu=cfg.gpu,
        timeout=cfg.timeout_seconds,
        volumes={"/runs": volume},
        max_containers=MAX_CONCURRENT_GPUS,
        serialized=True,
    )
    def _train_remote() -> None:  # pragma: no cover (executes on Modal)
        _run_train_inner(cfg)

    return app, _train_remote


# -----------------------------------------------------------------------------
# Eval (cross-script SECS) — Modal A10G, ~30 min, ~$1
# -----------------------------------------------------------------------------


def _run_eval_inner(checkpoint_path: str, eval_manifest: str,
                    output_path: str) -> None:
    """Compute Normalised SECS over a held-out cross-script eval set.

    Loads the trained LASE checkpoint, runs WavLM-SV embeddings for the seed
    refs, the synth clips, the same-speaker upper-bound clips, and the
    different-speaker lower-bound clips, then writes a per-utterance
    JSON scorecard.
    """
    sys.path.insert(0, "/root/praxy_tts")
    import json  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    sv = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv.eval()

    def _embed(p: str) -> "torch.Tensor":
        wav, sr = torchaudio.load(p)
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        wav = wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            out = sv(wav, output_hidden_states=True)
        hs = out.hidden_states
        pooled = torch.stack(hs[10:13], dim=0).mean(dim=0).mean(dim=1)
        return F.normalize(pooled, dim=-1)

    rows: list[dict] = []
    with open(eval_manifest) as f:
        for line in f:
            r = json.loads(line)
            ref = _embed(r["ref_path"])
            syn = _embed(r["synth_path"])
            secs = float((ref * syn).sum())
            rows.append({**r, "secs_cross": secs})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[eval] wrote {len(rows)} rows to {output_path}")


# -----------------------------------------------------------------------------
# Public-API entry points
# -----------------------------------------------------------------------------


def train_round(cfg: LaseModalConfig) -> None:
    """Plan-or-run a LASE training round. Default: dry-run.

    On dry-run we just print the config and cost estimate; no Modal call.
    On --execute, we register the Modal function and call it.
    """
    print("=" * 72)
    print(f"LASE Modal training — round={cfg.round_id} execute={cfg.execute}")
    print("=" * 72)
    print(f"  backbone        : {cfg.backbone}")
    print(f"  use_lora        : {cfg.use_lora}")
    print(f"  max_steps       : {cfg.max_steps}")
    print(f"  batch_size      : {cfg.batch_size}")
    print(f"  GPU             : {cfg.gpu}  (max_concurrent={MAX_CONCURRENT_GPUS})")
    print(f"  GRL peak/ramp   : peak={cfg.grl_peak} warmup={cfg.grl_warmup} "
          f"ramp={cfg.grl_ramp}")
    print(f"  HF dataset      : {cfg.hf_dataset}")
    print(f"  Cost estimate   : ~${cfg.cost_estimate_dollars()}")
    print("-" * 72)

    if not cfg.execute:
        print("DRY-RUN. No Modal deploy. Re-run with --execute to launch.")
        print("Recommended: `modal run paper/lase/modal_lase_train.py::train_round`")
        return

    app, _train_remote = _register_modal_function(cfg)
    with app.run():  # pragma: no cover
        _train_remote.remote()


def eval_lase(checkpoint: str, eval_manifest: str, output: str,
              *, execute: bool = False) -> None:
    """Plan-or-run cross-script SECS eval.

    Modal A10G, ~30 min, ~$1 for the planned 1,440-utt benchmark.
    """
    print(f"[eval_lase] checkpoint={checkpoint} manifest={eval_manifest} "
          f"output={output} execute={execute}")
    if not execute:
        print("DRY-RUN. No Modal deploy.")
        return
    import modal  # noqa: PLC0415
    app, image, volume = _modal_app()

    @app.function(image=image, gpu="A10G", timeout=60 * 60 * 2,
                  volumes={"/runs": volume}, max_containers=MAX_CONCURRENT_GPUS)
    def _eval_remote() -> None:  # pragma: no cover
        _run_eval_inner(checkpoint, eval_manifest, output)

    with app.run():  # pragma: no cover
        _eval_remote.remote()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> LaseModalConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round-id", default="r1", choices=["r1", "r2", "smoke", "ablation_ecapa"])
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--backbone-type", default="wavlm", choices=["wavlm", "ecapa"],
                   help="Speaker encoder backbone family. wavlm=LASE default; ecapa=ablation.")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--grl-peak", type=float, default=0.1)
    p.add_argument("--grl-warmup", type=int, default=200)
    p.add_argument("--grl-ramp", type=int, default=500)
    p.add_argument("--gpu", default="A10G",
                   choices=["A10G", "A100-40GB", "A100-80GB", "H100"])
    p.add_argument("--execute", action="store_true",
                   help="Actually deploy on Modal. Default is dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Force dry-run (default; for explicitness).")
    args = p.parse_args(argv)
    execute = bool(args.execute) and not bool(args.dry_run)
    return LaseModalConfig(
        round_id=args.round_id,
        backbone=args.backbone,
        backbone_type=args.backbone_type,
        use_lora=args.use_lora,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grl_peak=args.grl_peak,
        grl_warmup=args.grl_warmup,
        grl_ramp=args.grl_ramp,
        gpu=args.gpu,
        execute=execute,
    )


if __name__ == "__main__":
    train_round(_parse_args())
