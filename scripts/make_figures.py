"""Generate the two main figures for paper #4 LASE.

Fig 1: 3-distribution comparison across encoders (boxplot or violin).
       Shows within-script / cross-script / across-voice cosine distributions
       for WavLM-SV, ECAPA-TDNN, LASE r1 side-by-side. The headline figure.

Fig 2: Training loss curves — L_speaker, L_lang_adv, lambda over 1000 steps.

Output: paper/lase/figures/{boxplot.pdf, loss_curves.pdf}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _load_multi_encoder_distributions():
    """Returns {encoder: {bucket: list_of_cosines}}.

    Re-runs the embedder on a small sample to recover the underlying cosines
    (the JSON only stored summary stats). Cheap (CPU, ~1 min/encoder).
    """
    # Cheaper path: re-derive cosines directly from the saved summary by
    # using the percentile data points as a proxy distribution. We have
    # min/p25/median/p75/max for each (encoder, bucket).
    summary = json.loads(
        (ROOT / "data/codeswitch_pairs_lase_heldout/multi_encoder_secs.json").read_text()
    )
    out: dict[str, dict[str, list[float]]] = {}
    for enc, stats in summary["encoders"].items():
        if "error" in stats:
            continue
        out[enc] = {}
        for bucket in ("within_script", "cross_script", "across_voice"):
            s = stats[bucket]
            # Synthesise the box from the 5-number summary
            out[enc][bucket] = [s["min"], s["p25"], s["median"], s["p75"], s["max"]]
    return out


def fig_boxplot():
    dists = _load_multi_encoder_distributions()
    encoders = ["wavlm_sv", "ecapa_tdnn", "lase_r1"]
    encoder_labels = ["WavLM-base-plus-sv", "ECAPA-TDNN", "LASE r1 (ours)"]
    buckets = ["within_script", "cross_script", "across_voice"]
    bucket_labels = ["within-speaker\nwithin-script", "within-speaker\nCROSS-script", "across-speaker\nwithin-script"]
    bucket_colors = ["#2ca02c", "#1f77b4", "#d62728"]  # green, blue, red

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, enc, label in zip(axes, encoders, encoder_labels):
        if enc not in dists:
            ax.set_title(f"{label} (missing)")
            continue
        # Build a 5-number-summary box manually
        positions = [1, 2, 3]
        five_summary_data = []
        for bucket in buckets:
            stats = dists[enc][bucket]  # [min, p25, median, p75, max]
            five_summary_data.append(stats)

        bp = ax.bxp(
            [{"med": s[2], "q1": s[1], "q3": s[3], "whislo": s[0], "whishi": s[4],
              "fliers": []} for s in five_summary_data],
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], bucket_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(bucket_labels, fontsize=8.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

        # Annotate with median values
        for pos, stats in zip(positions, five_summary_data):
            ax.text(pos, stats[2] + 0.04, f"{stats[2]:.3f}",
                    ha="center", fontsize=7.5, fontweight="bold")

    axes[0].set_ylabel("WavLM-SV cosine similarity", fontsize=9)
    fig.suptitle("Cross-script speaker-identity gap across speaker encoders\n"
                 "(held-out 1043-pair corpus, 8 voices × en/hi/te/ta)",
                 fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "boxplot_3encoder.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def fig_loss_curves():
    """Plot LASE r1 training curves. Data from server-side modal app logs
    captured during the 1000-step run (every 25 steps).
    Source: /runs/r1/last.pt training log, manually copied from app logs
    because the local /tmp/lase_modal_r1.log only captured heartbeat errors
    after the connection dropped mid-run."""
    # Full 40-row series at every-25-step granularity. Verified against the
    # server-side modal logs of run ap-pP2UoYDkNRtwcUBrZGBOo2 (2026-05-01).
    steps = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325,
             350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625,
             650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925,
             950, 975, 1000]
    l_speaker = [2.6651, 2.6447, 2.2734, 2.0095, 2.0108, 1.6436, 1.3137,
                 1.3510, 1.2445, 1.3214, 1.4997, 1.2994, 1.5322, 0.8401,
                 1.5150, 1.1763, 1.4152, 1.7021, 0.9514, 0.9345, 0.8599,
                 0.7849, 1.1240, 0.7457, 0.9346, 0.8833, 0.7177, 0.9601,
                 0.9816, 0.9680, 0.8078, 0.7667, 0.6635, 0.8046, 0.7808,
                 0.7014, 0.4285, 0.8322, 1.0717, 1.3292]
    l_lang    = [1.3882, 1.3974, 1.3887, 1.3873, 1.3893, 1.3921, 1.3915,
                 1.3841, 1.3895, 1.3558, 1.3648, 1.3380, 1.3459, 1.3926,
                 1.3431, 1.3127, 1.3393, 1.4062, 1.2397, 1.3789, 1.2307,
                 1.3202, 1.3335, 1.3788, 1.2709, 1.2267, 1.3231, 1.3473,
                 1.2142, 1.2149, 1.4528, 1.2914, 1.3109, 1.2196, 1.3067,
                 1.3145, 1.2643, 1.3112, 1.1889, 1.4099]
    lam       = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
                 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080,
                 0.085, 0.090, 0.095, 0.100, 0.100, 0.100, 0.100, 0.100,
                 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100]
    assert len(steps) == len(l_speaker) == len(l_lang) == len(lam) == 40

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.plot(steps, l_speaker, "-", color="#1f77b4", linewidth=1.5,
             label=r"$L_{\mathrm{speaker}}$ (SupCon)", marker="o", markersize=3)
    ax1.plot(steps, l_lang, "-", color="#d62728", linewidth=1.5,
             label=r"$L_{\mathrm{lang\,adv}}$ (CE, post-GRL)", marker="s", markersize=3)
    ax1.axhline(np.log(4), linestyle="--", color="#d62728", alpha=0.5,
                linewidth=0.8, label=r"$\ln 4 \approx 1.386$ (4-class uniform)")
    ax1.set_xlabel("training step", fontsize=10)
    ax1.set_ylabel("loss", fontsize=10)
    ax1.set_ylim(0, 3.0)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot(steps, lam, "--", color="grey", linewidth=1.2,
             label=r"$\lambda_t$ (GRL strength)")
    ax2.set_ylabel(r"$\lambda_t$", fontsize=10, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, 0.15)

    fig.suptitle("LASE r1 training: speaker loss drops, language loss stays at chance",
                 fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "loss_curves.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_boxplot()
    fig_loss_curves()
