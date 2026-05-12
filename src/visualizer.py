from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {
    "DEFAULT": "#4C72B0",
    "MEMORY_OPTIMIZED": "#DD8452",
    "PARALLEL": "#55A868",
}


def generate_chart(results: dict, output_dir) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    configs = results["configs"]
    names = list(configs.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "EdgeInfer — ONNX Runtime Session Config Comparison",
        fontsize=13, fontweight="bold",
    )

    # ── subplot 1: overlaid latency histograms ──────────────────────────────
    for name in names:
        ax1.hist(
            configs[name]["latencies_ms"],
            bins=30,
            alpha=0.6,
            color=_COLORS[name],
            label=name,
            edgecolor="none",
        )
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Latency Distribution")
    ax1.legend()

    # ── subplot 2: P50 / P99 grouped bar chart ──────────────────────────────
    x = np.arange(len(names))
    width = 0.35
    p50s = [configs[n]["p50_ms"] for n in names]
    p99s = [configs[n]["p99_ms"] for n in names]

    bars_p50 = ax2.bar(
        x - width / 2, p50s, width,
        label="P50",
        color=[_COLORS[n] for n in names],
        alpha=0.85,
    )
    bars_p99 = ax2.bar(
        x + width / 2, p99s, width,
        label="P99",
        color=[_COLORS[n] for n in names],
        alpha=0.45,
        hatch="//",
    )

    for bar in (*bars_p50, *bars_p99):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=10, ha="right")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("P50 vs P99 by Config")
    ax2.legend()

    plt.tight_layout()
    out_path = output_dir / "benchmark_chart.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved → {out_path}")
    return out_path
