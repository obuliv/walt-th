"""Compares reward-model training runs logged by train.py (see tracking.log_run):
prints a table and plots the headline metrics (top1_accuracy, pairwise_accuracy, mrr)
across runs in chronological order, so newer approaches can be checked against
earlier ones at a glance.

Usage:
    python -m walt.rm.model.visualize --runs-dir data/output/runs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from walt.rm.model.tracking import load_runs

METRIC_KEYS = ["top1_accuracy", "pairwise_accuracy", "mrr"]

# dataviz skill reference palette, categorical slots 1-3 (blue/orange/aqua) — the only
# three slots validated for all-pairs CVD separation, which is exactly how many series
# this chart needs. Chart chrome (surface/ink/gridline) is the same reference palette's
# light-mode values; this is a static PNG, not a themed page, so only light mode applies.
SERIES_COLORS = {"top1_accuracy": "#2a78d6", "pairwise_accuracy": "#eb6834", "mrr": "#1baf7a"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


def print_table(runs: list[dict]) -> None:
    headers = ["run_name", "timestamp", "embedding_model", *METRIC_KEYS, "n_train", "n_test"]
    rows = []
    for run in runs:
        cfg = run["config"]
        metrics = run["metrics"]
        rows.append(
            [
                run["run_name"],
                run["timestamp"],
                cfg.get("embedding_model", "?"),
                *(f"{metrics.get(k, float('nan')):.4f}" for k in METRIC_KEYS),
                str(cfg.get("n_train", "?")),
                str(cfg.get("n_test", "?")),
            ]
        )
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(line_fmt.format(*headers))
    print(line_fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(line_fmt.format(*row))


def plot_comparison(runs: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = list(range(len(runs)))
    labels = [run["run_name"] for run in runs]

    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for key in METRIC_KEYS:
        y = [run["metrics"].get(key, float("nan")) for run in runs]
        ax.plot(
            x,
            y,
            label=key,
            color=SERIES_COLORS[key],
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgewidth=0,
        )
        # selective direct label: annotate only the last point of each line (<=4 series)
        ax.annotate(
            f"{y[-1]:.3f}",
            xy=(x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=SERIES_COLORS[key],
            fontsize=9,
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, len(x) - 1 + 0.6)  # headroom on the right for endpoint value labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=INK_MUTED)
    ax.set_ylabel("score", color=INK_SECONDARY)
    ax.set_title("Reward model comparison across runs", color=INK_PRIMARY, fontsize=13, loc="left")

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        spine.set_color(AXIS if spine_name == "bottom" else "none")

    # placed in the axes' empty lower region rather than near the data/endpoint labels,
    # which all sit in the upper part of the 0-1 range for this metric set
    legend = ax.legend(frameon=False, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    print(f"Wrote chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/runs"), help="Directory of run records written by train.py")
    parser.add_argument("--output", type=Path, default=None, help="Chart output path (default: <runs-dir>/comparison.png)")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        return

    print_table(runs)
    plot_comparison(runs, args.output or args.runs_dir / "comparison.png")


if __name__ == "__main__":
    main()
