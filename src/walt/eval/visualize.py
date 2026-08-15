"""Compares agent-eval runs logged by evaluate.py (see tracking.log_run — reused as-is,
just pointed at data/output/eval_runs/ instead of RM training's data/output/runs/, since
the metric shape here is different): prints a table and plots sql_pass_rate/qa_accuracy,
each with vs without RM reranking, across runs in chronological order — the RM-vs-no-RM
gap is exactly what should widen as the RM/data/LLM improve.

Usage:
    python -m walt.eval.visualize --runs-dir data/output/eval_runs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from walt.rm.model.tracking import load_runs

TABLE_KEYS = [
    "rm_top1_accuracy",
    "sql_pass_rate_with_rm",
    "sql_pass_rate_without_rm",
    "qa_accuracy_with_rm",
    "qa_accuracy_without_rm",
    "oracle_ceiling",
]

# Two series *groups* (sql_pass_rate, qa_accuracy), each split with_rm/without_rm — drawn
# as color-per-group + linestyle-per-condition rather than 4 distinct colors, since only
# the first two dataviz reference-palette slots (blue/orange) are validated for all-pairs
# CVD separation (see rm/model/visualize.py's SERIES_COLORS comment).
GROUP_COLORS = {"sql_pass_rate": "#2a78d6", "qa_accuracy": "#eb6834"}
LINESTYLES = {"with_rm": "-", "without_rm": "--"}
# oracle_ceiling has no with/without split (it's a property of the LLM's candidates, not
# the reranker) — drawn as its own solid line in a third, distinct color rather than
# forced into the group/linestyle pairing above; it's the headroom qa_accuracy_with_rm
# is bounded by, so seeing both trend together over runs is the point.
CEILING_COLOR = "#3fa34d"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


def print_table(runs: list[dict]) -> None:
    headers = ["run_name", "timestamp", "ollama_model", *TABLE_KEYS, "n_val"]
    rows = []
    for run in runs:
        cfg = run["config"]
        metrics = run["metrics"]

        def fmt(k: str) -> str:
            v = metrics.get(k)
            return f"{v:.4f}" if isinstance(v, float) else "n/a"

        rows.append(
            [
                run["run_name"],
                run["timestamp"],
                cfg.get("ollama_model", "?"),
                *(fmt(k) for k in TABLE_KEYS),
                str(cfg.get("n_val_examples", "?")),
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

    for group, color in GROUP_COLORS.items():
        for condition, linestyle in LINESTYLES.items():
            key = f"{group}_{condition}"
            y = [run["metrics"].get(key) for run in runs]
            if all(v is None for v in y):
                continue
            ax.plot(
                x,
                y,
                label=key,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgewidth=0,
            )
            if y[-1] is not None:
                ax.annotate(
                    f"{y[-1]:.3f}",
                    xy=(x[-1], y[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    ceiling_y = [run["metrics"].get("oracle_ceiling") for run in runs]
    if any(v is not None for v in ceiling_y):
        ax.plot(
            x,
            ceiling_y,
            label="oracle_ceiling",
            color=CEILING_COLOR,
            linestyle=":",
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgewidth=0,
        )
        if ceiling_y[-1] is not None:
            ax.annotate(
                f"{ceiling_y[-1]:.3f}",
                xy=(x[-1], ceiling_y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=CEILING_COLOR,
                fontsize=9,
            )

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, len(x) - 1 + 0.6)  # headroom on the right for endpoint value labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=INK_MUTED)
    ax.set_ylabel("score", color=INK_SECONDARY)
    ax.set_title("Agent eval comparison across runs", color=INK_PRIMARY, fontsize=13, loc="left")

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        spine.set_color(AXIS if spine_name == "bottom" else "none")

    legend = ax.legend(frameon=False, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    print(f"Wrote chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/output/eval_runs"), help="Directory of run records written by evaluate.py")
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
