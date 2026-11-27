"""Plot Part II metrics from the archived analysis and training summaries."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("original", "teacher", "sft", "opd")
LABELS = ("Original", "Teacher", "SFT", "OPD")
COLORS = ("#526478", "#292d44", "#137d79", "#a94c2b")


def grouped_bars(ax, groups, series, labels, colors):
    width = 0.78 / len(series)
    for index, (values, label, color) in enumerate(zip(series, labels, colors, strict=True)):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar([i + offset for i in range(len(groups))], values, width, label=label, color=color)
    ax.set_xticks(range(len(groups)), groups)


def save(fig, output, name):
    fig.savefig(output / name, dpi=190)
    plt.close(fig)


def main():
    root = Path(sys.argv[1])
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "font.size": 10})

    with (root / "summary.csv").open() as source:
        scores = {(row["benchmark"], row["variant"]): row for row in csv.DictReader(source)}
    metrics = ["pass@1", "pass@2", "pass@4"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, [("math", "MATH-500"), ("ifeval", "IFEval")], strict=True):
        series = [[100 * float(scores[task, arm][key]) for key in metrics] for arm in ARMS]
        grouped_bars(ax, metrics, series, LABELS, COLORS)
        ax.set(title=title, ylim=(0, 100))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Pass rate (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    save(fig, output, "benchmark-pass-rates.png")

    analysis = json.loads((root / "analysis.json").read_text())
    levels = analysis["math_by_level"]
    order = sorted(levels)
    fig, ax = plt.subplots(figsize=(7.7, 3.7), layout="constrained")
    series = [[100 * levels[level]["pass_at_1"][arm] for level in order] for arm in ARMS]
    grouped_bars(ax, [f"{level} (n={levels[level]['questions']})" for level in order],
                 series, LABELS, COLORS)
    ax.set(title="MATH-500 by difficulty", ylabel="Pass@1 (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncols=4, fontsize=8)
    save(fig, output, "math-difficulty.png")

    names = ("retained", "lost", "gained", "still_failing")
    transitions = analysis["ifeval_transitions"]
    fig, ax = plt.subplots(figsize=(6.8, 3.7), layout="constrained")
    grouped_bars(ax, ("Retained", "Lost", "Gained", "Still failing"),
                 [[transitions[arm][name] for name in names] for arm in ARMS[2:]],
                 LABELS[2:], COLORS[2:])
    ax.set(title="IFEval pass@4 transitions from the original model", ylabel="Prompt count")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    save(fig, output, "ifeval-transitions.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), layout="constrained")
    for ax, arm, label, color in zip(axes, ARMS[2:], LABELS[2:], COLORS[2:], strict=True):
        run = json.loads((root / f"{arm}-training.json").read_text())
        points = [(row["step"], row["loss"]) for row in run["history"]
                  if "step" in row and "loss" in row]
        if not points:
            raise ValueError(f"No logged loss for {arm}")
        steps, losses = zip(*points)
        ax.plot(steps, losses, color=color, linewidth=1.4, marker=".", markersize=3)
        ax.set(title=f"{label} ({run['steps']} steps)", xlabel="Training step",
               ylabel=f"{label} training loss")
        ax.grid(alpha=.2)
    fig.suptitle("Training objectives on separate scales")
    save(fig, output, "training-loss.png")


if __name__ == "__main__":
    main()
