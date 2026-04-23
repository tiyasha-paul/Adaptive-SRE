#!/usr/bin/env python3
"""
plot_rewards.py — Generate reward improvement charts for AdaptiveSRE pitch
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "plot_rewards.py requires matplotlib. Install it first, for example: pip install matplotlib"
    ) from exc


def generate_plot(
    baseline_means: Dict[str, float],
    trained_means: Dict[str, float],
    output_path: str = "rewards_curve.png",
    title: str = "AdaptiveSRE: Reward Improvement via GRPO"
):
    """
    Generate a bar chart comparing baseline vs trained rewards per task.
    """
    tasks = ["easy", "medium", "hard"]
    task_labels = ["Easy\n(Static Lead)", "Medium\n(Hidden Lead)", "Hard\n(Drifting Lead)"]

    baseline_vals = [baseline_means.get(t, 0.0) for t in tasks]
    trained_vals = [trained_means.get(t, 0.0) for t in tasks]

    x = range(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([i - width/2 for i in x], baseline_vals, width, label="Gen 0 (Baseline)", color="#ef4444", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], trained_vals, width, label="Gen 1 (GRPO Trained)", color="#22c55e", alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Episode Reward (scaled -1 to +1)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend(loc="upper left")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylim(-1.2, 1.2)

    # Add annotation for hard task
    if trained_vals[2] > baseline_vals[2]:
        ax.annotate("Drift detection\nlearned here",
                    xy=(2, trained_vals[2]),
                    xytext=(2.3, trained_vals[2] + 0.3),
                    arrowprops=dict(arrowstyle="->", color="green"),
                    fontsize=9, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {output_path}")


def generate_plot_from_eval(eval_results: Dict, output_path: str = "rewards_curve.png"):
    """Generate plot from eval.py JSON output."""
    baseline = {task: eval_results["gen0"][task]["mean_reward"] for task in eval_results["gen0"]}
    trained = {task: eval_results["gen1"][task]["mean_reward"] for task in eval_results["gen1"]}
    generate_plot(baseline, trained, output_path)


def main():
    # Check for eval_results.json
    if Path("eval_results.json").exists():
        with open("eval_results.json") as f:
            results = json.load(f)
        generate_plot_from_eval(results, "rewards_curve.png")
        return

    # Fallback: use placeholder data for structure demo
    print("No eval_results.json found. Using placeholder data for demo chart.")
    baseline = {"easy": -0.15, "medium": -0.35, "hard": -0.62}
    trained = {"easy": 0.25, "medium": 0.12, "hard": 0.08}
    generate_plot(baseline, trained, "rewards_curve_placeholder.png")


if __name__ == "__main__":
    main()