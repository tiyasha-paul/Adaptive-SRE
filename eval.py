#!/usr/bin/env python3
"""
eval.py — Compare Gen 0 (baseline) vs Gen 1 (trained) across all tasks
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from train import SREClient, run_episode, DEFAULT_BASE_URL


TASKS = ["easy", "medium", "hard"]
EPISODES_PER_TASK = 20
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


def load_model(checkpoint_path: str):
    """Load a trained checkpoint."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def evaluate_model(model, tokenizer, task: str, env_url: str, episodes: int = 20) -> Dict:
    """Run N episodes and collect statistics."""
    client = SREClient(base_url=env_url)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rewards: List[float] = []
    steps: List[int] = []
    drift_detections: List[bool] = []

    for ep in range(episodes):
        result = run_episode(client, task, model, tokenizer, device)
        rewards.append(result["episode_reward"])
        steps.append(result["num_steps"])

        # Check if drift was detected (look through trajectory)
        detected = False
        for traj in result["trajectory"]:
            if traj["action"].get("drift_detected"):
                detected = True
                break
        drift_detections.append(detected)

    client.close()

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "std_reward": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5,
        "mean_steps": sum(steps) / len(steps),
        "drift_detection_rate": sum(drift_detections) / len(drift_detections),
        "rewards": rewards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, default=MODEL_NAME,
                       help="Base model for Gen 0 (default: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)")
    parser.add_argument("--trained_model", type=str, required=True,
                       help="Path to Gen 1 checkpoint directory")
    parser.add_argument("--env_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="./eval_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("AdaptiveSRE Evaluation: Gen 0 vs Gen 1")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load Models ──
    print("Loading baseline model (Gen 0)...")
    baseline_model, baseline_tokenizer = load_model(args.baseline_model)

    print(f"Loading trained model (Gen 1) from {args.trained_model}...")
    trained_model, trained_tokenizer = load_model(args.trained_model)

    # ── Evaluate All Tasks ──
    results = {"gen0": {}, "gen1": {}}

    for task in TASKS:
        print(f"\n{'='*70}")
        print(f"Task: {task.upper()}")
        print(f"{'='*70}")

        # Gen 0
        print(f"Running Gen 0 ({args.episodes} episodes)...")
        gen0 = evaluate_model(baseline_model, baseline_tokenizer, task, args.env_url, args.episodes)
        results["gen0"][task] = gen0

        # Gen 1
        print(f"Running Gen 1 ({args.episodes} episodes)...")
        gen1 = evaluate_model(trained_model, trained_tokenizer, task, args.env_url, args.episodes)
        results["gen1"][task] = gen1

        # Print comparison
        improvement = gen1["mean_reward"] - gen0["mean_reward"]
        pct = (improvement / abs(gen0["mean_reward"]) * 100) if gen0["mean_reward"] != 0 else 0

        print(f"\n  Gen 0 mean: {gen0['mean_reward']:+.3f} (±{gen0['std_reward']:.3f})")
        print(f"  Gen 1 mean: {gen1['mean_reward']:+.3f} (±{gen1['std_reward']:.3f})")
        print(f"  Improvement: {improvement:+.3f} ({pct:+.1f}%)")
        print(f"  Drift detection: Gen0={gen0['drift_detection_rate']:.1%}, Gen1={gen1['drift_detection_rate']:.1%}")

    # ── Summary Table ──
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Task':<12} {'Gen0 Mean':<12} {'Gen1 Mean':<12} {'Improvement':<12} {'Drift Detect':<15}")
    print("-" * 70)
    for task in TASKS:
        g0 = results["gen0"][task]["mean_reward"]
        g1 = results["gen1"][task]["mean_reward"]
        imp = g1 - g0
        pct = (imp / abs(g0) * 100) if g0 != 0 else 0
        dd0 = results["gen0"][task]["drift_detection_rate"]
        dd1 = results["gen1"][task]["drift_detection_rate"]
        print(f"{task:<12} {g0:+.3f}       {g1:+.3f}       {imp:+.3f} ({pct:+.1f}%)   {dd0:.0%} → {dd1:.0%}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ── Generate plot ──
    try:
        from plot_rewards import generate_plot_from_eval
        generate_plot_from_eval(results, "rewards_curve.png")
        print("Plot saved to rewards_curve.png")
    except ImportError:
        print("plot_rewards.py not found, skipping plot generation")


if __name__ == "__main__":
    main()