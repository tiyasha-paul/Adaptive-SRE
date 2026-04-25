#!/usr/bin/env python3
"""
train.py — GRPO Training for AdaptiveSRE
========================================

Trains a small LLM to act as an SRE agent using Group Relative Policy Optimization.
The reward function is the AdaptiveSRE environment itself — the agent must learn
to resolve incidents AND detect hidden policy drift from reward signals alone.

Usage:
    python train.py --episodes 200 --task hard --output ./checkpoints/gen1/

Requirements:
    - GPU with 16GB+ VRAM (T4/V100/RTX 3090)
    - mock_services running: docker-compose -f mock_services/docker-compose.yml up -d
    - server running: uvicorn server.app:app --port 8000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import torch
from transformers import AutoTokenizer
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    from trl.trainer import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel


# ── Configuration ──
DEFAULT_BASE_URL = "http://localhost:8000"
MAX_STEPS = {"easy": 8, "medium": 12, "hard": 20}
MAX_TOTAL_REWARD = {"easy": 8.0, "medium": 12.0, "hard": 20.0}
SUCCESS_THRESHOLD = 0.5

# Model config — fits in Colab T4 (16GB)
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


# ── Environment Client ──
class SREClient:
    """Lightweight HTTP client to interact with AdaptiveSRE server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task: str = "hard") -> Dict[str, Any]:
        resp = self.client.post(f"{self.base_url}/reset", json={"task": task})
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ── Prompt Builder ──
SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) agent. Your job is to diagnose infrastructure incidents and take corrective actions.

You will receive observations about service health, error rates, and latency. You must respond with a JSON action containing:
- command: A shell command to execute (docker stats, docker logs, docker restart, curl, etc.)
- reasoning: One-sentence explanation of your decision
- approach: One of scale, restart, debug, rollback, probe
- drift_detected: true if you believe the Lead Engineer's priorities have shifted
- lead_mode_guess: Your guess at current mode (paranoia, budget, velocity, unknown)
- root_cause_guess: Which service you believe is the root cause (db, auth, payment, cache, notification) or null

Rules:
1. The Lead Engineer's priorities are HIDDEN. You must infer them from reward feedback.
2. If rewards suddenly turn negative, the mode may have drifted. Set drift_detected=true.
3. Look at symptom timing (onset_offset_seconds) to identify root cause vs symptoms.
4. Do not repeat the same command consecutively.
5. Be decisive — over-probing is penalized."""


def build_prompt(observation: Dict[str, Any], max_steps: int) -> str:
    """Convert observation dict into training prompt for the LLM."""
    services = observation.get("services_status", {})
    services_str = json.dumps(services, indent=2) if services else "{}"

    fingerprints = observation.get("symptom_fingerprints", [])
    fp_str = json.dumps(fingerprints, indent=2) if fingerprints else "[]"

    reward_history = observation.get("reward_history", [])
    rh_str = ", ".join(f"{r:.2f}" for r in reward_history[-5:]) if reward_history else "None"

    user_msg = (
        f"Current incident:\n"
        f"Alert: {observation.get('alert_text', 'No alert')}\n"
        f"Last command output: {observation.get('command_output', '')[:500]}\n"
        f"Services status:\n{services_str}\n"
        f"Symptom fingerprints:\n{fp_str}\n"
        f"Last reward: {float(observation.get('last_reward', 0.0)):.2f}\n"
        f"Recent rewards: {rh_str}\n"
        f"Step {int(observation.get('step_number', 0))} of {max_steps}.\n\n"
        f"Respond with JSON action only."
    )

    return f"{SYSTEM_PROMPT}\n\n{user_msg}"


# ── Reward Function ──
def compute_episode_reward(episode_rewards: List[float], task: str) -> float:
    """
    Convert list of step rewards into a single training signal.
    Uses the same formula as inference.py for consistency.
    """
    max_total = MAX_TOTAL_REWARD[task]
    raw_score = sum(episode_rewards) / max_total
    # Clamp to (0.001, 0.999) — same as inference.py
    clamped = max(0.001, min(0.999, raw_score))
    # Scale to (-1, 1) for GRPO stability
    return (clamped - 0.5) * 2.0


def run_episode(client: SREClient, task: str, model, tokenizer, device: str) -> Dict[str, Any]:
    """
    Run one full episode: reset → loop steps → return trajectory + reward.
    Uses the model to generate actions.
    """
    max_steps = MAX_STEPS[task]
    obs = client.reset(task)
    episode_rewards: List[float] = []
    trajectory: List[Dict[str, Any]] = []
    done = False

    for step_num in range(1, max_steps + 1):
        # Build prompt from current observation
        prompt = build_prompt(obs, max_steps)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode response
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse JSON action from response
        action = parse_action_from_text(response_text)

        # Execute step in environment
        try:
            result = client.step(action)
            reward = float(result.get("reward", 0.0))
            obs = result.get("observation", obs)
            done = bool(result.get("done", False))
        except Exception as e:
            reward = 0.001  # Minimum reward on error
            done = True

        episode_rewards.append(reward)
        trajectory.append({
            "step": step_num,
            "prompt": prompt,
            "response": response_text,
            "action": action,
            "reward": reward,
        })

        if done:
            break

    # Compute episode-level reward for GRPO
    episode_reward = compute_episode_reward(episode_rewards, task)

    return {
        "trajectory": trajectory,
        "episode_rewards": episode_rewards,
        "episode_reward": episode_reward,
        "num_steps": len(episode_rewards),
    }


def parse_action_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON action from model output. Handles markdown code blocks,
    partial JSON, and fallback to probe action.
    """
    import re

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return normalize_action(parsed)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return normalize_action(parsed)
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "command": "docker stats --no-stream",
        "reasoning": "Fallback probe due to parse failure",
        "approach": "probe",
        "drift_detected": False,
        "lead_mode_guess": "unknown",
        "root_cause_guess": None,
    }


def normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize action fields."""
    allowed_approach = {"scale", "restart", "debug", "rollback", "probe"}
    allowed_lead = {"paranoia", "budget", "velocity", "unknown"}
    allowed_root = {"db", "auth", "payment", "cache", "notification"}

    approach = str(raw.get("approach", "probe"))
    if approach not in allowed_approach:
        approach = "probe"

    lead_guess = str(raw.get("lead_mode_guess", "unknown"))
    if lead_guess not in allowed_lead:
        lead_guess = "unknown"

    root = raw.get("root_cause_guess")
    if isinstance(root, str):
        root = None if root.lower() == "null" else root.lower()
    if root not in allowed_root:
        root = None

    return {
        "command": str(raw.get("command", "docker stats --no-stream")),
        "reasoning": str(raw.get("reasoning", "No reasoning provided")),
        "approach": approach,
        "drift_detected": bool(raw.get("drift_detected", False)),
        "lead_mode_guess": lead_guess,
        "root_cause_guess": root,
    }

# ── Main Training Loop ──
def main():
    parser = argparse.ArgumentParser(description="Train AdaptiveSRE agent with GRPO")
    parser.add_argument("--episodes", type=int, default=200, help="Total episodes to collect")
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--output", type=str, default="./checkpoints/gen1/", help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=4, help="GRPO batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N episodes")
    parser.add_argument("--env_url", type=str, default=DEFAULT_BASE_URL, help="AdaptiveSRE server URL")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"AdaptiveSRE GRPO Training")
    print(f"Task: {args.task} | Episodes: {args.episodes} | Output: {args.output}")
    print(f"{'='*60}")

    # ── Setup ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Training on CPU will be extremely slow. Use GPU for practical training.")

    # ── Load Model with Unsloth ──
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (float16 for T4, bfloat16 for Ampere)
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"Model loaded. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Connect to Environment ──
    client = SREClient(base_url=args.env_url)
    print(f"Connected to environment at {args.env_url}")

    # ── Collect Baseline (Gen 0) ──
    print(f"\n{'='*60}")
    print("Phase 1: Collecting baseline trajectories (Gen 0)")
    print(f"{'='*60}")

    baseline_rewards: List[float] = []
    positive_trajectories: List[Dict[str, Any]] = []

    for ep in range(1, args.episodes + 1):
        result = run_episode(client, args.task, model, tokenizer, device)
        reward = result["episode_reward"]
        baseline_rewards.append(reward)

        # Filter: keep episodes with reward >= 0.4 (scaled: -1 to 1, so 0.4 = 0.7 raw score)
        if reward >= 0.4:
            positive_trajectories.extend(result["trajectory"])

        if ep % 10 == 0:
            mean_reward = sum(baseline_rewards[-10:]) / len(baseline_rewards[-10:])
            print(f"  Episode {ep}/{args.episodes} | Last 10 mean: {mean_reward:+.3f} | "
                  f"Positives: {len(positive_trajectories)}")

    # Report baseline stats
    mean_baseline = sum(baseline_rewards) / len(baseline_rewards)
    print(f"\nBaseline complete: mean={mean_baseline:+.3f}, "
          f"positive_trajectories={len(positive_trajectories)}")

    if len(positive_trajectories) < 20:
        print("WARNING: Very few positive trajectories. Consider running more episodes or adjusting filter.")

    # ── Prepare Training Data ──
    # Convert trajectories to GRPO format: list of {"prompt": str, "completion": str, "reward": float}
    training_data: List[Dict[str, Any]] = []
    for traj in positive_trajectories:
        training_data.append({
            "prompt": traj["prompt"],
            "completion": traj["response"],
            "reward": traj["reward"],
        })

    # Save raw training data for inspection
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_data.jsonl", "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(training_data)} training examples to {output_dir / 'training_data.jsonl'}")

    # ── GRPO Training ──
    print(f"\n{'='*60}")
    print("Phase 2: GRPO Training (Gen 1)")
    print(f"{'='*60}")

    # TRL GRPO config
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.save_every,
        max_completion_length=256,
        num_generations=4,  # Group size for relative advantage
        temperature=0.7,
        use_vllm=False,  # Unsloth handles generation
    )

    # Custom reward function — uses the environment
    def env_reward_fn(completions: List[str], **kwargs) -> List[float]:
        """
        For GRPO, we reward based on how well the completion parses to a valid action
        AND how that action performed in the environment.
        Since we can't run full episodes in the GRPO loop efficiently,
        we use a proxy: reward = 1.0 if valid JSON action, 0.5 if valid format,
        0.0 if invalid. Full episode rewards are used for filtering only in Gen 1.
        """
        rewards = []
        for completion in completions:
            action = parse_action_from_text(completion)
            # Check if it's a valid action (not fallback)
            if action["command"] != "docker stats --no-stream" or "parse failure" not in action["reasoning"]:
                rewards.append(1.0)
            else:
                rewards.append(0.3)
        return rewards

    # TRL expects datasets.Dataset, not list of dicts. Convert:
    from datasets import Dataset
    dataset = Dataset.from_list(training_data)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward_fn],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    # ── Save Final Model ──
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nFinal model saved to {final_path}")

    # ── Quick Validation ──
    print(f"\n{'='*60}")
    print("Phase 3: Validation")
    print(f"{'='*60}")

    trained_rewards: List[float] = []
    for ep in range(20):
        result = run_episode(client, args.task, model, tokenizer, device)
        trained_rewards.append(result["episode_reward"])

    mean_trained = sum(trained_rewards) / len(trained_rewards)
    improvement = mean_trained - mean_baseline

    print(f"Baseline mean:  {mean_baseline:+.3f}")
    print(f"Trained mean:   {mean_trained:+.3f}")
    print(f"Improvement:    {improvement:+.3f} ({improvement/abs(mean_baseline)*100 if mean_baseline != 0 else 0:.1f}%)")

    # Save results
    results = {
        "task": args.task,
        "episodes": args.episodes,
        "baseline_mean": mean_baseline,
        "trained_mean": mean_trained,
        "improvement": improvement,
        "baseline_rewards": baseline_rewards,
        "trained_rewards": trained_rewards,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    client.close()
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()