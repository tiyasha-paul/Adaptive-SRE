import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "no-key-set")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_NAME = "adaptive-sre"
ENV_HTTP_BASE = os.environ.get("ENV_HTTP_BASE", "http://localhost:8000")
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = {"easy": 8, "medium": 12, "hard": 20}
MAX_TOTAL_REWARD = {"easy": 8.0, "medium": 12.0, "hard": 20.0}

ALLOWED_APPROACH = {"scale", "restart", "debug", "rollback", "probe"}
ALLOWED_LEAD_GUESS = {"paranoia", "budget", "velocity", "unknown"}
ALLOWED_ROOT_CAUSE = {"db", "auth", "payment", "cache", "notification"}

FALLBACK_ACTION: Dict[str, Any] = {
    "command": "docker stats --no-stream",
    "reasoning": "Fallback probe action due to JSON parse failure",
    "approach": "probe",
    "drift_detected": False,
    "lead_mode_guess": "unknown",
    "root_cause_guess": None,
}

USE_REMOTE_MODEL = HF_TOKEN != "no-key-set"

DEFAULT_OBSERVATION: Dict[str, Any] = {
    "alert_text": "",
    "command_output": "",
    "services_status": {},
    "symptom_fingerprints": [],
    "last_reward": 0.0,
    "reward_history": [],
    "step_number": 0,
    "episode_id": "offline",
}


def clamp_score(score: float) -> float:
    return max(0.001, min(0.999, score))


def build_step_prompt(observation: Dict[str, Any], max_steps: int) -> str:
    return (
        "You are an SRE agent. Current observation:\n"
        f"Alert: {observation.get('alert_text', '')}\n"
        f"Last command output: {observation.get('command_output', '')}\n"
        f"Services: {observation.get('services_status', {})}\n"
        f"Last reward: {float(observation.get('last_reward', 0.0)):.2f}\n"
        f"Reward history: {observation.get('reward_history', [])}\n"
        f"Step {int(observation.get('step_number', 0))} of {max_steps}.\n\n"
        "Respond with a JSON object only, no other text:\n"
        "{\n"
        "  \"command\": \"docker stats|docker logs|docker restart|curl http://...\",\n"
        "  \"reasoning\": \"one sentence why\",\n"
        "  \"approach\": \"scale|restart|debug|rollback|probe\",\n"
        "  \"drift_detected\": false,\n"
        "  \"lead_mode_guess\": \"paranoia|budget|velocity|unknown\",\n"
        "  \"root_cause_guess\": \"db|auth|payment|cache|notification|null\"\n"
        "}"
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()

    # Handle ```json ... ``` wrappers.
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\\s*", "", stripped)
        stripped = re.sub(r"\\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def normalize_action(raw_action: Dict[str, Any]) -> Dict[str, Any]:
    action = dict(FALLBACK_ACTION)

    command = str(raw_action.get("command", action["command"]))
    reasoning = str(raw_action.get("reasoning", action["reasoning"]))

    approach = str(raw_action.get("approach", action["approach"]))
    if approach not in ALLOWED_APPROACH:
        approach = action["approach"]

    drift_detected = bool(raw_action.get("drift_detected", action["drift_detected"]))

    lead_mode_guess = str(raw_action.get("lead_mode_guess", action["lead_mode_guess"]))
    if lead_mode_guess not in ALLOWED_LEAD_GUESS:
        lead_mode_guess = action["lead_mode_guess"]

    root_cause_guess = raw_action.get("root_cause_guess", action["root_cause_guess"])
    if isinstance(root_cause_guess, str):
        lowered = root_cause_guess.lower()
        root_cause_guess = None if lowered == "null" else lowered
    if root_cause_guess not in ALLOWED_ROOT_CAUSE:
        root_cause_guess = None

    return {
        "command": command,
        "reasoning": reasoning,
        "approach": approach,
        "drift_detected": drift_detected,
        "lead_mode_guess": lead_mode_guess,
        "root_cause_guess": root_cause_guess,
    }


def choose_action(observation: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    if not USE_REMOTE_MODEL:
        return dict(FALLBACK_ACTION)

    prompt = build_step_prompt(observation, max_steps)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=10,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            return dict(FALLBACK_ACTION)
        return normalize_action(parsed)
    except Exception:
        return dict(FALLBACK_ACTION)


def run_task(task_name: str, http_client: httpx.Client) -> None:
    max_steps = MAX_STEPS[task_name]
    max_total = MAX_TOTAL_REWARD[task_name]

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    observation = dict(DEFAULT_OBSERVATION)
    reset_failed = False
    try:
        reset_resp = http_client.post(f"{ENV_HTTP_BASE}/reset", json={"task": task_name})
        reset_resp.raise_for_status()
        observation = reset_resp.json()
    except Exception:
        reset_failed = True
        observation["step_number"] = 0

    rewards: List[float] = []
    done = False
    step_count = 0

    for step in range(1, max_steps + 1):
        action = choose_action(observation, max_steps)
        action_json = json.dumps(action, separators=(",", ":"))

        error_msg: Optional[str] = None
        reward_value = 0.001

        if reset_failed:
            done = step == max_steps
            error_msg = "timeout"
            observation["step_number"] = step
            observation["last_reward"] = reward_value
            observation["reward_history"] = rewards + [reward_value]
        else:
            try:
                step_resp = http_client.post(f"{ENV_HTTP_BASE}/step", json=action)
                step_resp.raise_for_status()
                step_payload = step_resp.json()

                reward_value = float(step_payload.get("reward", 0.001))
                done = bool(step_payload.get("done", False))
                observation = step_payload.get("observation", observation)
            except httpx.TimeoutException:
                done = step == max_steps
                error_msg = "timeout"
            except Exception:
                done = step == max_steps
                error_msg = "timeout"

        rewards.append(reward_value)
        step_count = step

        error_field = "null" if error_msg is None else error_msg.replace("\n", " ")
        print(
            f"[STEP] step={step} action={action_json} "
            f"reward={reward_value:.2f} done={str(done).lower()} error={error_field}"
            ,
            flush=True,
        )

        if done:
            break

    score = clamp_score(sum(rewards) / max_total)
    success = score >= SUCCESS_SCORE_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={score:.4f} rewards={rewards_str}"
        ,
        flush=True,
    )


def main() -> None:
    with httpx.Client(timeout=30.0) as http_client:
        for task in TASKS:
            run_task(task, http_client)


if __name__ == "__main__":
    main()
