from typing import Any, Dict, List, Optional

import os

os.environ["MPLBACKEND"] = "Agg"

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .environment import SREEnvironment
from .models import SREAction


class ResetRequest(BaseModel):
    task: str = "easy"


TASK_CONFIGS: List[Dict[str, Any]] = [
    {"name": "easy", "max_steps": 8, "description": "Static lead mode, single fault"},
    {"name": "medium", "max_steps": 12, "description": "Hidden lead mode, 2 faults"},
    {"name": "hard", "max_steps": 20, "description": "Drifting lead, cascade, 20% coincident"},
]

VALID_TASKS = {"easy", "medium", "hard"}

# IMPORTANT: single global environment instance
env = SREEnvironment()

app = FastAPI(title="AdaptiveSRE", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None) -> Dict[str, Any]:
    task = payload.task if payload else "easy"
    if task not in VALID_TASKS:
        task = "easy"
    observation = env.reset(task)
    return observation.model_dump()


@app.post("/step")
def step(action: SREAction) -> Dict[str, Any]:
    result = env.step(action)
    observation = result["observation"].model_dump()
    return {
        "observation": observation,
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"],
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state().model_dump()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def tasks() -> List[Dict[str, Any]]:
    return TASK_CONFIGS


def _alignment_html(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    color = "#16a34a" if clamped >= 0.67 else "#d97706" if clamped >= 0.34 else "#dc2626"
    return (
        "<div style='font-size:42px; font-weight:700; color:" + color + ";'>"
        f"Alignment Score: {clamped:.2f}</div>"
    )


def _health_bar(health: float, width: int = 20) -> str:
    clamped = max(0.0, min(1.0, health))
    filled = int(round(clamped * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _snapshot(done_override: Optional[bool] = None) -> tuple[Any, ...]:
    st = env.state()
    done = env.done if done_override is None else done_override

    table_rows: List[List[str]] = []
    for service_name, svc in st.services.items():
        health_val = float(svc.get("health", 0.0))
        table_rows.append([service_name, f"{health_val:.3f}", _health_bar(health_val)])

    rewards = env.reward_history[-10:]
    rewards_text = ", ".join(f"{r:.4f}" for r in rewards) if rewards else "[]"
    lead_mode_display = st.lead_mode if done else "???"

    return (
        str(st.step_number),
        _alignment_html(float(st.alignment_score)),
        table_rows,
        rewards_text,
        lead_mode_display,
        str(done),
    )


def _command_to_approach(command: str) -> str:
    c = command.lower()
    if "scale" in c:
        return "scale"
    if "restart" in c or "recover" in c:
        return "restart"
    if "rollback" in c:
        return "rollback"
    if "debug" in c or "inspect" in c or "logs" in c:
        return "debug"
    return "probe"


def _ui_reset(task: str) -> tuple[Any, ...]:
    env.reset(task)
    return _snapshot(done_override=False)


def _ui_step(command: str) -> tuple[Any, ...]:
    cmd = (command or "").strip()
    if not cmd:
        cmd = "docker stats auth"

    action = SREAction(
        command=cmd,
        reasoning="Manual UI command",
        approach=_command_to_approach(cmd),
        drift_detected=False,
        lead_mode_guess="unknown",
        root_cause_guess=None,
    )
    env.step(action)
    return _snapshot()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AdaptiveSRE") as demo:
        gr.Markdown("## AdaptiveSRE Control Panel")

        with gr.Row():
            btn_easy = gr.Button("Reset Easy")
            btn_medium = gr.Button("Reset Medium")
            btn_hard = gr.Button("Reset Hard")

        step_number = gr.Textbox(label="Current Step", interactive=False)
        alignment_display = gr.HTML(label="Alignment")

        services_table = gr.Dataframe(
            headers=["service", "health", "bar"],
            datatype=["str", "str", "str"],
            row_count=(5, "fixed"),
            col_count=(3, "fixed"),
            label="Services",
            interactive=False,
        )

        rewards_last_10 = gr.Textbox(label="Last 10 rewards", interactive=False)
        lead_mode = gr.Textbox(label="Lead Mode", interactive=False)
        done_flag = gr.Textbox(label="Done", interactive=False)

        with gr.Row():
            command_input = gr.Textbox(label="Step Command", placeholder="docker stats auth")
            step_btn = gr.Button("Step")

        outputs = [step_number, alignment_display, services_table, rewards_last_10, lead_mode, done_flag]

        btn_easy.click(lambda: _ui_reset("easy"), outputs=outputs)
        btn_medium.click(lambda: _ui_reset("medium"), outputs=outputs)
        btn_hard.click(lambda: _ui_reset("hard"), outputs=outputs)
        step_btn.click(_ui_step, inputs=[command_input], outputs=outputs)
        command_input.submit(_ui_step, inputs=[command_input], outputs=outputs)

        demo.load(lambda: _ui_reset("easy"), outputs=outputs)

    return demo


demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
