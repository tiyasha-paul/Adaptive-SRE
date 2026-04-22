from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class SREObservation(BaseModel):
    alert_text: str
    command_output: str
    services_status: Dict[str, Dict]
    symptom_fingerprints: List[Dict]
    last_reward: float
    reward_history: List[float]
    step_number: int
    episode_id: str


class SREAction(BaseModel):
    command: str
    reasoning: str
    approach: Literal["scale", "restart", "debug", "rollback", "probe"]
    drift_detected: bool
    lead_mode_guess: Literal["paranoia", "budget", "velocity", "unknown"]
    root_cause_guess: Optional[str] = None


class SREReward(BaseModel):
    total_score: float
    incident_score: float
    alignment_score: float
    drift_score: float
    root_cause_bonus: float
    breakdown: Dict[str, float]


class SREState(BaseModel):
    episode_id: str
    step_number: int
    lead_mode: str
    drift_occurred: bool
    drift_step: Optional[int]
    services: Dict[str, Dict]
    alignment_score: float
    cumulative_reward: float
