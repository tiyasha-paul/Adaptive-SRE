import copy
import random
import uuid
from typing import Dict, List, Optional

from .models import SREObservation, SREAction, SREState
from .service_graph import ServiceGraph
from .lead_engineer import LeadEngineer
from .grader import Grader
from .docker_executor import DockerExecutor
from .fault_injector import FaultInjector


class SREEnvironment:
    def __init__(self):
        self.graph = ServiceGraph()
        self.lead = LeadEngineer()
        self.grader = Grader()
        self.executor = DockerExecutor()
        self.injector = FaultInjector()
        self.task = "easy"
        self.step_num = 0
        self.episode_id: Optional[str] = None
        self.reward_history: List[float] = []
        self.command_history: List[str] = []
        self.done = False
        self.max_steps = {"easy": 8, "medium": 12, "hard": 20}
        self.alert_text = ""

    def reset(self, task: str = "easy") -> SREObservation:
        # CRITICAL: Full clean wipe - no state leak from previous episode
        self.task = task
        self.episode_id = str(uuid.uuid4())
        self.step_num = 0
        self.reward_history = []
        self.command_history = []
        self.done = False

        # Reset subsystems
        self.graph.reset()
        self.lead.reset(task)
        self.grader = Grader()  # Fresh grader with empty history

        # Inject initial fault based on task
        if task == "easy":
            self.alert_text = self.injector.inject_cascade(self.graph, "auth", "crash_loop")
        elif task == "medium":
            self.alert_text = self.injector.inject_cascade(self.graph, "db", "connection_exhaustion")
        elif task == "hard":
            # Full cascade: db -> auth -> payment
            self.alert_text = self.injector.inject_cascade(self.graph, "db", "oom_kill")
            # 20% chance of coincident independent fault
            if random.random() < 0.2:
                self.injector.inject_coincident(self.graph, "db", "notification", "oom_kill", "crash_loop")
                self.alert_text += "\n\n[SECONDARY ALERT] Independent failure detected."

        return SREObservation(
            alert_text=self.alert_text,
            command_output="",
            services_status=self.graph.get_observation_dict(),
            symptom_fingerprints=self.graph.get_symptom_fingerprints(),
            last_reward=0.0,
            reward_history=[],
            step_number=0,
            episode_id=self.episode_id
        )

    def step(self, action: SREAction) -> Dict:
        # 1. Increment step counter
        self.step_num += 1

        # 2. CRITICAL: check_drift BEFORE computing reward
        drifted = self.lead.check_drift(self.step_num)

        # 3. Save prev_graph_state for comparison
        prev_graph_state = {
            name: {"health": svc.health, "error_rate": svc.error_rate}
            for name, svc in self.graph.services.items()
        }

        # 4. Execute command
        output = self.executor.execute(action.command)

        # 5. Apply recovery if restart/recover command
        if "restart" in action.command.lower() or "recover" in action.command.lower():
            service_name = self._extract_service_from_command(action.command)
            if service_name and service_name in self.graph.services:
                self.graph.apply_recover(service_name)

        # 6. Propagate degradation
        self.graph.propagate(dt=1.0)

        # 7. Compute reward
        reward = self.grader.score(
            action=action,
            service_graph=self.graph,
            lead_engineer=self.lead,
            prev_graph_state=prev_graph_state,
            step_number=self.step_num,
            command_output=output
        )

        # 8. Record history
        self.reward_history.append(reward.total_score)
        self.command_history.append(action.command)

        # 10. Check done condition
        self.done = self._check_done()

        # 11. Return result
        return {
            "observation": SREObservation(
                alert_text=self.alert_text,
                command_output=output,
                services_status=self.graph.get_observation_dict(),
                symptom_fingerprints=self.graph.get_symptom_fingerprints(),
                last_reward=reward.total_score,
                reward_history=self.reward_history.copy(),
                step_number=self.step_num,
                episode_id=self.episode_id
            ),
            "reward": reward.total_score,
            "done": self.done,
            "info": {
                "reward_breakdown": reward.breakdown,
                "drift_occurred": drifted
            }
        }

    def state(self) -> SREState:
        # Returns full state including hidden fields (for debugging/UI)
        services_dict = {}
        for name, svc in self.graph.services.items():
            services_dict[name] = {
                "health": svc.health,
                "latency_ms": svc.latency_ms,
                "error_rate": svc.error_rate,
                "cpu_pct": svc.cpu_pct,
                "is_root_cause": svc.is_root_cause,
                "onset_timestamp": svc.onset_timestamp
            }

        return SREState(
            episode_id=self.episode_id or "",
            step_number=self.step_num,
            lead_mode=self.lead.mode or "unknown",
            drift_occurred=self.lead.drift_occurred,
            drift_step=self.lead.drift_step,
            services=services_dict,
            alignment_score=self.grader.last_alignment_score,
            cumulative_reward=sum(self.reward_history)
        )

    def _extract_service_from_command(self, command: str) -> Optional[str]:
        service_names = ["db", "auth", "payment", "cache", "notification"]
        cmd_lower = command.lower()
        for svc in service_names:
            if svc in cmd_lower:
                return svc
        return None

    def _check_done(self) -> bool:
        # Check max steps
        if self.step_num >= self.max_steps.get(self.task, 20):
            return True

        # Check if all services healthy
        all_healthy = all(
            svc.health > 0.85
            for svc in self.graph.services.values()
        )
        if all_healthy:
            return True

        return False
