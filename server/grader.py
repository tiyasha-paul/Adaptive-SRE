from typing import List, Dict
from .models import SREAction, SREReward
from .service_graph import ServiceGraph
from .lead_engineer import LeadEngineer


class Grader:
    APPROACH_ALIGNMENT = {
        "paranoia": {"scale": 1.0, "restart": 0.2, "debug": 0.4, "rollback": 0.5, "probe": 0.6},
        "budget": {"restart": 1.0, "debug": 0.9, "rollback": 0.7, "probe": 0.5, "scale": 0.0},
        "velocity": {"scale": 0.7, "restart": 0.8, "debug": 0.5, "rollback": 0.6, "probe": 0.2}
    }

    def __init__(self):
        self.last_alignment_score: float = 0.0
        self.command_history: List[str] = []
        self.approach_history: List[str] = []

    def _clamp(self, value: float) -> float:
        return max(0.001, min(0.999, round(value, 4)))

    def _count_degrading_services(self, graph: ServiceGraph) -> int:
        count = 0
        for svc in graph.services.values():
            if svc.health < 0.9 or svc.error_rate > 0.1:
                count += 1
        return count

    def _command_errored(self, command_output: str) -> bool:
        return "Error" in command_output or "not found" in command_output or "error" in command_output.lower()

    def _command_repeated(self, command: str) -> bool:
        return command in self.command_history[-3:]

    def compute_alignment_score(self, lead_mode: str) -> float:
        if lead_mode not in self.APPROACH_ALIGNMENT:
            return 0.5
        if len(self.approach_history) == 0:
            return 0.5
        scores = []
        for approach in self.approach_history[-3:]:
            score = self.APPROACH_ALIGNMENT[lead_mode].get(approach, 0.5)
            scores.append(score)
        return sum(scores) / len(scores)

    def score(self, action: SREAction, service_graph: ServiceGraph,
              lead_engineer: LeadEngineer, prev_graph_state: dict,
              step_number: int, command_output: str = "") -> SREReward:

        self.command_history.append(action.command)
        if len(self.command_history) > 10:
            self.command_history.pop(0)
        self.approach_history.append(action.approach)
        if len(self.approach_history) > 10:
            self.approach_history.pop(0)

        incident_score = 0.0
        services_restored = 0
        services_improved = 0
        prev_degrading = sum(1 for k, v in prev_graph_state.items() if v.get("health", 1.0) < 0.9)
        curr_degrading = self._count_degrading_services(service_graph)

        for name, svc in service_graph.services.items():
            prev_health = prev_graph_state.get(name, {}).get("health", 1.0)
            health_improvement = svc.health - prev_health
            if health_improvement > 0.3:
                services_improved += 1
                incident_score += 0.3
            if svc.health > 0.9 and prev_health < 0.9:
                services_restored += 1
                incident_score += 1.0

        if curr_degrading < prev_degrading and prev_degrading > 0:
            incident_score += 0.2

        if self._command_errored(command_output):
            incident_score -= 0.2

        if self._command_repeated(action.command):
            incident_score -= 0.15

        if action.approach == "probe":
            probe_count = sum(1 for a in self.approach_history[-5:] if a == "probe")
            if probe_count > 4:
                incident_score -= 0.05 * (probe_count - 4)

        if action.approach == "probe" and services_restored == 0 and services_improved == 0:
            incident_score -= 0.1

        alignment_reward = lead_engineer.compute_policy_alignment(action.approach)
        alignment_score = self._clamp(alignment_reward)

        drift_score = 0.0
        if action.drift_detected and lead_engineer.drift_occurred:
            drift_score += 0.5
        elif action.drift_detected and not lead_engineer.drift_occurred:
            drift_score -= 0.2
        elif not action.drift_detected and lead_engineer.drift_occurred:
            drift_score -= 0.1

        if action.lead_mode_guess == lead_engineer.mode:
            drift_score += 0.3

        root_cause_bonus = 0.0
        root_cause_services = [name for name, svc in service_graph.services.items() if svc.is_root_cause]
        if action.root_cause_guess:
            if action.root_cause_guess in root_cause_services:
                root_cause_bonus += 0.3
                prev_health = prev_graph_state.get(action.root_cause_guess, {}).get("health", 1.0)
                curr_health = service_graph.services[action.root_cause_guess].health
                if curr_health > prev_health:
                    root_cause_bonus += 0.2
            else:
                root_cause_bonus -= 0.1

        total_score = incident_score + alignment_score + drift_score + root_cause_bonus

        self.last_alignment_score = self.compute_alignment_score(lead_engineer.mode)

        return SREReward(
            total_score=self._clamp(total_score),
            incident_score=self._clamp(incident_score),
            alignment_score=self._clamp(alignment_score),
            drift_score=self._clamp(drift_score),
            root_cause_bonus=self._clamp(root_cause_bonus),
            breakdown={
                "incident": round(incident_score, 4),
                "policy_alignment": round(alignment_reward, 4),
                "drift_detection": round(drift_score, 4),
                "root_cause": round(root_cause_bonus, 4),
                "total": round(total_score, 4)
            }
        )
