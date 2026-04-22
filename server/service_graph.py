from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
import time


@dataclass
class ServiceState:
    name: str
    health: float = 1.0
    latency_ms: float = 50.0
    error_rate: float = 0.0
    cpu_pct: float = 15.0
    onset_timestamp: Optional[float] = None
    is_root_cause: bool = False


DEPENDENCY_GRAPH = {
    "db": {"auth": 0.7, "cache": 0.4},
    "cache": {"notification": 0.5},
    "auth": {"payment": 0.6},
    "payment": {"api_gateway": 0.5},
    "notification": {}
}


class ServiceGraph:
    def __init__(self):
        self.services: Dict[str, ServiceState] = {
            "db": ServiceState(name="db"),
            "auth": ServiceState(name="auth"),
            "payment": ServiceState(name="payment"),
            "cache": ServiceState(name="cache"),
            "notification": ServiceState(name="notification")
        }
        self.episode_id: str = ""
        self.episode_start_time: float = 0.0
        self.reset()

    def reset(self) -> None:
        self.episode_id = str(uuid.uuid4())
        self.episode_start_time = time.time()
        for service in self.services.values():
            service.health = 1.0
            service.latency_ms = 50.0
            service.error_rate = 0.0
            service.cpu_pct = 15.0
            service.onset_timestamp = None
            service.is_root_cause = False

    def propagate(self, dt: float = 1.0) -> None:
        for upstream_name, downstream_deps in DEPENDENCY_GRAPH.items():
            if upstream_name not in self.services:
                continue
            upstream = self.services[upstream_name]
            for downstream_name, weight in downstream_deps.items():
                if downstream_name not in self.services:
                    continue
                downstream = self.services[downstream_name]
                health_degradation = max(0, (1.0 - upstream.health)) * weight * dt * 0.1
                downstream.health -= health_degradation
                downstream.health = max(0.0, min(1.0, downstream.health))
                error_bleed = max(0, upstream.error_rate) * weight * dt * 0.05
                downstream.error_rate += error_bleed
                downstream.error_rate = max(0.0, min(1.0, downstream.error_rate))
                if upstream.health < 0.8:
                    downstream.latency_ms = min(3000, downstream.latency_ms * (1 + weight * dt * 0.2))
                else:
                    downstream.latency_ms = max(20, downstream.latency_ms * 0.95)
                downstream.cpu_pct = max(0, min(100, downstream.cpu_pct))

    def get_observation_dict(self) -> Dict[str, Dict]:
        result = {}
        for name, svc in self.services.items():
            result[name] = {
                "health": svc.health,
                "latency_ms": svc.latency_ms,
                "error_rate": svc.error_rate,
                "cpu_pct": svc.cpu_pct
            }
        return result

    def get_symptom_fingerprints(self) -> List[Dict]:
        fingerprints = []
        current_time = time.time()
        for name, svc in self.services.items():
            if svc.health < 0.9 or svc.error_rate > 0.1:
                anomaly_type = "error_rate_spike" if svc.error_rate > 0.3 else "latency_spike"
                onset_offset = 0.0
                if svc.onset_timestamp:
                    onset_offset = current_time - svc.onset_timestamp
                severity = max(svc.error_rate, 1.0 - svc.health)
                fingerprints.append({
                    "service": name,
                    "anomaly": anomaly_type,
                    "onset_offset_seconds": round(onset_offset, 2),
                    "severity": round(severity, 2)
                })
        fingerprints.sort(key=lambda x: x["onset_offset_seconds"])
        return fingerprints

    def apply_fault(self, service_name: str, fault_type: str) -> None:
        if service_name not in self.services:
            return
        svc = self.services[service_name]
        svc.onset_timestamp = time.time()
        svc.is_root_cause = True
        fault_configs = {
            "oom_kill": {"health": 0.05, "error_rate": 0.95, "latency_ms": 2500, "cpu_pct": 98},
            "crash_loop": {"health": 0.15, "error_rate": 0.85, "latency_ms": 1800, "cpu_pct": 75},
            "network_partition": {"health": 0.2, "error_rate": 0.7, "latency_ms": 2800, "cpu_pct": 45},
            "connection_exhaustion": {"health": 0.1, "error_rate": 0.9, "latency_ms": 2200, "cpu_pct": 90}
        }
        config = fault_configs.get(fault_type, fault_configs["crash_loop"])
        svc.health = config["health"]
        svc.error_rate = config["error_rate"]
        svc.latency_ms = config["latency_ms"]
        svc.cpu_pct = config["cpu_pct"]

    def apply_recover(self, service_name: str) -> None:
        if service_name not in self.services:
            return
        svc = self.services[service_name]
        svc.health = 1.0
        svc.error_rate = 0.0
        svc.latency_ms = 50.0
        svc.cpu_pct = 15.0
        svc.is_root_cause = False
