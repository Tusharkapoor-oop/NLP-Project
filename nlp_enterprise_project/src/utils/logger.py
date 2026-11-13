"""Simple cross-project logger and experiment tracker."""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Lightweight logger that writes metrics, params, and artifacts to disk."""

    def __init__(self, log_dir: str, experiment_name: str, seed: Optional[int] = None):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.seed = seed
        self.run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.artifacts: Dict[str, Any] = {}

    def log(self, message: str):
        timestamp = datetime.now().isoformat()
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.run_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_param(self, key: str, value: Any):
        self.params[key] = value
        self.log(f"PARAM  {key} = {value}")

    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        metric_key = f"{key}_step_{step}" if step is not None else key
        self.metrics[metric_key] = float(value)
        self.log(f"METRIC {metric_key} = {value}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log_metric(k, v, step=step)
            else:
                self.log_param(k, v)

    def log_artifact(self, path: str, artifact_type: str = "artifact"):
        p = Path(path)
        if p.exists():
            self.artifacts[p.name] = str(p.resolve())
            self.log(f"ARTIFACT saved: {p.name} ({artifact_type})")

    def save_summary(self):
        summary = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.log(f"Summary written to {self.run_dir / 'summary.json'}")
