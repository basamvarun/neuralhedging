"""
experiment.py — Experiment Tracking

File-based logger that saves per-run artifacts to experiments/<name>_<timestamp>/:
    - config.yaml   — exact config snapshot
    - metrics.json  — train/test metrics per epoch
    - summary.json  — final risk metrics (VaR, CVaR, mean P&L)
    - checkpoint path reference
"""

import json
import yaml
import time
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """
    Tracks and saves experiment results for reproducibility.

    Usage:
        logger = ExperimentLogger("dense_baseline", base_dir="experiments")
        logger.log_config(cfg.to_dict())
        for epoch in range(epochs):
            logger.log_epoch(epoch, train_loss=loss, test_cvar=cvar)
        logger.log_summary(dh_metrics, bs_metrics)
        logger.save()
    """

    def __init__(self, name: str, base_dir: str = "experiments"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{name}_{timestamp}"
        self.run_dir = Path(base_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._config = {}
        self._epoch_log = []
        self._summary = {}

        print(f"[ExperimentLogger] Run directory: {self.run_dir}")

    def log_config(self, config_dict: dict) -> None:
        """Save the config snapshot."""
        self._config = config_dict
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def log_epoch(self, epoch: int, **kwargs) -> None:
        """
        Log scalar metrics for one epoch.
        e.g. logger.log_epoch(1, train_cvar=0.32, test_cvar=0.41)
        """
        entry = {"epoch": epoch, **kwargs}
        self._epoch_log.append(entry)

    def log_summary(self, **named_metrics: dict) -> None:
        """
        Log final risk metrics dicts.
        e.g. logger.log_summary(deep_hedger=dh_metrics, bs_hedge=bs_metrics)
        """
        self._summary = named_metrics

    def save(self) -> None:
        """Write all logs to disk."""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self._epoch_log, f, indent=2)

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(self._summary, f, indent=2)

        print(f"[ExperimentLogger] Results saved to {self.run_dir}")

    @property
    def path(self) -> Path:
        return self.run_dir
