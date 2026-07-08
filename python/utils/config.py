"""
config.py — YAML Configuration Loader

Loads experiment configs from YAML files and provides typed access.

Usage:
    cfg = load_config("configs/default.yaml")
    cfg = load_config("configs/default.yaml",
                      overrides={"training.epochs": 200})
    print(cfg.training.epochs)
"""

import yaml
from pathlib import Path


class Config:
    """
    Recursive config object built from a nested dict.
    Allows attribute-style access: cfg.training.lr
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        """Recursively convert back to a plain dict."""
        out = {}
        for key, val in self.__dict__.items():
            out[key] = val.to_dict() if isinstance(val, Config) else val
        return out

    def __repr__(self):
        return f"Config({self.to_dict()})"


def _set_nested(d: dict, key_path: str, value):
    """Set a nested dict value using dot-notation key, e.g. 'training.lr'."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def recursive_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            recursive_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str, overrides: dict = None) -> Config:
    """
    Load a YAML config file, optionally merging key overrides.

    Args:
        config_path: Path to the YAML file.
        overrides:   Dict of dot-notation key -> value overrides.
                     e.g. {"training.epochs": 200, "model.hidden_dim": 128}

    Returns:
        Config object with attribute-style access.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        for key, value in overrides.items():
            _set_nested(data, key, value)

    return Config(data)


def load_config_with_overrides(base_path: str, override_path: str = None) -> Config:
    """
    Load base_path configuration and recursively merge override_path if provided.
    """
    base_p = Path(base_path)
    if not base_p.exists():
        raise FileNotFoundError(f"Base config file not found: {base_path}")

    with open(base_p, "r") as f:
        data = yaml.safe_load(f) or {}

    if override_path:
        override_p = Path(override_path)
        if override_p.exists():
            with open(override_p, "r") as f:
                override_data = yaml.safe_load(f) or {}
            data = recursive_merge(data, override_data)

    return Config(data)
