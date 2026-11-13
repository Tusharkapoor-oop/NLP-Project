"""Configuration loader and validator."""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Simple wrapper around a YAML config with dot-access."""

    def __init__(self, config_path: str):
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f)

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self._cfg
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str):
        return self._cfg[key]

    def __contains__(self, key: str):
        return key in self._cfg

    def to_dict(self) -> Dict[str, Any]:
        return self._cfg

    @property
    def path(self) -> Path:
        return self._path
