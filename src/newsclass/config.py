"""Typed application configuration.

Every tunable lives here so behaviour is controlled by environment variables
(12-factor), not by editing code. Defaults are safe for local development.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, read from environment variables (prefix ``NEWSCLASS_``)."""

    model_config = SettingsConfigDict(
        env_prefix="NEWSCLASS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service identity -------------------------------------------------
    app_name: str = "newsclass"
    environment: str = Field(default="dev", description="dev | staging | prod")
    log_level: str = "INFO"

    # --- Model routing ----------------------------------------------------
    # Which tier serves predictions by default. Can be overridden per request
    # only when the caller is allowed to choose (see allow_tier_override).
    default_tier: str = "gold"
    allow_tier_override: bool = True
    # Fail fast at startup instead of accepting traffic we cannot serve.
    require_default_tier: bool = True

    # Which tiers to load into memory at boot, on top of the default one.
    #
    # Only the default tier is warmed by default, because the gold model is
    # ~255 MB and loading it costs ~39 s on this CPU — paying that at boot for
    # traffic that never uses it would be wasteful. The cost is that the *first*
    # request for an unwarmed tier waits for that load, which is a real
    # production concern: a client with a 30 s timeout would fail.
    #
    # Set this (e.g. NEWSCLASS_PRELOAD_TIERS="bronze,silver,gold") when the
    # instance must answer any tier immediately, or point a load balancer's
    # readiness check at a warmed instance.
    preload_tiers: str = ""

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    registry_filename: str = "registry.json"

    # --- Inference limits -------------------------------------------------
    max_batch_size: int = 64
    # Cheap guard so a single huge document cannot exhaust memory.
    max_text_chars: int = 20_000
    request_timeout_seconds: float = 10.0

    # --- HTTP -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Observability ----------------------------------------------------
    metrics_enabled: bool = True
    # Expose /docs and /redoc. Turn off in locked-down environments.
    docs_enabled: bool = True

    @field_validator("default_tier")
    @classmethod
    def _valid_tier(cls, value: str) -> str:
        allowed = {"bronze", "silver", "gold"}
        lowered = value.lower()
        if lowered not in allowed:
            raise ValueError(f"default_tier must be one of {sorted(allowed)}")
        return lowered

    @field_validator("environment")
    @classmethod
    def _valid_environment(cls, value: str) -> str:
        allowed = {"dev", "staging", "prod"}
        lowered = value.lower()
        if lowered not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}")
        return lowered

    @field_validator("preload_tiers")
    @classmethod
    def _valid_preload(cls, value: str) -> str:
        """Reject unknown tier names at startup.

        A typo here would silently never warm a tier — the exact failure this
        setting exists to prevent — so it is a startup error, not a warning.
        """
        allowed = {"bronze", "silver", "gold"}
        names = [part.strip().lower() for part in value.split(",") if part.strip()]
        unknown = sorted(set(names) - allowed)
        if unknown:
            raise ValueError(f"preload_tiers contains unknown tiers: {unknown}")
        return value

    @property
    def preload_tier_list(self) -> tuple[str, ...]:
        """``preload_tiers`` parsed into an ordered, de-duplicated tuple."""
        names = [part.strip().lower() for part in self.preload_tiers.split(",") if part.strip()]
        return tuple(dict.fromkeys(names))

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"

    @property
    def registry_path(self) -> Path:
        return self.artifacts_dir / self.registry_filename


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (one instance per process)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cache. Used by tests that monkeypatch the environment."""
    get_settings.cache_clear()


# Convenience for scripts running outside the app (training, CI).
def env_artifacts_dir() -> Path:
    raw = os.getenv("NEWSCLASS_ARTIFACTS_DIR")
    return Path(raw) if raw else PROJECT_ROOT / "artifacts"
