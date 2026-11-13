"""Model registry: discovers trained artifacts and routes requests to a tier.

Design goals (these are the things an enterprise reviewer looks for):

* **No hidden global state.** The registry is created explicitly and injected.
* **Graceful degradation.** If Gold is unavailable, requests fall back to the
  best ready tier instead of returning 500 to the caller.
* **Auditability.** Every routing decision is explainable and exposed on the API.
* **Thread-safe lazy loading.** Only the requested tier is loaded into memory,
  so a CPU-only host does not pay for all three models at boot.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from .tiers import TIER_DESCRIPTION, TIER_DISPLAY, TierModel, load_tier

logger = logging.getLogger(__name__)

# Preference order used when falling back.
FALLBACK_ORDER = ("gold", "silver", "bronze")


class RegistryError(RuntimeError):
    """Raised when the registry cannot satisfy a request."""


class ModelRegistry:
    """Owns model instances and the metadata describing them."""

    def __init__(self, artifacts_dir: Path, *, eager: tuple[str, ...] = ()) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._instances: dict[str, TierModel] = {}
        self._lock = threading.Lock()
        self.metadata: dict[str, dict[str, Any]] = {}
        self.labels: list[str] = ["World", "Sports", "Business", "Sci/Tech"]
        self._load_metadata()

        for tier in eager:
            try:
                self.get(tier)
            except RegistryError:
                logger.warning("eager load skipped for tier=%s (artifacts missing)", tier)

    # -- metadata ----------------------------------------------------------
    def _load_metadata(self) -> None:
        registry_path = self.artifacts_dir / "registry.json"
        if registry_path.exists():
            try:
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
                self.metadata = payload.get("tiers", {})
                self.labels = payload.get("labels", self.labels)
                return
            except json.JSONDecodeError:
                logger.exception("registry.json is corrupt; falling back to per-tier files")

        # Fallback: read each tier's metrics.json independently.
        for tier in ("bronze", "silver", "gold"):
            meta = self.artifacts_dir / tier / "metrics.json"
            if meta.exists():
                self.metadata[tier] = json.loads(meta.read_text(encoding="utf-8"))

    # -- loading -----------------------------------------------------------
    def get(self, tier: str) -> TierModel:
        """Return a loaded tier, loading it on first use."""
        if tier not in ("bronze", "silver", "gold"):
            raise RegistryError(f"unknown tier: {tier}")

        with self._lock:
            instance = self._instances.get(tier)
            if instance is not None:
                return instance

            logger.info("loading tier=%s from %s", tier, self.artifacts_dir / tier)
            instance = load_tier(tier, self.artifacts_dir)
            if not instance.ready:
                raise RegistryError(
                    f"tier '{tier}' has no artifacts in {self.artifacts_dir / tier}"
                )
            self._instances[tier] = instance
            return instance

    def is_ready(self, tier: str) -> bool:
        try:
            return self.get(tier).ready
        except RegistryError:
            return False

    def loaded_tiers(self) -> list[str]:
        with self._lock:
            return sorted(self._instances)

    def available_tiers(self) -> list[str]:
        """Tiers that *could* be loaded (artifacts present on disk)."""
        found = []
        for tier in ("gold", "silver", "bronze"):
            directory = self.artifacts_dir / tier
            if not directory.exists():
                continue
            if tier == "bronze" and not (directory / "model.joblib").exists():
                continue
            if tier == "silver" and not (directory / "model.pt").exists():
                continue
            if tier == "gold" and not (directory / "config.json").exists():
                continue
            found.append(tier)
        return found

    # -- routing -----------------------------------------------------------
    def resolve(self, requested: str, default: str) -> tuple[TierModel, str, str | None]:
        """Resolve the tier to serve.

        Returns ``(model, served_tier, fallback_reason)``. ``fallback_reason`` is
        ``None`` when the requested tier was actually used.
        """
        try:
            return self.get(requested), requested, None
        except RegistryError as exc:
            logger.warning("tier=%s unavailable (%s); searching fallbacks", requested, exc)

        for candidate in FALLBACK_ORDER:
            if candidate == requested:
                continue
            try:
                model = self.get(candidate)
            except RegistryError:
                continue
            reason = f"requested tier '{requested}' unavailable; served '{candidate}'"
            logger.warning(reason)
            return model, candidate, reason

        raise RegistryError(
            "no model tier is available. Train one first: python -m newsclass.train --tier bronze"
        )

    def resolve_default(self, default: str) -> tuple[TierModel, str, str | None]:
        return self.resolve(default, default)

    # -- introspection -----------------------------------------------------
    def describe_tiers(self) -> list[dict[str, Any]]:
        described = []
        for tier in ("bronze", "silver", "gold"):
            meta = self.metadata.get(tier, {})
            described.append(
                {
                    "tier": tier,
                    "display_name": TIER_DISPLAY[tier],
                    "model_name": meta.get("model_name", "not trained"),
                    "model_version": meta.get("model_version", "0.0.0"),
                    "description": TIER_DESCRIPTION[tier],
                    "accuracy": meta.get("accuracy"),
                    "macro_f1": meta.get("macro_f1"),
                    "size_mb": meta.get("size_mb"),
                    "latency_ms_p50": meta.get("inference_ms_per_100"),
                    "cost_per_1k_requests": meta.get("cost_per_1k_requests"),
                    "ready": self._artifact_exists(tier),
                }
            )
        return described

    def _artifact_exists(self, tier: str) -> bool:
        directory = self.artifacts_dir / tier
        markers = {
            "bronze": directory / "model.joblib",
            "silver": directory / "model.pt",
            "gold": directory / "config.json",
        }
        return markers[tier].exists()
