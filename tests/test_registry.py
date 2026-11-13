"""Registry behaviour: loading, fallback and failure modes.

These are the tests that matter most operationally: a wrong routing decision
silently degrades every prediction in production.
"""

from __future__ import annotations

import json

import pytest

from newsclass.registry import ModelRegistry, RegistryError


class TestMetadataLoading:
    def test_reads_registry_json(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        assert "bronze" in registry.metadata
        assert registry.labels == ["World", "Sports", "Business", "Sci/Tech"]

    def test_falls_back_to_per_tier_metrics(self, artifacts_dir):
        (artifacts_dir / "registry.json").unlink()
        tier_metrics = artifacts_dir / "bronze" / "metrics.json"
        assert tier_metrics.exists()
        registry = ModelRegistry(artifacts_dir)
        assert "bronze" in registry.metadata

    def test_survives_corrupt_registry(self, artifacts_dir):
        (artifacts_dir / "registry.json").write_text("{not json", encoding="utf-8")
        registry = ModelRegistry(artifacts_dir)  # must not raise
        assert registry.labels  # fell back to defaults

    def test_missing_artifacts_dir_is_tolerated(self, tmp_path):
        registry = ModelRegistry(tmp_path / "does-not-exist")
        assert registry.available_tiers() == []


class TestAvailability:
    def test_detects_available_tier(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        assert "bronze" in registry.available_tiers()
        assert "gold" not in registry.available_tiers()

    def test_is_ready_true_for_trained_tier(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        assert registry.is_ready("bronze") is True

    def test_is_ready_false_for_missing_tier(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        assert registry.is_ready("gold") is False

    def test_unknown_tier_is_rejected(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        with pytest.raises(RegistryError):
            registry.get("platinum")


class TestRouting:
    def test_resolves_available_tier_without_fallback(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        model, tier, reason = registry.resolve("bronze", "bronze")
        assert tier == "bronze"
        assert reason is None
        assert model.ready

    def test_falls_back_when_requested_missing(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        _model, tier, reason = registry.resolve("gold", "gold")
        assert tier == "bronze"
        assert reason is not None
        assert "gold" in reason

    def test_raises_when_nothing_available(self, tmp_path):
        registry = ModelRegistry(tmp_path / "none")
        with pytest.raises(RegistryError):
            registry.resolve("gold", "gold")

    def test_caches_loaded_instance(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        first = registry.get("bronze")
        second = registry.get("bronze")
        assert first is second

    def test_tracks_loaded_tiers(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        assert registry.loaded_tiers() == []
        registry.get("bronze")
        assert registry.loaded_tiers() == ["bronze"]


class TestDescribe:
    def test_describes_all_tiers_even_when_missing(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        described = registry.describe_tiers()
        assert len(described) == 3
        by_tier = {d["tier"]: d for d in described}
        assert by_tier["bronze"]["ready"] is True
        assert by_tier["gold"]["ready"] is False

    def test_every_tier_has_a_human_description(self, artifacts_dir):
        registry = ModelRegistry(artifacts_dir)
        for item in registry.describe_tiers():
            assert item["description"]
            assert item["display_name"]


class TestConcurrency:
    def test_concurrent_loads_return_same_instance(self, artifacts_dir):
        import threading

        registry = ModelRegistry(artifacts_dir)
        results = []
        errors = []

        def _load():
            try:
                results.append(registry.get("bronze"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=_load) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len({id(r) for r in results}) == 1  # one shared instance
