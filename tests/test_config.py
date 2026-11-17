"""Configuration tests.

Config bugs cause production incidents (wrong tier, wrong port, docs exposed on
the public internet), so the validation rules are tested explicitly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from newsclass.config import Settings


class TestDefaults:
    def test_sensible_defaults(self):
        settings = Settings()
        assert settings.default_tier == "gold"
        assert settings.environment == "dev"
        assert settings.max_batch_size == 64

    def test_registry_path_is_derived(self):
        settings = Settings()
        assert settings.registry_path.name == "registry.json"

    def test_is_production_flag(self):
        assert Settings(environment="prod").is_production is True
        assert Settings(environment="dev").is_production is False


class TestValidation:
    def test_rejects_unknown_tier(self):
        with pytest.raises(ValidationError):
            Settings(default_tier="platinum")

    def test_rejects_unknown_environment(self):
        with pytest.raises(ValidationError):
            Settings(environment="testing")

    def test_normalises_tier_case(self):
        assert Settings(default_tier="GOLD").default_tier == "gold"

    def test_normalises_environment_case(self):
        assert Settings(environment="PROD").environment == "prod"


class TestPreloadTiers:
    """``preload_tiers`` moves a tier's one-off load cost from the first caller
    to boot. A typo would silently never warm a tier, so it must be an error."""

    def test_default_is_empty(self):
        assert Settings().preload_tier_list == ()

    def test_parses_a_single_tier(self):
        assert Settings(preload_tiers="gold").preload_tier_list == ("gold",)

    def test_parses_all_tiers_in_order(self):
        settings = Settings(preload_tiers="bronze,silver,gold")
        assert settings.preload_tier_list == ("bronze", "silver", "gold")

    def test_tolerates_whitespace_and_case(self):
        assert Settings(preload_tiers=" Gold , SILVER ").preload_tier_list == ("gold", "silver")

    def test_deduplicates(self):
        assert Settings(preload_tiers="gold,gold,bronze").preload_tier_list == ("gold", "bronze")

    def test_ignores_empty_segments(self):
        assert Settings(preload_tiers="gold,,silver,").preload_tier_list == ("gold", "silver")

    def test_rejects_unknown_tier(self):
        with pytest.raises(ValidationError):
            Settings(preload_tiers="bronze,golds")

    def test_rejects_unknown_tier_from_environment(self, monkeypatch):
        monkeypatch.setenv("NEWSCLASS_PRELOAD_TIERS", "platinum")
        with pytest.raises(ValidationError):
            Settings()


class TestEnvironmentOverrides:
    def test_reads_from_environment(self, monkeypatch):
        monkeypatch.setenv("NEWSCLASS_DEFAULT_TIER", "bronze")
        monkeypatch.setenv("NEWSCLASS_MAX_BATCH_SIZE", "8")
        settings = Settings()
        assert settings.default_tier == "bronze"
        assert settings.max_batch_size == 8

    def test_settings_are_cached(self):
        from newsclass.config import get_settings, reset_settings_cache

        reset_settings_cache()
        assert get_settings() is get_settings()
        reset_settings_cache()
