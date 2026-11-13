"""Request/response contracts for the prediction API.

These are the public contract. Changing them is a breaking change, so they are
kept explicit rather than returning ad-hoc dictionaries.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Tier(str, Enum):
    """Model service tiers, cheapest to most capable."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Raw news text. Preprocessing is handled by the service.",
        examples=["Apple unveils a new chip that is twice as fast as the last one."],
    )
    tier: Tier | None = Field(
        default=None,
        description="Optional tier override. Ignored when disabled by the service.",
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="Documents to classify. The service enforces its own "
        "max_batch_size limit and returns 413 when exceeded.",
    )
    tier: Tier | None = None

    @field_validator("texts")
    @classmethod
    def _no_blank_items(cls, values: list[str]) -> list[str]:
        for index, value in enumerate(values):
            if not value or not value.strip():
                raise ValueError(f"texts[{index}] is empty")
        return values


class ClassScore(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Highest-scoring class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: list[ClassScore] = Field(..., description="All classes, best first")
    tier: Tier
    model_version: str
    latency_ms: float = Field(..., description="Server-side inference time")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    tier: Tier
    model_version: str
    total_latency_ms: float


class TierInfo(BaseModel):
    tier: Tier
    display_name: str
    model_name: str
    model_version: str
    description: str
    accuracy: float | None = None
    macro_f1: float | None = None
    size_mb: float | None = None
    latency_ms_p50: float | None = None
    cost_per_1k_requests: float | None = None
    ready: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    ready: bool
    default_tier: Tier
    loaded_tiers: list[Tier]
