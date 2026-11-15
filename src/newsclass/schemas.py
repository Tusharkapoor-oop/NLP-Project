"""Request/response contracts for the prediction API.

These are the public contract. Changing them is a breaking change, so they are
kept explicit rather than returning ad-hoc dictionaries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
class ExplainRequest(PredictionRequest):
    """Same contract as predict; the response gains the explanation."""


class TokenContribution(BaseModel):
    token: str
    contribution: float | None = None
    importance: float | None = None


class ExplanationResponse(PredictionResponse):
    explanation_method: str
    explanation_context: str
    tokens: list[TokenContribution]


# ---------------------------------------------------------------------------
# Feedback (human-in-the-loop)
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    corrected_label: str = Field(..., description="What the human says the true topic is")
    predicted_label: str = Field(..., description="What the model said")
    predicted_confidence: float = Field(..., ge=0.0, le=1.0)
    tier: Tier
    reason: str | None = Field(default=None, max_length=500)
    request_id: str | None = None

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class FeedbackEntry(BaseModel):
    id: str
    recorded_at: str
    text: str
    corrected_label: str
    predicted_label: str
    predicted_confidence: float
    tier: Tier
    reason: str | None = None
    request_id: str | None = None


class FeedbackListResponse(BaseModel):
    entries: list[FeedbackEntry]
    summary: dict


# ---------------------------------------------------------------------------
# Insights (live aggregate view)
# ---------------------------------------------------------------------------
class InsightsResponse(BaseModel):
    """A typed view of the bounded in-process traffic window.

    The payload is allowed to grow (``extra="allow"``) so the UI can be updated
    ahead of the API without a breaking contract change, but the common fields
    are declared explicitly so the OpenAPI schema is useful rather than a bare
    object.
    """

    model_config = ConfigDict(extra="allow")

    window: dict[str, int] = Field(default_factory=dict)
    empty: bool = False
    updated_at: str | None = None
    label_mix: list[dict[str, Any]] = Field(default_factory=list)
    tier_usage: list[dict[str, Any]] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    fallbacks: int = 0
    series_labels: dict[str, list[int]] = Field(default_factory=dict)
