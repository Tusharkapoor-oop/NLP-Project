"""FastAPI application exposing the tiered classification service.

Endpoints
---------
GET  /healthz              liveness  (process is up; no model dependency)
GET  /readyz               readiness (default tier can serve)
GET  /v1/tiers             metadata for every tier, incl. accuracy/cost
GET  /v1/model             info about the tier that will serve by default
POST /v1/predict           single text
POST /v1/predict/batch     up to max_batch_size texts
GET  /metrics              Prometheus exposition
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings, get_settings
from .logging_setup import configure_logging
from .metrics import (
    CONTENT_TYPE,
    metrics_registry,
    observe_confidence,
    observe_fallback,
    observe_prediction,
)
from .registry import ModelRegistry, RegistryError
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClassScore,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
    Tier,
    TierInfo,
)
from .text import warm_up

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_prediction_counter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration and models once, not per request."""
    settings: Settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "starting app=%s version=%s env=%s default_tier=%s",
        settings.app_name,
        __version__,
        settings.environment,
        settings.default_tier,
    )

    registry = ModelRegistry(settings.artifacts_dir, eager=(settings.default_tier,))
    app.state.registry = registry
    app.state.settings = settings

    try:
        warm_up()
    except Exception:  # pragma: no cover - resource availability only
        logger.warning("preprocessing warm-up failed; first request may be slower")

    if settings.require_default_tier and not registry.is_ready(settings.default_tier):
        available = registry.available_tiers()
        logger.warning(
            "default tier '%s' is not available; available tiers: %s. "
            "Requests will fall back automatically.",
            settings.default_tier,
            available or "none",
        )

    yield
    logger.info("shutting down")


# ---------------------------------------------------------------------------
# Dependencies (module level on purpose)
# ---------------------------------------------------------------------------
# These MUST live at module level, not inside ``create_app``. This module uses
# ``from __future__ import annotations``, which turns every annotation into a
# string; FastAPI resolves those strings in *module* globals. A closure-local
# dependency would therefore be unresolvable and FastAPI would silently treat
# the parameter as a query parameter, returning 422 on every request.


def get_registry(request: Request) -> ModelRegistry:
    """Return the registry bound to this app instance."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="registry not initialised",
        )
    return registry


def current_settings(request: Request) -> Settings:
    """Return the settings bound to this app instance."""
    bound = getattr(request.app.state, "settings", None)
    return bound if bound is not None else get_settings()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (lets tests build isolated instances)."""
    settings = settings or get_settings()

    app = FastAPI(
        title="News Topic Classification API",
        description=(
            "Tiered NLP classification service.\n\n"
            "**Tiers** — bronze (TF-IDF), silver (Word2Vec+BiLSTM), gold (DistilBERT).\n\n"
            "Every response reports the tier that actually served it, plus latency."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )


    # -- observability -----------------------------------------------------
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.2f}"
        # Skip access logging for probes to keep logs readable.
        if request.url.path not in ("/healthz", "/readyz", "/metrics"):
            logger.info(
                "request id=%s method=%s path=%s status=%s duration_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    @app.exception_handler(RegistryError)
    async def registry_error_handler(_request: Request, exc: RegistryError):
        logger.error("registry error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc), "type": "model_unavailable"},
        )

    # -- probes ------------------------------------------------------------
    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    def healthz(active: Annotated[Settings, Depends(current_settings)]) -> HealthResponse:
        """Liveness: returns 200 as long as the process is running."""
        return HealthResponse(status="ok", version=__version__, environment=active.environment)

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    def readyz(
        active: Annotated[Settings, Depends(current_settings)],
        registry: Annotated[ModelRegistry, Depends(get_registry)],
    ) -> ReadyResponse:
        """Readiness: can this instance serve predictions right now?"""
        ready = registry.is_ready(active.default_tier) or bool(registry.available_tiers())
        if not ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no trained model artifacts found",
            )
        return ReadyResponse(
            ready=True,
            default_tier=Tier(active.default_tier),
            loaded_tiers=[Tier(t) for t in registry.loaded_tiers()],
        )

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        """Prometheus scrape endpoint."""
        if not settings.metrics_enabled:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="metrics disabled")
        return Response(content=metrics_registry(), media_type=CONTENT_TYPE)

    # -- catalogue ---------------------------------------------------------
    @app.get("/v1/tiers", response_model=list[TierInfo], tags=["models"])
    def list_tiers(registry: Annotated[ModelRegistry, Depends(get_registry)]) -> list[TierInfo]:
        """Describe every tier, including measured accuracy and cost."""
        return [TierInfo(**item) for item in registry.describe_tiers()]

    @app.get("/v1/model", tags=["models"])
    def model_info(
        active: Annotated[Settings, Depends(current_settings)],
        registry: Annotated[ModelRegistry, Depends(get_registry)],
    ) -> dict:
        return {
            "default_tier": active.default_tier,
            "available_tiers": registry.available_tiers(),
            "loaded_tiers": registry.loaded_tiers(),
            "labels": registry.labels,
            "allow_tier_override": active.allow_tier_override,
        }

    # -- inference ---------------------------------------------------------
    def _serve(texts: list[str], requested: Tier | None, active: Settings, registry: ModelRegistry):
        if requested is not None and not active.allow_tier_override:
            requested_tier = active.default_tier
        elif requested is not None:
            requested_tier = requested.value
        else:
            requested_tier = active.default_tier

        model, served_tier, fallback_reason = registry.resolve(
            requested_tier, active.default_tier
        )

        started = time.perf_counter()
        probabilities = model.predict_proba(texts)
        latency_ms = (time.perf_counter() - started) * 1000

        if fallback_reason:
            observe_fallback(requested_tier=requested_tier, served_tier=served_tier)

        return probabilities, served_tier, latency_ms, fallback_reason

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(
        payload: PredictionRequest,
        active: Annotated[Settings, Depends(current_settings)],
        registry: Annotated[ModelRegistry, Depends(get_registry)],
    ) -> PredictionResponse:
        """Classify one document."""
        if len(payload.text) > active.max_text_chars:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"text exceeds max_text_chars={active.max_text_chars}",
            )

        probabilities, tier, latency_ms, fallback = _serve(
            [payload.text], payload.tier, active, registry
        )
        row = probabilities[0]
        order = np.argsort(row)[::-1]
        labels = registry.labels

        observe_prediction(tier=tier, labels=labels, index=int(order[0]), duration_s=latency_ms / 1000)
        observe_confidence(tier=tier, confidence=float(row[order[0]]))

        response = PredictionResponse(
            label=labels[int(order[0])],
            confidence=float(row[order[0]]),
            scores=[
                ClassScore(label=labels[int(i)], probability=float(row[i])) for i in order
            ],
            tier=Tier(tier),
            model_version=registry.metadata.get(tier, {}).get("model_version", __version__),
            latency_ms=round(latency_ms, 2),
        )
        if fallback:
            logger.warning("served with fallback: %s", fallback)
        return response

    @app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    def predict_batch(
        payload: BatchPredictionRequest,
        active: Annotated[Settings, Depends(current_settings)],
        registry: Annotated[ModelRegistry, Depends(get_registry)],
    ) -> BatchPredictionResponse:
        """Classify several documents in one call (more efficient)."""
        if len(payload.texts) > active.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"batch exceeds max_batch_size={active.max_batch_size}",
            )
        for index, text in enumerate(payload.texts):
            if len(text) > active.max_text_chars:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"texts[{index}] exceeds max_text_chars={active.max_text_chars}",
                )

        probabilities, tier, latency_ms, _fallback = _serve(
            payload.texts, payload.tier, active, registry
        )
        labels = registry.labels
        version = registry.metadata.get(tier, {}).get("model_version", __version__)

        predictions = []
        for row in probabilities:
            order = np.argsort(row)[::-1]
            predictions.append(
                PredictionResponse(
                    label=labels[int(order[0])],
                    confidence=float(row[order[0]]),
                    scores=[
                        ClassScore(label=labels[int(i)], probability=float(row[i])) for i in order
                    ],
                    tier=Tier(tier),
                    model_version=version,
                    latency_ms=round(latency_ms / len(payload.texts), 2),
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            tier=Tier(tier),
            model_version=version,
            total_latency_ms=round(latency_ms, 2),
        )

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        """Serve the single-page demo UI."""
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            {
                "service": "newsclass",
                "version": __version__,
                "docs": "/docs",
                "health": "/healthz",
                "tiers": "/v1/tiers",
            }
        )

    return app


app = create_app()
