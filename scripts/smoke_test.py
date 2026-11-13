"""Smoke-test a running instance of the API.

Used in CI after `docker run`, and locally after `uvicorn`.

    python scripts/smoke_test.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

SAMPLE_TEXTS = [
    "The team won the championship after a dramatic final.",
    "Stock markets rallied as inflation cooled.",
    "A new smartphone chip promises twice the speed.",
    "The president met foreign leaders to discuss trade.",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)
    print(f"  ok  {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--tier", default=None)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"Smoke-testing {base}")

    with httpx.Client(base_url=base, timeout=30.0) as client:
        # --- probes ---
        response = client.get("/healthz")
        check(response.status_code == 200, "GET /healthz returns 200")

        response = client.get("/readyz")
        check(response.status_code == 200, "GET /readyz returns 200")
        ready = response.json()
        check(ready.get("ready") is True, "service reports ready")

        # --- catalogue ---
        response = client.get("/v1/tiers")
        check(response.status_code == 200, "GET /v1/tiers returns 200")
        tiers = response.json()
        check(len(tiers) == 3, "three tiers are described")
        check(any(t["ready"] for t in tiers), "at least one tier has artifacts")

        # --- single prediction ---
        payload = {"text": SAMPLE_TEXTS[0]}
        if args.tier:
            payload["tier"] = args.tier
        response = client.post("/v1/predict", json=payload)
        check(response.status_code == 200, "POST /v1/predict returns 200")
        body = response.json()

        for field in ("label", "confidence", "scores", "tier", "model_version", "latency_ms"):
            check(field in body, f"response contains '{field}'")

        check(0.0 <= body["confidence"] <= 1.0, "confidence within [0, 1]")
        check(len(body["scores"]) == 4, "four class scores returned")

        probs = [score["probability"] for score in body["scores"]]
        check(probs == sorted(probs, reverse=True), "scores sorted descending")
        check(abs(sum(probs) - 1.0) < 1e-4, "scores sum to 1")
        check(body["label"] == body["scores"][0]["label"], "top label matches first score")

        print(f"  -> predicted '{body['label']}' with {body['confidence']:.1%} on tier '{body['tier']}'")

        # --- batch ---
        response = client.post("/v1/predict/batch", json={"texts": SAMPLE_TEXTS})
        check(response.status_code == 200, "POST /v1/predict/batch returns 200")
        batch = response.json()
        check(len(batch["predictions"]) == len(SAMPLE_TEXTS), "one prediction per input")

        # --- validation ---
        response = client.post("/v1/predict", json={"text": ""})
        check(response.status_code == 422, "empty text rejected with 422")

        # --- metrics ---
        response = client.get("/metrics")
        check(response.status_code == 200, "GET /metrics returns 200")
        check("newsclass_predictions_total" in response.text, "prediction counter is exposed")

        # --- consistency across tiers ---
        ready_tiers = {t["tier"] for t in tiers if t["ready"]}
        for candidate in ("bronze", "silver", "gold"):
            if candidate not in ready_tiers:
                continue
            response = client.post("/v1/predict", json={"text": SAMPLE_TEXTS[1], "tier": candidate})
            check(response.status_code == 200, f"tier '{candidate}' serves a prediction")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
