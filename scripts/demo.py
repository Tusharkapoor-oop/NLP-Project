"""Interactive demo client for the classification API.

Usage:
    python scripts/demo.py                                  # uses localhost:8000
    python scripts/demo.py --base-url http://host:8000
    python scripts/demo.py --text "some headline"
"""

from __future__ import annotations

import argparse
import sys

import httpx

EXAMPLES = [
    "The team won the championship after a dramatic final.",
    "Stock markets rallied as inflation finally cooled.",
    "A new smartphone chip promises twice the speed of last year's model.",
    "The president met foreign leaders to discuss trade agreements.",
    "Oil prices surged after the pipeline was shut down.",
    "The film won three awards at the festival.",
]

BAR_WIDTH = 28


def render_bar(probability: float) -> str:
    filled = int(round(probability * BAR_WIDTH))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def show(client: httpx.Client, text: str, tier: str | None) -> None:
    payload: dict = {"text": text}
    if tier:
        payload["tier"] = tier

    response = client.post("/v1/predict", json=payload)
    if response.status_code != 200:
        print(f"  error {response.status_code}: {response.text[:200]}")
        return

    body = response.json()
    print(f"\n  text   : {text[:88]}")
    print(
        f"  result : {body['label']}  ({body['confidence']:.1%})   "
        f"tier={body['tier']}  {body['latency_ms']:.1f} ms"
    )
    print()
    for score in body["scores"]:
        print(
            f"    {score['label']:<10} {render_bar(score['probability'])} {score['probability']:6.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo client for the news classifier")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--text", default=None, help="classify a single custom string")
    parser.add_argument("--tier", default=None, choices=["bronze", "silver", "gold"])
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=60.0) as client:
        try:
            health = client.get("/healthz")
        except httpx.HTTPError as exc:
            print(f"Cannot reach {base}: {exc}", file=sys.stderr)
            print(
                "Start the service first:  uvicorn newsclass.app:app --port 8000", file=sys.stderr
            )
            return 1

        if health.status_code != 200:
            print(f"Service unhealthy: {health.status_code}", file=sys.stderr)
            return 1

        info = client.get("/v1/model").json()
        print(f"Connected to {base}")
        print(f"  default tier : {info['default_tier']}")
        print(f"  available    : {', '.join(info['available_tiers']) or 'none'}")
        print(f"  labels       : {', '.join(info['labels'])}")

        if args.text:
            show(client, args.text, args.tier)
            return 0

        print("\n" + "=" * 78)
        for example in EXAMPLES:
            show(client, example, args.tier)
        print("\n" + "=" * 78)

        # Same text through every tier, to make the trade-off concrete.
        ready = client.get("/v1/tiers").json()
        available = [t["tier"] for t in ready if t["ready"]]
        if len(available) > 1:
            sample = "Apple shares rose after the company announced a new laptop."
            print(f"\nSame headline through every tier: {sample}\n")
            print(f"  {'tier':<8} {'label':<12} {'confidence':>11} {'latency':>10}")
            print("  " + "-" * 44)
            for tier in available:
                response = client.post("/v1/predict", json={"text": sample, "tier": tier})
                if response.status_code != 200:
                    continue
                body = response.json()
                print(
                    f"  {tier:<8} {body['label']:<12} "
                    f"{body['confidence']:>10.1%} {body['latency_ms']:>8.1f} ms"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
