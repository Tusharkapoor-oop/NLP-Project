"""Scheduled drift check: is accuracy on unseen news still where it was?

`scripts/ood_eval.py` answers "how good is the model on data it never saw".
This script answers the recurring question "is it still that good" — it fetches
today's headlines, scores them, compares against a recorded baseline, and
alerts when accuracy falls below an absolute floor or drops too far below that
baseline.

Designed to run unattended:

* exit 0 = healthy, 1 = alert, 2 = not enough data to judge;
* one JSON report per run plus an append-only history file;
* optional webhook (Slack-compatible JSON) via ``NEWSCLASS_ALERT_WEBHOOK``.

Usage::

    python scripts/drift_check.py --tier bronze --min-accuracy 0.65
    python scripts/drift_check.py --tier gold --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Headlines contain characters the Windows console cannot encode; pytest and
# other hosts replace sys.stdout with objects that may not support reconfigure.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ood_eval import fetch, predict, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "artifacts" / "drift_baseline.json"
DEFAULT_REPORT = ROOT / "artifacts" / "drift_report.json"
DEFAULT_HISTORY = ROOT / "artifacts" / "drift_history.jsonl"
DEFAULT_API = os.environ.get("NEWSCLASS_API", "http://127.0.0.1:8010").rstrip("/")


def evaluate(
    rows: list[dict],
    baseline: float | None,
    min_accuracy: float,
    max_drop: float,
) -> dict:
    """Accuracy, interval and verdict for one tier's scored rows."""
    n = len(rows)
    correct = sum(1 for r in rows if r["label"] == r["gold"])
    accuracy = correct / n if n else 0.0
    low, high = wilson(correct, n)
    drop = None if baseline is None else round(baseline - accuracy, 4)

    reasons = []
    if n == 0:
        status = "no_data"
    elif accuracy < min_accuracy:
        reasons.append(f"accuracy {accuracy:.4f} below floor {min_accuracy:.4f}")
    if drop is not None and drop > max_drop:
        reasons.append(f"dropped {drop:.4f} from baseline {baseline:.4f} (max {max_drop:.4f})")

    status = "alert" if reasons and n else ("ok" if n else "no_data")
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "baseline": baseline,
        "drop": drop,
        "status": status,
        "reasons": reasons,
    }


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"warning: {path.name} is not valid JSON; ignoring the baseline")
        return {}
    return payload.get("tiers", {})


def write_baseline(path: Path, results: dict, source: str) -> None:
    tiers = {}
    for tier, res in results.items():
        if not res["n"]:
            continue
        entry = {"accuracy": res["accuracy"], "n": res["n"], "ci95": res["ci95"]}
        # The per-tier floor must survive a refresh. Dropping it silently restores
        # the global default, which alerts on silver every week by design.
        if "min_accuracy" in res:
            entry["min_accuracy"] = res["min_accuracy"]
        tiers[tier] = entry
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "tiers": tiers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"baseline updated: {path}")


def send_alert(webhook: str, message: str, payload: dict) -> bool:
    """Post to a Slack-compatible webhook; a failed alert must not mask the alert."""
    body = json.dumps({"text": message, "drift": payload}).encode("utf-8")
    request = urllib.request.Request(
        webhook, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError) as exc:
        print(f"warning: could not deliver the alert webhook: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheduled accuracy drift check on live news")
    parser.add_argument("--tier", default="bronze", help="tier to check, or 'all'")
    parser.add_argument("--api", default=DEFAULT_API, help="base URL of a running service")
    parser.add_argument("--min-accuracy", type=float, default=0.65, help="absolute floor")
    parser.add_argument("--max-drop", type=float, default=0.10, help="allowed drop vs baseline")
    parser.add_argument("--min-docs", type=int, default=40, help="below this, do not judge")
    parser.add_argument("--per-feed", type=int, default=45, help="headlines requested per section")
    parser.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--offline", action="store_true", help="reuse cached docs instead of fetching"
    )
    parser.add_argument("--webhook-env", default="NEWSCLASS_ALERT_WEBHOOK")
    args = parser.parse_args()

    started = time.perf_counter()
    print(f"drift check against {args.api} (tier={args.tier})")

    if args.offline:
        cache = ROOT / ".pytest_runs" / "ood_docs.json"
        if not cache.exists():
            print(f"NOT JUDGED: --offline needs a cached document set at {cache}")
            print("  run scripts/ood_eval.py once online to create it")
            return 2
        docs = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  reusing cached documents: {len(docs)}")
    else:
        print("  fetching live Guardian section feeds")
        docs = fetch(per_feed=args.per_feed)
        print(f"  {len(docs)} headlines, {len({d['text'] for d in docs})} unique")

    tiers = ["bronze", "silver", "gold"] if args.tier == "all" else [args.tier]
    baseline = load_baseline(args.baseline_file)

    results: dict[str, dict] = {}
    for tier in tiers:
        rows = predict(docs, tier, base_url=args.api)
        known = baseline.get(tier, {})
        # A tier's own floor beats the global default: silver is legitimately weak
        # out of distribution, and alerting on that every week is alarm fatigue.
        floor = float(known.get("min_accuracy", args.min_accuracy))
        results[tier] = evaluate(rows, known.get("accuracy"), floor, args.max_drop)
        results[tier]["min_accuracy"] = floor
        res = results[tier]
        lo, hi = res["ci95"]
        print(
            f"  {tier:<7} acc={res['accuracy']:.4f} 95% CI [{lo:.3f}, {hi:.3f}] "
            f"n={res['n']} floor={floor:.2f} "
            f"drop={res['drop'] if res['drop'] is not None else 'n/a'} "
            f"-> {res['status'].upper()}"
        )

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": "theguardian.com RSS sections",
        "api": args.api,
        "documents": len(docs),
        "min_accuracy_default": args.min_accuracy,
        "max_drop": args.max_drop,
        "min_docs": args.min_docs,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "tiers": results,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "checked_at": report["checked_at"],
                    "tiers": {
                        tier: {k: res[k] for k in ("accuracy", "n", "ci95", "status")}
                        for tier, res in results.items()
                    },
                }
            )
            + "\n"
        )

    if args.update_baseline:
        write_baseline(args.baseline_file, results, report["source"])

    if len(docs) < args.min_docs:
        print(f"\nNOT JUDGED: only {len(docs)} documents (min {args.min_docs})")
        return 2

    alerts = {tier: res for tier, res in results.items() if res["status"] == "alert"}
    if not alerts:
        print("\nOK: unseen-data accuracy is within threshold")
        return 0

    lines = [
        f"{tier}: {res['accuracy']:.3f} (n={res['n']}) — {'; '.join(res['reasons'])}"
        for tier, res in alerts.items()
    ]
    message = "newsclass drift alert\n" + "\n".join(lines)
    print("\nALERT: " + message.replace("\n", "\n  "))

    webhook = os.environ.get(args.webhook_env)
    if webhook:
        delivered = send_alert(webhook, message, {"tiers": alerts})
        print(f"  webhook {'delivered' if delivered else 'failed'}")
    else:
        print(f"  no {args.webhook_env} set; alert is the exit code and report only")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
