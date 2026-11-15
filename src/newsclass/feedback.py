"""Human-feedback store: records corrections as a retraining-ready dataset.

The point of this module is the missing half of most ML demos: **the loop**.
When a human fixes a wrong label, that judgement is the most valuable training
signal the system can get. We record it with full context so it can be replayed:

* the text and the corrected label,
* what each tier predicted and with what confidence,
* a stable event id so a correction can always be traced back to its request.

Storage is an append-only JSONL file — one record per line, safe to append under
a lock, trivial to load for retraining (``for line in open(...)``), and diffable
in code review. SQLite would also work; JSONL keeps the artifact human-readable
and dependency-free.

No training ever happens silently from this file. It is input for a deliberate,
reviewed retraining run (see scripts/run_pipeline.py).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FeedbackStore:
    """Thread-safe append-only log of human corrections."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- writing ------------------------------------------------------------
    def record(
        self,
        *,
        text: str,
        corrected_label: str,
        predicted_label: str,
        predicted_confidence: float,
        tier: str,
        reason: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "id": uuid.uuid4().hex,
            "recorded_at": _now_iso(),
            "request_id": request_id,
            "text": text,
            "corrected_label": corrected_label,
            "predicted_label": predicted_label,
            "predicted_confidence": round(float(predicted_confidence), 4),
            "tier": tier,
            "reason": reason,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    # -- reading ------------------------------------------------------------
    def entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read corrections, newest first."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self._lock:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # a torn line never breaks the endpoint
        records.reverse()
        return records[:limit] if limit else records

    def summary(self) -> dict[str, Any]:
        records = self.entries()
        if not records:
            return {"total": 0, "by_tier": {}, "agreement_with_human": None}
        fixed_right = sum(1 for r in records if r["corrected_label"] == r["predicted_label"])
        by_tier: dict[str, int] = {}
        for r in records:
            by_tier[r.get("tier", "unknown")] = by_tier.get(r.get("tier", "unknown"), 0) + 1
        return {
            "total": len(records),
            "by_tier": by_tier,
            # A correction that matches the prediction is a "confirm" (the human
            # agreed the model was right); anything else is a "fix".
            "agreement_with_human": round(fixed_right / len(records), 3),
            "latest": records[0]["recorded_at"] if records else None,
        }
