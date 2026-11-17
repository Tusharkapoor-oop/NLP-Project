"""Out-of-distribution accuracy test: current news, never seen in training.

AG News is 2004-2005. These 2026 Guardian headlines come from publisher
sections (the ground truth is the section the editors filed it under), so
there is no temporal or lexical overlap with the training corpus.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = os.environ.get("NEWSCLASS_API", "http://127.0.0.1:8010").rstrip("/")
FEEDS = {
    "World": "https://www.theguardian.com/world/rss",
    "Sports": "https://www.theguardian.com/sport/rss",
    "Business": "https://www.theguardian.com/business/rss",
    "Sci/Tech": "https://www.theguardian.com/technology/rss",
}
PER_FEED = 45
BATCH = 64
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
TAG = re.compile(r"<[^>]+>")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", text))).strip()


def fetch(per_feed: int = PER_FEED) -> list[dict]:
    docs, seen = [], set()
    for label, url in FEEDS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "newsclass-ood-eval/1.0"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        root = ET.fromstring(raw)
        n = 0
        for item in root.iter("item"):
            title = clean(item.findtext("title") or "")
            desc = clean(item.findtext("description") or "")
            text = f"{title}. {desc}".strip()
            key = re.sub(r"[^a-z0-9 ]", "", title.lower())[:120]
            if len(text) < 30 or key in seen:
                continue
            seen.add(key)
            docs.append({"text": text[:1000], "gold": label})
            n += 1
            if n >= per_feed:
                break
        print(f"  {label:<9} {n:>3} headlines")
    return docs


def post(path: str, payload: dict, timeout: int = 600, base_url: str | None = None) -> dict:
    req = urllib.request.Request(
        (base_url or API) + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def predict(docs: list[dict], tier: str, base_url: str | None = None) -> list[dict]:
    out: list[dict] = []
    for start in range(0, len(docs), BATCH):
        chunk = docs[start : start + BATCH]
        res = post(
            "/v1/predict/batch",
            {"texts": [d["text"] for d in chunk], "tier": tier},
            base_url=base_url,
        )
        served = res.get("tier")
        if served != tier:
            raise SystemExit(
                f"refusing to score: requested tier={tier!r} but the server served {served!r}. "
                "Start a server with that tier installed, or the numbers would be a fallback."
            )
        for doc, pred in zip(chunk, res["predictions"], strict=True):
            out.append({**doc, **pred})
        print(f"    {tier}: {min(start + BATCH, len(docs))}/{len(docs)}")
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def prf(rows: list[dict], label: str) -> tuple[float, float, float, int]:
    tp = sum(1 for r in rows if r["gold"] == label and r["label"] == label)
    fp = sum(1 for r in rows if r["gold"] != label and r["label"] == label)
    fn = sum(1 for r in rows if r["gold"] == label and r["label"] != label)
    support = sum(1 for r in rows if r["gold"] == label)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return p, rec, f1, support


def report(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["label"] == r["gold"] for r in rows)
    acc = correct / n if n else 0.0
    lo, hi = wilson(correct, n)
    f1s = [prf(rows, label)[2] for label in LABELS]
    conf_ok = [r["confidence"] for r in rows if r["label"] == r["gold"]]
    conf_bad = [r["confidence"] for r in rows if r["label"] != r["gold"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    hi_conf = [r for r in rows if r["confidence"] >= 0.80]
    lo_conf = [r for r in rows if r["confidence"] < 0.60]

    print(f"\n=== {name} ===")
    print(f"  n={n}  accuracy={acc:.4f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  macro-F1={sum(f1s) / len(f1s):.4f}")
    print(f"  mean confidence  correct={mean(conf_ok):.3f}  wrong={mean(conf_bad):.3f}")
    print(
        f"  accuracy @ conf>=0.80: {sum(r['label'] == r['gold'] for r in hi_conf)}/{len(hi_conf)}"
        f"   @ conf<0.60: {sum(r['label'] == r['gold'] for r in lo_conf)}/{len(lo_conf)}"
    )
    print("  per class:")
    for label in LABELS:
        p, rec, f1, sup = prf(rows, label)
        print(f"    {label:<9} P={p:.3f} R={rec:.3f} F1={f1:.3f} n={sup}")
    print("  confusion (rows=truth, cols=predicted):")
    print(f"    {'':<10}" + "".join(f"{c[:9]:>11}" for c in LABELS))
    for g in LABELS:
        counts = Counter(r["label"] for r in rows if r["gold"] == g)
        print(f"    {g:<10}" + "".join(f"{counts[c]:>11}" for c in LABELS))

    errors = [r for r in rows if r["label"] != r["gold"]]
    print(f"  errors (showing {min(12, len(errors))} of {len(errors)}):")
    for r in errors[:12]:
        print(f"    [{r['gold']:<8}->{r['label']:<8} {r['confidence']:.2f}] {r['text'][:100]}")

    return {
        "n": n,
        "accuracy": acc,
        "ci95": [lo, hi],
        "macro_f1": sum(f1s) / len(f1s),
        "mean_conf_correct": mean(conf_ok),
        "mean_conf_wrong": mean(conf_bad),
        "per_class": {label: prf(rows, label) for label in LABELS},
        "errors": [
            {"gold": r["gold"], "pred": r["label"], "conf": r["confidence"], "text": r["text"]}
            for r in rows
            if r["label"] != r["gold"]
        ],
    }


def main() -> None:
    cache = Path(".pytest_runs/ood_docs.json")
    results = Path(".pytest_runs/fresh_ood_results.json")
    # A fresh clone has no scratch directory; without this the first run dies
    # with FileNotFoundError after the slow fetch has already happened.
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with cache.open(encoding="utf-8") as fh:
            docs = json.load(fh)
        print(f"Reusing cached documents: {len(docs)}")
    else:
        print("Fetching live Guardian section feeds...")
        docs = fetch()
        with cache.open("w", encoding="utf-8") as fh:
            json.dump(docs, fh, indent=1)
    print(f"Total {len(docs)} documents, {len(set(d['text'] for d in docs))} unique")
    out: dict = {"source": "theguardian.com RSS sections", "docs": len(docs), "tiers": {}}
    for tier in ("bronze", "silver", "gold"):
        started = time.perf_counter()
        rows = predict(docs, tier)
        elapsed = time.perf_counter() - started
        res = report(tier, rows)
        res["wall_seconds"] = round(elapsed, 1)
        res["rows"] = rows
        out["tiers"][tier] = res

    with results.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\n=== summary ===")
    for tier, res in out["tiers"].items():
        print(
            f"  {tier:<7} acc={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}"
            f"  {res['wall_seconds']}s"
        )


if __name__ == "__main__":
    main()
