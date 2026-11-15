"""End-to-end smoke test for the Wire Desk UI, driven by a real browser.

This is the test that catches what unit tests cannot: a vendor script that 404s,
a selector that no longer matches after an HTML edit, an API call the page fires
but never handles, or a console error that only appears in Chromium.

What it asserts
---------------
* the page loads with zero console errors and zero HTTP responses >= 400;
* the tier catalogue renders one card per ready desk;
* filing a story produces a verdict, four probability bars and a latency;
* the explanation panel renders at least one token with a named method;
* a correction reaches ``/v1/feedback`` and the toast confirms it;
* the desk comparison answers for every ready desk (no "failed" row);
* the signals panel reflects the traffic generated above;
* screenshots are written for the README.

Usage
-----
    # 1. serve the app in another terminal
    python scripts/run_pipeline.py --serve

    # 2. run the browser test
    pip install playwright && playwright install chromium
    python scripts/ui_smoke.py
    python scripts/ui_smoke.py --base-url http://127.0.0.1:8010 --no-shots

Exit codes: 0 pass, 1 UI/API failure, 2 nothing listening at the base URL.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Windows consoles default to a legacy code page, and this script prints arrows
# and box characters. Reconfigure instead of crashing on the report itself.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
DEFAULT_SHOTS = PROJECT_ROOT / "docs" / "assets"


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.console: list[str] = []

    def fail(self, message: str) -> None:
        self.problems.append(message)


def _reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/healthz", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _tiers(base_url: str) -> list[dict]:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/tiers", timeout=10) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return []


def run(base_url: str, shots: Path | None) -> int:
    if not _reachable(base_url):
        print(f"No service at {base_url}/healthz.", file=sys.stderr)
        print("Start it first:  python scripts/run_pipeline.py --serve", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed.", file=sys.stderr)
        print("  pip install playwright && playwright install chromium", file=sys.stderr)
        return 2

    catalogue = _tiers(base_url)
    ready = [tier["tier"] for tier in catalogue if tier.get("ready")]
    report = Report()
    print(f"service   : {base_url}")
    print(f"desks open: {', '.join(ready) or 'none'}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)

        def shot(name: str, settle_ms: int = 650) -> None:
            """Capture a README frame with the pointer parked out of the way.

            The custom cursor is part of the design, but a 64px ring frozen over
            a heading makes a bad figure, so it is moved to the top edge first.
            """
            if shots is None:
                return
            page.mouse.move(1432, 6)
            page.wait_for_timeout(settle_ms)
            page.screenshot(path=str(shots / f"{name}.png"))

        page.on(
            "console",
            lambda msg: (
                report.console.append(f"{msg.type}: {msg.text}") if msg.type == "error" else None
            ),
        )
        page.on("pageerror", lambda exc: report.console.append(f"pageerror: {exc}"))
        page.on(
            "response",
            lambda resp: (
                report.fail(f"HTTP {resp.status} {resp.url}")
                if resp.status >= 400 and not resp.url.endswith("/favicon.ico")
                else None
            ),
        )

        page.goto(base_url, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(2500)

        # --- static shell ---------------------------------------------------
        title = page.title()
        status = page.locator("#status-text").inner_text()
        desks = page.locator(".desk-card").count()
        ticks = page.locator(".ticker-item").count()
        print(f"title     : {title}")
        print(f"status    : {status}")
        print(f"desk cards: {desks}")
        print(f"ticker    : {ticks} items")
        if "Wire Desk" not in title:
            report.fail(f"unexpected page title: {title!r}")
        if desks != len(catalogue):
            report.fail(f"expected {len(catalogue)} desk cards, found {desks}")
        if ticks < 10:
            report.fail(f"expected a filled ticker, found {ticks} items")

        shot("hero")
        page.locator("#studio").scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        shot("studio-empty")

        # --- file a story ---------------------------------------------------
        page.fill(
            "#copy",
            "Apple unveiled a new chip on Tuesday, claiming it is twice as fast as the "
            "previous generation while using less power. Analysts said the move targets "
            "premium laptop buyers and could pressure rivals to cut prices.",
        )
        page.click("#btn-file")
        page.wait_for_selector("#verdict-result:not([hidden])", timeout=60_000)
        page.wait_for_timeout(1200)

        label = page.locator("#verdict-label").inner_text()
        confidence = page.locator("#confidence-value").inner_text()
        tier = page.locator("#meta-tier").inner_text()
        bars = page.locator(".bar").count()
        print(f"verdict   : {label} {confidence} ({tier})")
        if bars != 4:
            report.fail(f"expected 4 probability bars, found {bars}")
        shot("studio-verdict")

        # --- explanation ----------------------------------------------------
        page.click("#btn-explain")
        page.wait_for_selector(".token-pill", timeout=150_000)
        tokens = page.locator(".token-pill").count()
        method = page.locator("#explain-method").inner_text()
        print(f"explain   : {tokens} tokens · {method}")
        if tokens < 1:
            report.fail("no explanation tokens rendered")
        if "unavailable" in method.lower():
            report.fail(f"explanation unavailable: {method}")
        try:
            page.wait_for_function(
                "() => !document.querySelector('#toast')?.classList.contains('is-visible')",
                timeout=8_000,
            )
        except Exception:  # a lingering toast must not fail the run
            pass
        shot("studio-explain")

        # --- feedback -------------------------------------------------------
        page.select_option("#feedback-label", label="Sports")
        page.click("#btn-feedback")
        page.wait_for_function(
            """() => {
                const toast = document.querySelector('#toast');
                return !!toast && toast.classList.contains('is-visible')
                    && /recorded/i.test(toast.textContent);
            }""",
            timeout=20_000,
        )
        toast = page.locator("#toast").inner_text()
        print(f"feedback  : {toast}")
        if "recorded" not in toast.lower():
            report.fail(f"unexpected feedback toast: {toast}")

        # --- desk comparison (gold loads lazily on the first call) ----------
        page.click("#btn-compare")
        page.wait_for_selector(".compare-row", timeout=30_000)
        page.wait_for_function(
            "() => { const rows = [...document.querySelectorAll('.compare-row')];"
            " return rows.length >= 1 && rows.every(r => !/running/.test(r.textContent)); }",
            timeout=180_000,
        )
        rows = page.locator(".compare-row").all_inner_texts()
        print("compare   : " + " | ".join(" ".join(r.split()) for r in rows))
        if len(rows) != len(ready):
            report.fail(f"expected {len(ready)} comparison rows, found {len(rows)}")
        for row in rows:
            if "failed" in row.lower() or "running" in row.lower():
                report.fail(f"a desk did not answer: {' '.join(row.split())}")

        page.locator("#compare-panel").scroll_into_view_if_needed()
        try:
            page.wait_for_function(
                "() => !document.querySelector('#toast')?.classList.contains('is-visible')",
                timeout=8_000,
            )
        except Exception:  # a toast that outlives the wait is not a failure
            pass
        shot("studio-compare")

        # --- signals --------------------------------------------------------
        page.wait_for_timeout(1200)
        events = page.locator("#ins-events").inner_text()
        p50 = page.locator("#ins-p50").inner_text()
        legend = page.locator("#mix-legend li").count()
        updated = page.locator("#ins-updated").inner_text()
        print(f"signals   : {events}, p50={p50}, labels={legend}, at={updated}")
        if events.startswith("0"):
            report.fail("insights did not update after a prediction")
        if "unavailable" in updated.lower():
            report.fail("insights panel could not reach the API")
        if legend < 1:
            report.fail("label-mix legend is empty despite traffic")

        # --- section captures ------------------------------------------------
        for section, name in (
            ("#desks", "desks"),
            ("#method", "method"),
            ("#insights", "insights"),
        ):
            page.locator(section).scroll_into_view_if_needed()
            page.wait_for_timeout(1800)
            shot(name, settle_ms=350)
        shot("ui", settle_ms=350)

        # --- model lab --------------------------------------------------------
        lab = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
        lab.on(
            "console",
            lambda msg: (
                report.console.append(f"lab console: {msg.text}") if msg.type == "error" else None
            ),
        )
        lab.on("pageerror", lambda exc: report.console.append(f"lab pageerror: {exc}"))
        lab.on(
            "response",
            lambda resp: (
                report.fail(f"lab HTTP {resp.status} {resp.url}") if resp.status >= 400 else None
            ),
        )
        lab.goto(f"{base_url}/lab", wait_until="networkidle", timeout=60_000)
        lab.wait_for_selector('.lab-model[data-tier="bronze"]', timeout=30_000)
        cards = lab.locator(".lab-model").count()
        print(f"lab       : {cards} model cards, {len(ready)} ready")
        if cards != len(catalogue):
            report.fail(f"lab renders {cards} cards for {len(catalogue)} tiers")
        if not ready:
            report.fail("no desk is ready, so the lab cannot be exercised")
        else:
            lab.click("#lab-run-all")
            lab.wait_for_function(
                "() => document.querySelectorAll('.lab-model[data-state=\"done\"]').length"
                f" >= {len(ready)}",
                timeout=180_000,
            )
            labels = [
                item.strip()
                for item in lab.locator(
                    '.lab-model[data-state="done"] .lab-label'
                ).all_inner_texts()
            ]
            summary = lab.locator("#sum-agreement").inner_text()
            bars = lab.locator('.lab-model[data-state="done"] .bar').count()
            print(f"lab labels: {' | '.join(labels)}")
            print(f"lab agree : {summary}")
            if len(labels) != len(ready):
                report.fail(f"lab produced {len(labels)} labels for {len(ready)} ready desks")
            if any(label == "No run yet" for label in labels):
                report.fail("a desk finished without a label")
            if not summary or summary == "—":
                report.fail("lab summary did not update after the run")
            if bars < 4 * len(ready):
                report.fail(f"lab rendered {bars} bars for {len(ready)} desks")

            lab.locator('.lab-model[data-state="done"] [data-role="explain"]').first.click()
            lab.wait_for_selector(".lab-model .token-pill", timeout=120_000)
            tokens = lab.locator(".lab-model .token-pill").count()
            print(f"lab tokens: {tokens}")
            if tokens < 1:
                report.fail("the lab explanation rendered no tokens")

        lab.set_viewport_size({"width": 390, "height": 844})
        lab.wait_for_timeout(400)
        overflow = lab.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        print(f"lab mobile: overflow={overflow}px")
        if overflow > 1:
            report.fail(f"the lab overflows a 390px viewport by {overflow}px")
        lab.close()

        # --- mobile ----------------------------------------------------------
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.goto(base_url, wait_until="networkidle", timeout=60_000)
        mobile.wait_for_timeout(2000)
        if shots is not None:
            mobile.screenshot(path=str(shots / "mobile.png"))
        mobile.close()

        browser.close()

    print("\nconsole errors:", json.dumps(report.console, indent=2))
    print("problems      :", json.dumps(report.problems, indent=2))
    if report.console:
        print("FAIL: the page logged console errors")
    if report.problems:
        print("FAIL: UI or API contract violated")
    if report.console or report.problems:
        return 1
    print("PASS: the UI loaded, classified, explained, corrected and reported cleanly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Where the service is listening (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SHOTS,
        help="Directory for screenshots (default: docs/assets)",
    )
    parser.add_argument("--no-shots", action="store_true", help="Skip screenshot capture")
    args = parser.parse_args(argv)

    shots = None if args.no_shots else args.out
    if shots is not None:
        shots.mkdir(parents=True, exist_ok=True)
    return run(args.base_url.rstrip("/"), shots)


if __name__ == "__main__":
    sys.exit(main())
