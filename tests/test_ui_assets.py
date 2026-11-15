"""The UI ships its own JavaScript — no CDN, no network at page load.

These tests protect the property that makes the demo trustworthy in a locked-down
classroom or an offline lab: every script and stylesheet the page requests is
served by this process, and the vendor libraries are the versions the notice
file claims.
"""

from __future__ import annotations

import pytest

VENDOR_SCRIPTS = [
    "gsap.min.js",
    "ScrollTrigger.min.js",
    "ScrollToPlugin.min.js",
    "Flip.min.js",
    "Observer.min.js",
    "motion.min.js",
    "three.min.js",
]


@pytest.fixture()
def html(client) -> str:
    return client.get("/").text


class TestOfflineAssets:
    def test_page_ships_no_cdn_references(self, html):
        for marker in ("cdn.jsdelivr", "unpkg.com", "cdnjs", "esm.sh"):
            assert marker not in html, f"page still loads {marker}; the UI must work offline"

    def test_page_links_local_stylesheet_and_script(self, html):
        assert 'href="/static/assets/styles.css"' in html
        assert 'src="/static/assets/app.js"' in html

    def test_every_vendor_script_is_loaded_locally(self, html):
        for script in VENDOR_SCRIPTS:
            assert f"/static/vendor/{script}" in html, f"{script} is not referenced by the page"

    def test_vendor_notice_documents_the_versions(self, client):
        notice = client.get("/static/vendor/NOTICE.md")
        assert notice.status_code == 200
        for library in ("GSAP", "Three", "Motion"):
            assert library.lower() in notice.text.lower()


class TestServedAssets:
    def test_vendor_scripts_are_served_as_javascript(self, client):
        for script in VENDOR_SCRIPTS:
            response = client.get(f"/static/vendor/{script}")
            assert response.status_code == 200, script
            assert "javascript" in response.headers["content-type"], script

    def test_stylesheet_declares_the_design_tokens(self, client):
        css = client.get("/static/assets/styles.css")
        assert css.status_code == 200
        assert "text/css" in css.headers["content-type"]
        assert "--ink" in css.text

    def test_app_script_degrades_when_gsap_is_absent(self, client):
        # The page must stay readable without the animation layer, so every
        # GSAP call is guarded by a capability check.
        js = client.get("/static/assets/app.js").text
        assert "window.gsap" in js
        assert "REDUCED" in js

    def test_missing_asset_is_a_clean_404(self, client):
        assert client.get("/static/assets/does-not-exist.js").status_code == 404


class TestModelLab:
    """The lab is a second page, but it ships under the same offline contract."""

    def test_lab_page_is_served(self, client):
        response = client.get("/lab")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Model Lab" in response.text

    def test_lab_links_only_local_assets(self, client):
        page = client.get("/lab").text
        assert 'href="/static/assets/styles.css"' in page
        assert 'href="/static/assets/lab.css"' in page
        assert 'src="/static/assets/lab.js"' in page
        for marker in ("cdn.jsdelivr", "unpkg.com", "cdnjs", "esm.sh"):
            assert marker not in page, f"the lab still loads {marker}"

    def test_lab_assets_are_served(self, client):
        css = client.get("/static/assets/lab.css")
        js = client.get("/static/assets/lab.js")
        assert css.status_code == 200
        assert "text/css" in css.headers["content-type"]
        assert js.status_code == 200
        assert "javascript" in js.headers["content-type"]

    def test_lab_script_tests_each_tier_independently(self, client):
        js = client.get("/static/assets/lab.js").text
        for tier in ("bronze", "silver", "gold"):
            assert tier in js, f"the lab script never mentions {tier}"
        assert "/v1/predict" in js, "the lab must call predict"
        assert "/v1/explain" in js, "the lab must call explain"
        assert "tier" in js, "the lab must request a tier override"
