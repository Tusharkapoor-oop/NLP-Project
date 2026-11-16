/* ==========================================================================
   Wire Desk — interface behaviour
   --------------------------------------------------------------------------
   The page is a static shell: this module owns every live behaviour. It is
   deliberately dependency-light and fails soft — if the API is down, the copy
   desk still renders and explains that it cannot reach the service; if WebGL
   is unavailable, the CSS fallback takes over.

   Scripts loaded before this module (UMD globals): THREE, gsap, ScrollTrigger,
   ScrollToPlugin, Motion. Every one of them is optional: the module checks for
   its globals before use and degrades to plain CSS/DOM behaviour, so the page
   remains fully usable with scripts blocked or on a low-end device.
   ========================================================================== */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FINE_POINTER = window.matchMedia("(pointer: fine)").matches;

  const LABELS = ["World", "Sports", "Business", "Sci/Tech"];
  const TOPIC_COLORS = {
    World: "#6f9bb5",
    Sports: "#7fa87c",
    Business: "#e7c37e",
    "Sci/Tech": "#9a8fc4",
  };
  const TIER_ORDER = ["bronze", "silver", "gold"];

  const state = {
    tiers: [],
    labels: LABELS,
    last: null,
    filedText: "",
    bulletin: [],
    busy: false,
    defaultTier: null,
    allowOverride: true,
  };

  const BULLETIN_KEY = "wire-desk.bulletin.v1";

  function loadStoredBulletin() {
    try {
      const raw = localStorage.getItem(BULLETIN_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.filter((entry) => entry && entry.text).slice(0, 7) : [];
    } catch {
      return [];
    }
  }

  function persistBulletin() {
    try {
      localStorage.setItem(BULLETIN_KEY, JSON.stringify(state.bulletin.slice(0, 7)));
    } catch {
      /* private mode: the bulletin stays in memory for this tab */
    }
  }

  /* ------------------------------------------------------------ utilities */
  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const pct = (value, digits = 1) => (value * 100).toFixed(digits) + "%";

  const fmtSize = (mb) => {
    if (mb == null) return "—";
    return mb < 1 ? Math.round(mb * 1000) + " KB" : mb.toFixed(1) + " MB";
  };

  const clock = (date = new Date()) =>
    date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const tierClass = (tier) => (TIER_ORDER.includes(tier) ? tier : "bronze");

  function toast(message, kind = "ok") {
    const el = $("toast");
    if (!el) return;
    el.textContent = String(message).slice(0, 300);
    el.className = "toast is-visible " + (kind === "error" ? "is-error" : "is-ok");
    el.hidden = false;
    if (window.Motion && !REDUCED) {
      try {
        window.Motion.animate(el, { opacity: [0, 1], y: [8, 0] }, { duration: 0.28 });
      } catch {
        /* CSS already shows it */
      }
    }
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      el.classList.remove("is-visible");
      setTimeout(() => { el.hidden = true; }, 320);
    }, 3600);
  }

  /* --------------------------------------------------------------- service */
  const API = {
    async request(path, options = {}) {
      const { timeout = 30000, ...init } = options;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeout);
      try {
        const res = await fetch(path, { cache: "no-store", signal: controller.signal, ...init });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          // FastAPI validation errors can be long arrays. Show the field and
          // the reason, never the echoed document (which may be 20k chars).
          const detail = data && data.detail;
          let message;
          if (Array.isArray(detail)) {
            message = detail
              .map((item) => {
                const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : "value";
                return `${field}: ${item.msg || "invalid"}`;
              })
              .join("; ");
          } else if (typeof detail === "string") {
            message = detail;
          } else if (detail && typeof detail === "object") {
            message = JSON.stringify(detail);
          } else {
            message = "HTTP " + res.status;
          }
          throw new Error(message.slice(0, 240));
        }
        data.__requestId = res.headers.get("x-request-id") || data.__requestId || null;
        return data;
      } finally {
        clearTimeout(timer);
      }
    },
    get(path, timeout) {
      return this.request(path, { timeout });
    },
    post(path, body, timeout) {
      return this.request(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        timeout,
      });
    },
  };

  /* ============================================================ preloader
     Always dismiss. A stuck veil is worse than a missing animation. */
  function runPreloader() {
    const veil = $("preloader");
    if (!veil) return;
    const bar = $("preloader-bar");
    const count = $("preloader-count");
    const finish = () => {
      veil.classList.add("is-done");
      document.documentElement.classList.add("is-ready");
      setTimeout(() => veil.remove(), 900);
    };
    setTimeout(finish, 3000); // hard safety net

    if (REDUCED || !window.gsap) {
      if (bar) bar.style.width = "100%";
      if (count) count.textContent = "100";
      finish();
      return;
    }
    const progress = { v: 0 };
    gsap.to(progress, {
      v: 100,
      duration: 0.9,
      ease: "power2.inOut",
      onUpdate: () => {
        if (bar) bar.style.width = progress.v + "%";
        if (count) count.textContent = String(Math.round(progress.v)).padStart(2, "0");
      },
      onComplete: finish,
    });
  }

  /* ============================================================ 3D scene
     A slow field of filings drifting through the background. Decorative, so
     it is aria-hidden, paused when off-screen or hidden, and skipped entirely
     when the visitor asks for reduced motion or WebGL is missing. */
  function initScene() {
    const canvas = $("scene-canvas");
    if (!canvas || !window.THREE) {
      document.documentElement.classList.add("no-webgl");
      return;
    }
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    } catch {
      document.documentElement.classList.add("no-webgl");
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 100);
    camera.position.set(0, 0, 11);

    const group = new THREE.Group();
    scene.add(group);

    const makeCloud = (count, radius, spread, color, size, opacity) => {
      const positions = new Float32Array(count * 3);
      for (let i = 0; i < count; i += 1) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = radius + (Math.random() - 0.5) * spread;
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.62;
        positions[i * 3 + 2] = r * Math.cos(phi);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const material = new THREE.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
      });
      const points = new THREE.Points(geometry, material);
      group.add(points);
      return points;
    };

    const gold = makeCloud(700, 7.4, 7, 0xe7c37e, 0.035, 0.5);
    const green = makeCloud(420, 10, 9, 0x8fe8bd, 0.03, 0.26);

    // A faint wireframe anchor keeps the field from reading as random noise.
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(4.6, 1)),
      new THREE.LineBasicMaterial({ color: 0xf5f0e6, transparent: true, opacity: 0.045 }),
    );
    group.add(wire);

    let width = 0;
    let height = 0;
    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      renderer.setPixelRatio(dpr);
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener("resize", resize, { passive: true });

    const target = { x: 0, y: 0 };
    if (FINE_POINTER && window.gsap) {
      window.addEventListener("pointermove", (event) => {
        target.x = (event.clientX / window.innerWidth - 0.5) * 0.9;
        target.y = (event.clientY / window.innerHeight - 0.5) * 0.6;
        gsap.to(group.rotation, { x: target.y, y: target.x, duration: 1.4, ease: "power2.out" });
      }, { passive: true });
    }

    let visible = !document.hidden;
    document.addEventListener("visibilitychange", () => { visible = !document.hidden; });

    let last = performance.now();
    const render = () => {
      const now = performance.now();
      const dt = Math.min(64, now - last) / 16.67;
      last = now;
      if (visible) {
        if (!REDUCED) {
          group.rotation.z += 0.0006 * dt;
          gold.rotation.y += 0.0011 * dt;
          green.rotation.y -= 0.0008 * dt;
          wire.rotation.y += 0.0005 * dt;
        }
        renderer.render(scene, camera);
      }
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  /* ============================================================ motion shell */
  function initCursor() {
    if (!FINE_POINTER || REDUCED || !window.gsap) return;
    const dot = $("cursor-dot");
    const ring = $("cursor-ring");
    if (!dot || !ring) return;
    document.documentElement.classList.add("has-cursor");

    const dotX = gsap.quickTo(dot, "x", { duration: 0.12, ease: "power2.out" });
    const dotY = gsap.quickTo(dot, "y", { duration: 0.12, ease: "power2.out" });
    const ringX = gsap.quickTo(ring, "x", { duration: 0.42, ease: "power3.out" });
    const ringY = gsap.quickTo(ring, "y", { duration: 0.42, ease: "power3.out" });

    window.addEventListener("pointermove", (event) => {
      dotX(event.clientX - 3);
      dotY(event.clientY - 3);
      ringX(event.clientX - 19);
      ringY(event.clientY - 19);
    }, { passive: true });

    $$("a, button, label, select, .sample-chip, .desk-card, .b-item").forEach((el) => {
      el.addEventListener("pointerenter", () => ring.classList.add("is-active"));
      el.addEventListener("pointerleave", () => ring.classList.remove("is-active"));
    });
  }

  function initMagnetic() {
    if (!FINE_POINTER || REDUCED || !window.gsap) return;
    $$("[data-magnetic]").forEach((el) => {
      const xTo = gsap.quickTo(el, "x", { duration: 0.4, ease: "power3.out" });
      const yTo = gsap.quickTo(el, "y", { duration: 0.4, ease: "power3.out" });
      el.addEventListener("pointermove", (event) => {
        const rect = el.getBoundingClientRect();
        xTo((event.clientX - rect.left - rect.width / 2) * 0.22);
        yTo((event.clientY - rect.top - rect.height / 2) * 0.28);
      });
      el.addEventListener("pointerleave", () => { xTo(0); yTo(0); });
    });
  }

  function initHero() {
    const title = $("hero-title");
    if (title && !title.dataset.split && window.gsap && !REDUCED) {
      const lines = title.innerHTML.split(/<br\s*\/?>/i);
      title.innerHTML = lines
        .map((line) => `<span class="line-mask"><span class="line-inner">${line}</span></span>`)
        .join("");
      title.dataset.split = "true";
      gsap.from(title.querySelectorAll(".line-inner"), {
        yPercent: 115,
        duration: 1.05,
        ease: "power4.out",
        stagger: 0.12,
        delay: 0.25,
      });
    }
    if (title) setTimeout(() => title.classList.add("is-visible"), 120);
  }

  function initReveal() {
    const targets = $$(".reveal");
    if (!targets.length) return;
    if (!("IntersectionObserver" in window) || REDUCED) {
      targets.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    targets.forEach((el) => observer.observe(el));
  }

  /* Scroll-linked motion.
     ---------------------------------------------------------------------
     IntersectionObserver (above) decides *whether* something has appeared;
     ScrollTrigger decides *how much* the page has moved, which is what makes
     parallax and scrubbed progress possible. It is used only as an
     enhancement: without the plugin, or with reduced motion requested, every
     element is left in its final CSS state and nothing is pinned or hidden. */
  function initScrollScene() {
    if (REDUCED || !window.gsap || !window.ScrollTrigger) return;
    const { gsap: g, ScrollTrigger: ST } = window;
    g.registerPlugin(ST);

    // Hero: the copy drifts up and fades as the desk scrolls into view. Small
    // numbers on purpose — parallax should be felt, not watched.
    const hero = document.querySelector(".hero-inner");
    if (hero) {
      g.to(hero, {
        yPercent: -12,
        autoAlpha: 0.25,
        ease: "none",
        scrollTrigger: {
          trigger: ".hero",
          start: "top top",
          end: "bottom top",
          scrub: 0.6,
        },
      });
    }

    // The 3D field recedes in step with the hero so the two move as one plane.
    const canvas = $("scene-canvas");
    if (canvas) {
      g.to(canvas, {
        yPercent: 10,
        autoAlpha: 0.35,
        ease: "none",
        scrollTrigger: {
          trigger: ".hero",
          start: "top top",
          end: "bottom top",
          scrub: 0.8,
        },
      });
    }

    // Section headings settle in from the left — a reading cue, not a flourish.
    $$(".section-head h2").forEach((heading) => {
      g.from(heading, {
        x: -22,
        autoAlpha: 0,
        duration: 0.7,
        ease: "power3.out",
        scrollTrigger: { trigger: heading, start: "top 88%", once: true },
      });
    });

    // Static grids deal their cards in with a slight lift and stagger.
    // Only containers whose children exist at init belong here: #desks-grid is
    // filled from the registry response, so its cards are animated in
    // renderDesks() once they actually exist.
    [".insight-grid", ".method-steps"].forEach((selector) => {
      const grid = document.querySelector(selector);
      if (!grid) return;
      const cards = Array.from(grid.children);
      if (!cards.length) return;
      g.from(cards, {
        y: 24,
        autoAlpha: 0,
        duration: 0.55,
        ease: "power2.out",
        stagger: 0.08,
        scrollTrigger: { trigger: grid, start: "top 86%", once: true },
      });
    });

    // The method rail fills in proportion to how far it has been read, which
    // reads as progress rather than as a bar that simply appeared.
    const rail = $("method-progress");
    const board = document.querySelector(".method-board");
    if (rail && board) {
      g.fromTo(
        rail,
        { width: "0%" },
        {
          width: "100%",
          ease: "none",
          scrollTrigger: { trigger: board, start: "top 72%", end: "bottom 62%", scrub: 0.4 },
        },
      );
    }

    // Geometry changes (fonts, images, the insights grid filling in) move the
    // trigger points, so recalculate once the page has settled.
    window.addEventListener("load", () => ST.refresh(), { once: true });
  }

  function initNav() {
    const nav = $("nav");
    const bar = $("scroll-progress");    // Only in-page anchors: the nav also links to /lab, which is a route,
    // not a selector. Passing it to querySelector would throw and abort init.
    const links = $$('.nav-links a[href^="#"]');
    const sections = links
      .map((a) => document.querySelector(a.getAttribute("href")))
      .filter(Boolean);

    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const progress = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      if (bar) bar.style.transform = `scaleX(${progress})`;
      if (nav) nav.classList.toggle("is-scrolled", window.scrollY > 24);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    if ("IntersectionObserver" in window) {
      const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          links.forEach((a) => a.removeAttribute("aria-current"));
          const link = byId.get(entry.target.id);
          if (link) link.setAttribute("aria-current", "page");
        });
      }, { rootMargin: "-45% 0px -45% 0px" });
      sections.forEach((section) => observer.observe(section));
    }

    $$('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener("click", (event) => {
        const id = anchor.getAttribute("href");
        const section = document.querySelector(id);
        if (!section) return;
        event.preventDefault();
        if (window.ScrollToPlugin && window.gsap) {
          gsap.to(window, {
            duration: REDUCED ? 0 : 0.9,
            ease: "power2.inOut",
            scrollTo: { y: section, offsetY: 58 },
          });
        } else {
          section.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
        }
      });
    });
  }

  /* ============================================================ content */
  function initSamples() {
    const SAMPLES = [
      ["The striker's late header sent his side through to the semi-final after a tense replay.", "Sports"],
      ["Shares fell sharply after the central bank raised interest rates for a third time this year.", "Business"],
      ["Engineers released a patch for a flaw affecting millions of home routers worldwide.", "Sci/Tech"],
      ["Envoys met for a third day of talks over the disputed border region, mediators said.", "World"],
    ];
    const host = $("samples");
    if (!host) return;
    host.innerHTML = SAMPLES.map(
      ([text, label], index) =>
        `<button class="sample-chip" type="button" data-sample="${index}" title="${esc(text)}">${esc(label)}</button>`,
    ).join("");
    host.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-sample]");
      if (!chip) return;
      const copy = $("copy");
      copy.value = SAMPLES[Number(chip.dataset.sample)][0];
      copy.focus();
      updateCount();
    });
  }

  function initTicker() {
    const track = $("ticker-track");
    if (!track) return;
    const items = [
      "AG News · 20,000 training rows",
      "Bronze · TF-IDF + logistic regression",
      "Silver · Word2Vec CBOW + BiLSTM",
      "Gold · fine-tuned DistilBERT",
      "Macro-F1 release gate · 0.85",
      "Explanations · per-tier, not generic",
      "Feedback · one JSONL line per correction",
      "Prometheus · latency, confidence, fallbacks",
    ];
    const html = items.map((item) => `<span class="ticker-item">${esc(item)}</span>`).join("");
    track.innerHTML = html + html; // duplicated for a seamless -50% loop
  }

  async function loadTiers() {
    const grid = $("desks-grid");
    const caption = $("desks-caption");
    try {
      const tiers = await API.get("/v1/tiers");
      state.tiers = tiers;
      renderDesks(tiers);
      if (caption) {
        const ready = tiers.filter((t) => t.ready).length;
        caption.textContent = `${ready} of ${tiers.length} desks open · measured on held-out data · latency normalised per request`;
      }
      populateFeedbackLabels();
    } catch (error) {
      if (grid) {
        grid.innerHTML = `<p class="caption" style="grid-column:1/-1;">The registry could not be read: ${esc(error.message)}. The API may still be starting.</p>`;
      }
      if (caption) caption.textContent = "registry unavailable";
    }
  }

  function renderDesks(tiers) {
    const grid = $("desks-grid");
    if (!grid) return;
    grid.innerHTML = tiers
      .map((tier) => {
        const name = (tier.display_name || tier.tier).split(/\s[-–]\s/)[0];
        const accuracy = tier.accuracy == null ? "—" : pct(tier.accuracy);
        const f1 = tier.macro_f1 == null ? "—" : tier.macro_f1.toFixed(4);
        const perRequest =
          tier.latency_ms_p50 == null ? "—" : (tier.latency_ms_p50 / 100).toFixed(tier.latency_ms_p50 < 100 ? 2 : 1) + " ms";
        const meter = tier.accuracy == null ? 0 : Math.max(2, tier.accuracy * 100);
        return `<article class="desk-card" data-tier="${esc(tier.tier)}">
          <div class="desk-top">
            <h3 class="desk-name">${esc(name)}</h3>
            <span class="desk-badge ${tier.ready ? "is-ready" : ""}">${tier.ready ? "ready" : "not trained"}</span>
          </div>
          <p class="desk-model">${esc(tier.model_name)} · v${esc(tier.model_version)}</p>
          <p class="desk-desc">${esc(tier.description)}</p>
          <div class="desk-metrics">
            <div class="desk-metric"><span>Accuracy</span><b>${accuracy}</b></div>
            <div class="desk-metric"><span>Macro F1</span><b>${f1}</b></div>
            <div class="desk-metric"><span>Latency / req</span><b>${perRequest}</b></div>
            <div class="desk-metric"><span>Artifact</span><b>${fmtSize(tier.size_mb)}</b></div>
            <div class="desk-meter" aria-hidden="true"><span style="width:${meter.toFixed(1)}%"></span></div>
          </div>
        </article>`;
      })
      .join("");

    // The skeleton is gone and the real cards are in the DOM: this is the
    // earliest moment they can be animated. If ScrollTrigger is present the
    // deal-in waits until the grid is actually on screen; otherwise the cards
    // simply appear, which is the correct non-animated behaviour.
    const cards = Array.from(grid.children);
    if (!cards.length || REDUCED || !window.gsap) return;
    if (window.ScrollTrigger) {
      gsap.from(cards, {
        y: 22,
        autoAlpha: 0,
        duration: 0.5,
        ease: "power2.out",
        stagger: 0.09,
        scrollTrigger: { trigger: grid, start: "top 88%", once: true },
      });
    } else {
      gsap.from(cards, { autoAlpha: 0, y: 16, duration: 0.45, stagger: 0.07, ease: "power2.out" });
    }
  }

  function populateFeedbackLabels() {
    const select = $("feedback-label");
    if (!select) return;
    select.innerHTML = state.labels.map((label) => `<option value="${esc(label)}">${esc(label)}</option>`).join("");
  }

  /* ============================================================ verdict */
  let ringAnimation = null;

  function renderVerdict(body) {
    state.last = body;
    const empty = $("verdict-empty");
    const result = $("verdict-result");
    if (!empty || !result) return;
    empty.hidden = true;
    result.hidden = false;

    const rounded = body.scores.slice().sort((a, b) => b.probability - a.probability);
    const top = rounded[0];
    const ref = body.__requestId ? body.__requestId.slice(0, 8) : Math.random().toString(16).slice(2, 10);
    const refEl = $("verdict-ref");
    if (refEl) refEl.textContent = "ref · " + ref.toUpperCase();

    const labelEl = $("verdict-label");
    if (labelEl) labelEl.textContent = body.label;
    const stamp = $("verdict-stamp");
    if (stamp) stamp.textContent = body.confidence >= 0.6 ? "Filed under" : "Tentatively filed under";

    const tierEl = $("meta-tier");
    if (tierEl) {
      tierEl.textContent = body.tier;
      tierEl.dataset.tier = body.tier;
    }
    const latencyEl = $("meta-latency");
    if (latencyEl) latencyEl.textContent = Math.round(body.latency_ms) + " ms";
    const versionEl = $("meta-version");
    if (versionEl) versionEl.textContent = "v" + body.model_version;
    const timeEl = $("meta-time");
    if (timeEl) timeEl.textContent = clock();

    // Confidence ring: r=52 → circumference 326.73.
    const circumference = 2 * Math.PI * 52;
    const ring = $("confidence-ring");
    const value = $("confidence-value");
    if (ring) ring.style.strokeDasharray = String(circumference);
    if (value) value.textContent = pct(body.confidence, 1);
    if (ringAnimation) ringAnimation.kill();
    if (window.gsap) {
      const dial = { v: 0 };
      ringAnimation = gsap.to(dial, {
        v: body.confidence,
        duration: REDUCED ? 0 : 1.1,
        ease: "power2.out",
        onUpdate: () => {
          if (value) value.textContent = pct(dial.v, 1);
          if (ring) ring.style.strokeDashoffset = String(circumference * (1 - dial.v));
        },
      });
    } else if (ring) {
      ring.style.strokeDashoffset = String(circumference * (1 - body.confidence));
    }

    const bars = $("bars");
    if (bars) {
      bars.innerHTML = rounded
        .map(
          (score, index) => `<div class="bar ${index === 0 ? "is-best" : ""}" data-label="${esc(score.label)}" role="listitem">
            <div class="bar-top"><span class="bar-label">${esc(score.label)}</span><span class="bar-value mono">${pct(score.probability)}</span></div>
            <div class="bar-track"><span class="bar-fill" style="width:0%"></span></div>
          </div>`,
        )
        .join("");
      requestAnimationFrame(() => {
        $$(".bar-fill", bars).forEach((fill, index) => {
          fill.style.width = pct(rounded[index].probability);
        });
      });
    }

    renderExplanation(body);
    const select = $("feedback-label");
    if (select) select.value = body.label;
    if (window.gsap && !REDUCED) {
      gsap.from(result, { autoAlpha: 0, y: 12, duration: 0.5, ease: "power3.out" });
    }
  }

  function renderExplanation(body) {
    const method = $("explain-method");
    const tokens = $("explain-tokens");
    if (!tokens) return;
    const list = Array.isArray(body.tokens) ? body.tokens : [];
    if (method) {
      method.textContent = list.length
        ? body.explanation_method || "contributions"
        : body.explanation_context || "no token attribution available for this tier";
    }
    if (!list.length) {
      tokens.innerHTML = `<span class="hint">${esc(body.explanation_context || "This tier did not return token evidence.")}</span>`;
      return;
    }
    const max = Math.max(
      ...list.map((token) => Math.abs(token.contribution != null ? token.contribution : token.importance || 0)),
      1e-9,
    );
    tokens.innerHTML = list
      .map((token) => {
        const magnitude = Math.abs(token.contribution != null ? token.contribution : token.importance || 0);
        const weight = Math.max(0.08, magnitude / max);
        const negative = token.contribution != null && token.contribution < 0;
        const value = token.contribution != null
          ? (token.contribution > 0 ? "+" : "") + token.contribution.toFixed(2)
          : ((token.importance || 0) * 100).toFixed(0) + "%";
        return `<span class="token-pill ${negative ? "is-negative" : ""}" style="--w:${weight.toFixed(3)}" title="${esc(value)}">${esc(token.token)} <span class="mono">${esc(value)}</span></span>`;
      })
      .join("");
    if (window.gsap && !REDUCED) {
      gsap.from($$(".token-pill", tokens), { autoAlpha: 0, y: 6, duration: 0.3, stagger: 0.03 });
    }
  }

  async function explainAgain() {
    if (!state.last || !state.filedText) {
      toast("File a story first, then I can show the evidence.", "error");
      return;
    }
    const button = $("btn-explain");
    if (button) button.disabled = true;
    try {
      const tier = $("meta-tier")?.textContent || undefined;
      const payload = { text: state.filedText };
      if (tier && TIER_ORDER.includes(tier)) payload.tier = tier;
      const body = await API.post("/v1/explain", payload, 150000);
      body.label = state.last.label;
      renderExplanation(body);
      toast("Explanation refreshed for this filing.");
    } catch (error) {
      toast("Could not explain: " + error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  }

  function updateCount() {
    const copy = $("copy");
    const count = $("char-count");
    if (copy && count) count.textContent = `${copy.value.length.toLocaleString()} chars`;
  }

  /* ============================================================ filing */
  function currentTier() {
    const checked = document.querySelector('input[name="tier"]:checked');
    return checked && checked.value ? checked.value : null;
  }

  async function fileStory() {
    const copy = $("copy");
    const button = $("btn-file");
    if (!copy || state.busy) return;
    const text = copy.value.trim();
    if (!text) {
      toast("There is no copy to file.", "error");
      copy.focus();
      return;
    }

    state.busy = true;
    state.filedText = text;
    const loading = $("verdict-loading");
    if (loading) loading.hidden = false;
    if (button) {
      button.disabled = true;
      button.dataset.label = button.textContent;
      button.textContent = "Filing…";
    }

    try {
      const payload = { text };
      const tier = currentTier();
      if (tier) payload.tier = tier;
      const body = await API.post("/v1/explain", payload, 150000);
      renderVerdict(body);
      addBulletin(text, body);
      loadInsights();
      checkStatus();
    } catch (error) {
      toast("The desk could not file this copy: " + error.message, "error");
    } finally {
      state.busy = false;
      if (loading) loading.hidden = true;
      if (button) {
        button.disabled = false;
        button.textContent = button.dataset.label || "File story";
      }
    }
  }

  function renderBulletin() {
    const list = $("bulletin");
    const empty = $("bulletin-empty");
    const count = $("bulletin-count");
    if (!list) return;

    if (empty) empty.hidden = state.bulletin.length > 0;
    if (count) count.textContent = `${state.bulletin.length} filed`;

    list.innerHTML = state.bulletin
      .map((entry, index) => {
        const snippet = entry.text.length > 96 ? entry.text.slice(0, 96).trim() + "…" : entry.text;
        return `<li>
          <button class="b-item" type="button" data-index="${index}" title="Load this copy back into the desk">
            <span class="b-time">${clock(new Date(entry.at))}</span>
            <span class="b-label">${esc(entry.label)}</span>
            <span class="b-snippet">${esc(snippet)}</span>
            <span class="b-conf">${pct(entry.confidence, 1)}</span>
            <span class="b-tier" data-tier="${esc(entry.tier)}">${esc(entry.tier)}</span>
          </button>
        </li>`;
      })
      .join("");
  }

  function addBulletin(text, body) {
    const list = $("bulletin");
    if (!list) return;
    state.bulletin.unshift({
      text,
      label: body.label,
      confidence: body.confidence,
      tier: body.tier,
      at: Date.now(),
    });
    state.bulletin = state.bulletin.slice(0, 7);
    persistBulletin();
    renderBulletin();

    if (window.gsap && !REDUCED && list.firstElementChild) {
      gsap.from(list.firstElementChild, { autoAlpha: 0, y: -8, duration: 0.45, ease: "power2.out" });
    }
  }

  function initBulletin() {
    const list = $("bulletin");
    if (!list) return;
    state.bulletin = loadStoredBulletin();
    renderBulletin();

    list.addEventListener("click", (event) => {
      const item = event.target.closest("[data-index]");
      if (!item) return;
      const entry = state.bulletin[Number(item.dataset.index)];
      if (!entry) return;
      const copy = $("copy");
      copy.value = entry.text;
      updateCount();
      copy.focus();
      copy.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "center" });
    });
  }

  /* ============================================================ compare */
  async function compareDesks() {
    const copy = $("copy");
    const panel = $("compare-panel");
    const grid = $("compare-grid");
    const note = $("compare-note");
    const button = $("btn-compare");
    if (!copy || !panel || !grid) return;
    const text = copy.value.trim();
    if (!text) {
      toast("There is no copy to compare.", "error");
      return;
    }
    if (!state.tiers.length) await loadTiers();
    const open = state.tiers.filter((tier) => tier.ready).map((tier) => tier.tier);
    if (!open.length) {
      toast("No desk is open right now.", "error");
      return;
    }

    panel.hidden = false;
    const emptyVerdict = $("verdict-empty");
    if (emptyVerdict) emptyVerdict.hidden = true;
    const ref = $("verdict-ref");
    if (ref && ref.textContent.trim() === "—") ref.textContent = "ref · compare";
    if (button) button.disabled = true;

    const rowFor = (tier) => {
      const row = document.createElement("div");
      row.className = "compare-row";
      row.dataset.tier = tier;
      row.innerHTML = `<span class="compare-tier">${esc(tier)}</span>
        <span class="compare-label"><span class="dim">running…</span></span>
        <span class="compare-conf">—</span>`;
      return row;
    };

    // Render a row as soon as its desk answers. Gold is ~100x slower than
    // bronze, so waiting for the slowest desk would look like a hang.
    grid.innerHTML = "";
    const rows = open.map((tier) => {
      const row = rowFor(tier);
      grid.appendChild(row);
      return row;
    });
    if (note) note.textContent = "bronze answers first; gold may take a few seconds on first load.";

    const results = [];
    try {
      for (const [index, tier] of open.entries()) {
        const row = rows[index];
        try {
          const result = await API.post("/v1/predict", { text, tier }, 150000);
          results.push(result);
          row.innerHTML = `<span class="compare-tier">${esc(tier)}</span>
            <span class="compare-label"><span>${esc(result.label)}</span><span class="compare-track"><span style="width:0%" data-width="${pct(result.confidence)}"></span></span></span>
            <span class="compare-conf">${pct(result.confidence)}</span>`;
          requestAnimationFrame(() => {
            const fill = row.querySelector(".compare-track span");
            if (fill) fill.style.width = fill.dataset.width;
          });
        } catch {
          row.classList.add("is-failed");
          row.innerHTML = `<span class="compare-tier">${esc(tier)}</span>
            <span class="compare-label"><span class="dim">desk did not answer</span></span>
            <span class="compare-conf">failed</span>`;
        }
      }
    } finally {
      if (button) button.disabled = false;
    }

    if (note) {
      const labels = new Set(results.map((result) => result.label));
      note.textContent = results.length === 0
        ? "No desk answered this copy."
        : labels.size === 1
          ? `All ${results.length} desks agree on ${results[0].label}.`
          : `The desks disagree (${results.length} answered). Disagreement is where the useful review is.`;
    }
  }

  /* ============================================================ feedback */
  async function sendFeedback() {
    const button = $("btn-feedback");
    const select = $("feedback-label");
    if (!state.last || !select) {
      toast("File a story before correcting it.", "error");
      return;
    }
    const body = state.last;
    if (button) button.disabled = true;
    try {
      await API.post("/v1/feedback", {
        text: state.filedText || $("copy").value.trim(),
        corrected_label: select.value,
        predicted_label: body.label,
        predicted_confidence: body.confidence,
        tier: body.tier,
        reason: "corrected in the Wire Desk UI",
      });
      toast(
        select.value === body.label
          ? `Agreement recorded: ${body.label}. It still counts as signal.`
          : `Correction recorded: ${body.label} → ${select.value}. It is now retraining data.`,
      );
    } catch (error) {
      toast("Could not record the correction: " + error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  }

  /* ============================================================ insights */
  function renderMix(entries, total) {
    const segments = $("mix-segments");
    const legend = $("mix-legend");
    if (!segments || !legend) return;
    const circumference = 2 * Math.PI * 54;
    let offset = 0;
    segments.innerHTML = entries
      .map((entry) => {
        const share = total ? entry.count / total : 0;
        const length = Math.max(0, share * circumference - 2.5);
        const dash = `${length} ${Math.max(0, circumference - length)}`;
        const rotation = offset;
        offset += share * circumference;
        return `<circle class="mix-seg" cx="70" cy="70" r="54" stroke="${TOPIC_COLORS[entry.label] || "#f5f0e6"}"
          stroke-dasharray="${dash}" stroke-dashoffset="${(-rotation).toFixed(2)}" stroke-linecap="butt"></circle>`;
      })
      .join("");
    legend.innerHTML = entries
      .map(
        (entry) => `<li><span class="mix-swatch" style="background:${TOPIC_COLORS[entry.label] || "#f5f0e6"}"></span>
          <span>${esc(entry.label)}</span><b>${pct(entry.count / Math.max(1, total), 0)}</b></li>`,
      )
      .join("");
  }

  function renderSparkline(series) {
    const svg = $("sparkline");
    if (!svg) return;
    const W = 600;
    const H = 120;
    const pad = 6;
    const buckets = 12;
    const totals = new Array(buckets).fill(0);
    Object.values(series || {}).forEach((values) => {
      values.forEach((value, index) => {
        if (index < buckets) totals[index] += value;
      });
    });
    const max = Math.max(...totals, 1);
    const points = totals.map((value, index) => {
      const x = pad + (index / (buckets - 1)) * (W - pad * 2);
      const y = H - pad - (value / max) * (H - pad * 2);
      return [x, y];
    });
    const line = points.map(([x, y], index) => (index === 0 ? `M${x},${y}` : `L${x},${y}`)).join(" ");
    const area = `${line} L${W - pad},${H - pad} L${pad},${H - pad} Z`;
    svg.innerHTML = `
      <defs><linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="rgba(231,195,126,.35)"/>
        <stop offset="100%" stop-color="rgba(231,195,126,0)"/>
      </linearGradient></defs>
      <path d="${area}" fill="url(#spark-fill)"></path>
      <path d="${line}" fill="none" stroke="#e7c37e" stroke-width="1.6" stroke-linejoin="round"></path>
      ${points.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="2.2" fill="#f5f0e6" opacity=".75"></circle>`).join("")}`;
    if (window.gsap && !REDUCED) {
      const path = svg.querySelector("path:nth-of-type(2)");
      if (path) {
        const length = path.getTotalLength ? path.getTotalLength() : 0;
        if (length) gsap.fromTo(path, { strokeDasharray: length, strokeDashoffset: length }, { strokeDashoffset: 0, duration: 1.1, ease: "power2.out" });
      }
    }
  }

  // The API stamps events in UTC; the page speaks in the reader's clock, the
  // same way the filing time on the assignment card does.
  function stampLocal(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return String(iso).replace("T", " ").slice(0, 16);
    const pad = (value) => String(value).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  async function loadInsights() {
    const set = (id, value) => {
      const el = $(id);
      if (el) el.textContent = value;
    };
    try {
      const data = await API.get("/v1/insights");
      if (data.empty) {
        renderMix([], 0);
        renderSparkline({});
        set("ins-events", "0 events");
        ["ins-p50", "ins-p95", "ins-mean", "ins-conf", "ins-low", "ins-fallback"].forEach((id) => set(id, "—"));
        set("ins-updated", "no traffic yet");
        const usage = $("tier-usage");
        if (usage) usage.innerHTML = `<li><span class="dim">tier usage appears after the first filing</span></li>`;
        return;
      }
      const total = data.label_mix.reduce((sum, entry) => sum + entry.count, 0);
      renderMix(data.label_mix, total);
      renderSparkline(data.series_labels);
      set("ins-events", `${data.window.recent_events} events`);
      set("ins-p50", Math.round(data.latency.p50_ms) + " ms");
      set("ins-p95", Math.round(data.latency.p95_ms) + " ms");
      set("ins-mean", Math.round(data.latency.mean_ms) + " ms");
      set("ins-conf", pct(data.confidence.mean, 1));
      set("ins-low", pct(data.confidence.low_confidence_share, 0));
      set("ins-fallback", String(data.fallbacks));
      set("ins-updated", stampLocal(data.updated_at));

      const usage = $("tier-usage");
      if (usage) {
        const maxTier = Math.max(...data.tier_usage.map((entry) => entry.count), 1);
        usage.innerHTML = data.tier_usage
          .map(
            (entry) => `<li><span>${esc(entry.tier)}</span>
              <i><span style="width:${((entry.count / maxTier) * 100).toFixed(1)}%"></span></i>
              <b>${entry.count}</b></li>`,
          )
          .join("");
      }
    } catch {
      set("ins-updated", "insights unavailable");
    }
  }

  /* ============================================================ method */
  function initMethod() {
    const steps = $$(".method-step");
    const progress = $("method-progress");
    if (!steps.length) return;
    const counters = $$("[data-count]");

    // Ownership of the rail is exclusive. When ScrollTrigger is driving it
    // (initScrollScene), it scrubs the width continuously and this function
    // must not also write it per active step — two writers on one property
    // produce a bar that jitters between the two values. So the rail is
    // scroll-driven when available, and step-driven only as the fallback.
    const railDrivenByScroll = Boolean(!REDUCED && window.gsap && window.ScrollTrigger);
    const setRail = (width) => {
      if (progress && !railDrivenByScroll) progress.style.width = width;
    };

    const startCounter = (el) => {
      if (el.dataset.started) return;
      el.dataset.started = "true";
      const target = Number(el.dataset.count || 0);
      const decimals = Number(el.dataset.decimals || 0);
      const suffix = el.dataset.suffix || "";
      const format = (value) =>
        value.toLocaleString("en-US", {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        }) + suffix;
      if (REDUCED) {
        el.textContent = format(target);
        return;
      }
      const started = performance.now();
      const duration = 1200;
      const tick = (now) => {
        const progress = Math.min(1, (now - started) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = format(target * eased);
        if (progress < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    };

    if (!("IntersectionObserver" in window)) {
      counters.forEach(startCounter);
      steps.forEach((step) => step.classList.add("is-active"));
      setRail("100%");
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const index = steps.indexOf(entry.target);
        entry.target.classList.add("is-active");
        const stepCount = steps.filter((step) => step.classList.contains("is-active")).length;
        setRail(((stepCount / steps.length) * 100).toFixed(1) + "%");
        if (index >= 0) {
          const counter = entry.target.querySelector("[data-count]");
          if (counter) startCounter(counter);
        }
      });
    }, { rootMargin: "-20% 0px -40% 0px", threshold: 0.2 });
    steps.forEach((step) => observer.observe(step));
  }

  function initTerminal() {
    const host = $("terminal-text");
    if (!host) return;
    const tierLine = (tier) => {
      const found = state.tiers.find((entry) => entry.tier === tier);
      if (!found) return null;
      return `  ${tier.padEnd(7)} acc=${found.accuracy == null ? "—" : found.accuracy.toFixed(3)}  f1=${found.macro_f1 == null ? "—" : found.macro_f1.toFixed(3)}  size=${fmtSize(found.size_mb)}`;
    };
    const lines = [
      { text: "python scripts/run_pipeline.py --serve", cls: "t-dim" },
      { text: "Preflight: environment and dependencies", cls: "" },
      { text: "  ok    python + artifacts directory writable", cls: "t-ok" },
      { text: "Verify: loading each trained tier through the registry", cls: "" },
      { text: "  ok    probabilities are n x 4 and every row sums to 1", cls: "t-ok" },
      { text: "Registry: merging per-tier metrics into registry.json", cls: "" },
      { text: tierLine("bronze") || "  bronze  acc=0.895  f1=0.894  size=0.8 MB", cls: "" },
      { text: tierLine("silver") || "  silver  acc=0.892  f1=0.891  size=9.2 MB", cls: "" },
      { text: tierLine("gold") || "  gold    acc=0.907  f1=0.907  size=268.6 MB", cls: "" },
      { text: "Evaluate: release gate (macro-F1 >= 0.85)", cls: "" },
      { text: "  ok    every ready tier cleared the gate", cls: "t-ok" },
      { text: "PIPELINE COMPLETE - artifacts verified and gated", cls: "t-ok" },
      { text: "Serve: uvicorn newsclass.app:app", cls: "t-dim" },
    ];

    let played = false;
    const play = () => {
      if (played) return;
      played = true;
      host.innerHTML = "";
      if (REDUCED) {
        host.innerHTML = lines.map((line) => `<span class="${line.cls}">${esc(line.text)}</span>`).join("\n");
        return;
      }
      let index = 0;
      const write = () => {
        if (index >= lines.length) {
          host.insertAdjacentHTML("beforeend", '<span class="t-cursor" aria-hidden="true"></span>');
          return;
        }
        const line = lines[index];
        host.insertAdjacentHTML("beforeend", `<span class="${line.cls}">${esc(line.text)}</span>`);
        host.appendChild(document.createTextNode("\n"));
        index += 1;
        setTimeout(write, line.cls ? 110 : 190);
      };
      write();
    };

    if (!("IntersectionObserver" in window)) {
      play();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        play();
        observer.disconnect();
      }
    }, { threshold: 0.25 });
    observer.observe(host);
  }

  /* ============================================================ status */
  async function loadServiceInfo() {
    try {
      const model = await API.get("/v1/model");
      if (Array.isArray(model.labels) && model.labels.length) {
        state.labels = model.labels;
        populateFeedbackLabels();
      }
      state.defaultTier = model.default_tier || null;
      state.allowOverride = model.allow_tier_override !== false;
      const backend = $("foot-backend");
      if (backend) backend.textContent = "metrics: " + (model.metrics_backend || "—");
    } catch {
      /* catalogue endpoint is optional; /v1/tiers still drives the cards */
    }
    try {
      const health = await API.get("/healthz");
      const version = $("foot-version");
      if (version && health.version) version.textContent = "v" + health.version;
    } catch {
      /* keep the static version label */
    }
  }

  async function checkStatus() {
    const pill = $("status-pill");
    const text = $("status-text");
    try {
      const ready = await API.get("/readyz");
      if (pill) {
        pill.classList.add("is-live");
        pill.classList.remove("is-down");
      }
      if (text) text.textContent = `live · ${ready.default_tier}`;
    } catch {
      if (pill) {
        pill.classList.remove("is-live");
        pill.classList.add("is-down");
      }
      if (text) text.textContent = "offline";
    }
  }

  /* ============================================================ wiring */
  function init() {
    document.documentElement.classList.add("anims");
    runPreloader();
    initScene();
    initCursor();
    initMagnetic();
    initHero();
    initReveal();
    initScrollScene();
    initNav();
    initSamples();
    initTicker();
    // Labels exist before the first network call, so a very fast first filing
    // can still be corrected instead of hitting an empty <select>.
    initMethod();
    initTerminal();
    initBulletin();

    const copy = $("copy");
    if (copy) {
      copy.addEventListener("input", updateCount);
      copy.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
          event.preventDefault();
          fileStory();
        }
      });
    }
    updateCount();

    populateFeedbackLabels();

    $("btn-file")?.addEventListener("click", fileStory);
    $("btn-compare")?.addEventListener("click", compareDesks);
    $("btn-compare-close")?.addEventListener("click", () => {
      const panel = $("compare-panel");
      if (panel) panel.hidden = true;
    });
    $("btn-explain")?.addEventListener("click", explainAgain);
    $("btn-feedback")?.addEventListener("click", sendFeedback);
    $("btn-clear")?.addEventListener("click", () => {
      if (!copy) return;
      copy.value = "";
      updateCount();
      state.last = null;
      state.filedText = "";
      const result = $("verdict-result");
      const empty = $("verdict-empty");
      if (result) result.hidden = true;
      if (empty) empty.hidden = false;
      const ref = $("verdict-ref");
      if (ref) ref.textContent = "—";
      copy.focus();
    });

    const version = $("foot-version");
    if (version && window.NEWSCLASS_VERSION) version.textContent = "v" + window.NEWSCLASS_VERSION;

    loadServiceInfo();
    loadTiers();
    loadInsights();
    checkStatus();
    setInterval(checkStatus, 30000);
    setInterval(loadInsights, 15000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
