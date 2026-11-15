/* ==========================================================================
   Model Lab — run every tier independently on the same copy.
   --------------------------------------------------------------------------
   Deliberately framework-free: this page is a measuring instrument, so the
   code that talks to the API should be readable and auditable. No result is
   shared between cards; each one is its own POST /v1/predict.
   ========================================================================== */

const ORDER = ["bronze", "silver", "gold"];
const LIMIT = 20_000;
const PREDICT_TIMEOUT = 180_000;
const EXPLAIN_TIMEOUT = 180_000;

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case '"': return "&quot;";
      default: return "&#39;";
    }
  });

const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const fmtMs = (value) => {
  const ms = Number(value);
  if (!Number.isFinite(ms)) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
};
const cap = (value) => String(value || "").replace(/^./, (c) => c.toUpperCase());
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const SAMPLES = [
  ["Chip launch", "Apple unveiled a new chip on Tuesday, claiming it is twice as fast as the previous generation while using less power."],
  ["Match report", "The striker scored twice in the second half as the visitors came from behind to win 3-2."],
  ["Rate decision", "The central bank raised interest rates by a quarter point, citing persistent services inflation."],
  ["Ceasefire", "A ceasefire was announced after weeks of border clashes, aid agencies said."],
  ["Off-domain", "Whisk the eggs with the sugar until the mixture is pale and thick."],
];

/* ------------------------------------------------------------------- API */

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function detailMessage(data, status) {
  const detail = data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => (typeof item === "string" ? item : item?.msg || ""))
      .filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  return `the desk returned HTTP ${status}`;
}

async function request(path, { method = "GET", body = null, timeout = 15_000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(detailMessage(data, response.status), response.status);
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new ApiError(`no answer within ${Math.round(timeout / 1000)} s`, 0);
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError("network error — is the service still running?", 0);
  } finally {
    clearTimeout(timer);
  }
}

/* ----------------------------------------------------------------- state */

const state = {
  tiers: [],
  byTier: new Map(),
  results: new Map(),
  explanations: new Map(),
  running: new Set(),
  sequence: new Map(),
  epoch: 0,
  allowOverride: true,
  defaultTier: null,
  calls: 0,
};

const el = {
  copy: $("#lab-copy"),
  count: $("#lab-count"),
  samples: $("#lab-samples"),
  runAll: $("#lab-run-all"),
  clear: $("#lab-clear"),
  grid: $("#lab-grid"),
  inlineError: $("#lab-inline-error"),
  agreement: $("#sum-agreement"),
  fastest: $("#sum-fastest"),
  spread: $("#sum-spread"),
  calls: $("#sum-calls"),
  statusPill: $("#status-pill"),
  statusText: $("#status-text"),
  lamp: $("#lamp"),
  version: $("#lab-version"),
  backend: $("#lab-backend"),
};

/* ---------------------------------------------------------------- chrome */

function setStatus(live, label = "") {
  el.statusPill.classList.toggle("is-live", Boolean(live));
  el.statusPill.classList.toggle("is-down", !live);
  el.statusText.textContent = live ? `live${label ? ` · ${label}` : ""}` : "offline";
}

function showInline(message) {
  el.inlineError.textContent = message;
  el.inlineError.hidden = false;
}

function clearInline() {
  el.inlineError.textContent = "";
  el.inlineError.hidden = true;
}

function syncCount() {
  const length = el.copy.value.length;
  el.count.textContent = `${length.toLocaleString()} / ${LIMIT.toLocaleString()}`;
  el.count.classList.toggle("is-near", length > LIMIT * 0.9);
}

function syncNav() {
  $("#nav")?.classList.toggle("is-scrolled", window.scrollY > 12);
}

/* ----------------------------------------------------------------- cards */

function tierInfo(tier) {
  return state.byTier.get(tier) || { tier, display_name: cap(tier), ready: false };
}

function cardFor(tier) {
  return $(`.lab-model[data-tier="${tier}"]`);
}

function createCard(info) {
  const tier = info.tier;
  const card = document.createElement("article");
  card.className = "lab-model";
  card.dataset.tier = tier;
  card.dataset.state = "idle";
  card.setAttribute("aria-labelledby", `lab-tier-${tier}`);
  card.innerHTML = `
    <header class="lab-model-head">
      <div class="lab-tier">
        <h3 class="lab-tier-name" id="lab-tier-${esc(tier)}">${esc(info.display_name || cap(tier))}</h3>
        <span class="lab-tier-model mono">${esc(info.model_name || "model metadata unavailable")}</span>
      </div>
      <span class="lab-badge" data-role="badge">${info.ready ? "ready" : "not installed"}</span>
    </header>
    <div class="lab-verdict">
      <span class="lab-label is-empty" data-role="label">No run yet</span>
      <span class="lab-conf mono" data-role="conf">—</span>
    </div>
    <div class="bars" data-role="bars" aria-label="${esc(cap(tier))} class probabilities">
      <p class="hint mono" data-role="bars-empty">probabilities appear after a run</p>
    </div>
    <dl class="lab-meta">
      <div><dt>server</dt><dd class="mono" data-role="server">—</dd></div>
      <div><dt>round trip</dt><dd class="mono" data-role="rtt">—</dd></div>
      <div><dt>size</dt><dd class="mono" data-role="size">${info.size_mb ? `${Number(info.size_mb).toFixed(1)} MB` : "—"}</dd></div>
      <div><dt>test acc</dt><dd class="mono" data-role="acc">${info.accuracy != null ? pct(info.accuracy) : "—"}</dd></div>
    </dl>
    <div class="lab-model-actions">
      <button class="btn btn-ghost" type="button" data-role="run" ${info.ready ? "" : "disabled"}>Run this desk</button>
      <button class="btn btn-quiet" type="button" data-role="explain" disabled>Explain</button>
    </div>
    <div class="lab-explain" data-role="explain-panel" hidden>
      <div class="lab-explain-head">
        <span class="mono dim" data-role="method"></span>
        <button class="lab-explain-close" type="button" data-role="explain-close" aria-label="Hide explanation">hide</button>
      </div>
      <div class="tokens" data-role="tokens"></div>
    </div>
    <p class="lab-error" data-role="error" hidden></p>
  `;

  $('[data-role="run"]', card).addEventListener("click", () => runTier(tier));
  $('[data-role="explain"]', card).addEventListener("click", () => toggleExplain(tier));
  $('[data-role="explain-close"]', card).addEventListener("click", () => {
    $('[data-role="explain-panel"]', card).hidden = true;
  });
  return card;
}

function renderCards(tiers) {
  const ordered = [...tiers].sort(
    (a, b) => ORDER.indexOf(a.tier) - ORDER.indexOf(b.tier),
  );
  el.grid.innerHTML = "";
  el.grid.setAttribute("aria-busy", "false");
  if (!ordered.length) {
    el.grid.innerHTML = '<p class="lab-loading mono">no desks reported by the registry</p>';
    return;
  }
  for (const info of ordered) {
    state.byTier.set(info.tier, info);
    el.grid.appendChild(createCard(info));
  }
  if (window.gsap && !reduceMotion) {
    window.gsap.from($$(".lab-model", el.grid), {
      y: 22,
      opacity: 0,
      duration: 0.55,
      stagger: 0.07,
      ease: "power2.out",
    });
  }
}

function resetCard(tier) {
  const card = cardFor(tier);
  if (!card) return;
  const info = tierInfo(tier);
  card.dataset.state = "idle";
  card.classList.remove("is-busy");
  card.setAttribute("aria-busy", "false");
  $('[data-role="label"]', card).textContent = "No run yet";
  $('[data-role="label"]', card).classList.add("is-empty");
  $('[data-role="conf"]', card).textContent = "—";
  $('[data-role="bars"]', card).innerHTML =
    '<p class="hint mono" data-role="bars-empty">probabilities appear after a run</p>';
  $('[data-role="server"]', card).textContent = "—";
  $('[data-role="rtt"]', card).textContent = "—";
  $('[data-role="badge"]', card).textContent = info.ready ? "ready" : "not installed";
  $('[data-role="error"]', card).hidden = true;
  $('[data-role="explain-panel"]', card).hidden = true;
  $('[data-role="tokens"]', card).innerHTML = "";
  const explain = $('[data-role="explain"]', card);
  explain.disabled = true;
  explain.textContent = "Explain";
  const run = $('[data-role="run"]', card);
  run.disabled = !info.ready;
}

function setBusy(card, busy, tier) {
  card.classList.toggle("is-busy", busy);
  card.setAttribute("aria-busy", String(busy));
  if (busy) {
    card.dataset.state = "running";
    $('[data-role="badge"]', card).textContent = "running…";
  } else if (card.dataset.state === "running") {
    card.dataset.state = "idle";
    $('[data-role="badge"]', card).textContent = tierInfo(tier).ready ? "ready" : "not installed";
  }
  $('[data-role="run"]', card).disabled = busy || !tierInfo(tier).ready;
  $('[data-role="explain"]', card).disabled = busy || !state.results.has(tier);
}

function renderResult(tier, result, roundTrip) {
  const card = cardFor(tier);
  if (!card) return;
  const label = $('[data-role="label"]', card);
  label.textContent = result.label;
  label.classList.remove("is-empty");
  $('[data-role="conf"]', card).textContent = pct(result.confidence);
  $('[data-role="server"]', card).textContent = fmtMs(result.latency_ms);
  $('[data-role="rtt"]', card).textContent = fmtMs(roundTrip);

  const served = result.tier;
  const fallback = served !== tier;
  const badge = $('[data-role="badge"]', card);
  badge.textContent = fallback ? `via ${served}` : "done";
  badge.title = fallback
    ? `Requested ${tier}, but the service answered from the ${served} desk.`
    : `Answered by the ${served} desk.`;

  const bars = $('[data-role="bars"]', card);
  const scores = Array.isArray(result.scores) ? result.scores : [];
  bars.innerHTML = scores
    .map(
      (score, index) => `
      <div class="bar${index === 0 ? " is-best" : ""}" data-label="${esc(score.label)}">
        <div class="bar-top">
          <span class="bar-label">${esc(score.label)}</span>
          <span class="bar-value mono">${pct(score.probability)}</span>
        </div>
        <div class="bar-track"><span class="bar-fill" data-width="${pct(score.probability)}"></span></div>
      </div>`,
    )
    .join("");
  requestAnimationFrame(() => {
    for (const fill of $$(".bar-fill", bars)) {
      fill.style.width = fill.dataset.width || "0%";
    }
  });

  card.dataset.state = "done";
  $('[data-role="error"]', card).hidden = true;
  const explain = $('[data-role="explain"]', card);
  explain.disabled = false;
  explain.textContent = "Explain";
}

function showError(tier, message) {
  const card = cardFor(tier);
  if (!card) return;
  card.dataset.state = "error";
  $('[data-role="badge"]', card).textContent = "error";
  const box = $('[data-role="error"]', card);
  box.textContent = String(message).slice(0, 240);
  box.hidden = false;
}

function resetResults() {
  state.results.clear();
  state.explanations.clear();
  for (const tier of ORDER) resetCard(tier);
  updateSummary();
}

/* -------------------------------------------------------------- predict */

function text() {
  return el.copy.value.trim();
}

function requireText() {
  const value = text();
  if (value) return value;
  clearInline();
  showInline("Paste a headline or choose an example first.");
  el.copy.focus();
  return null;
}

async function runTier(tier, options = {}) {
  const value = requireText();
  if (!value) return;
  await ensureReady();
  const card = cardFor(tier);
  if (!card || state.running.has(tier)) return;

  const run = (state.sequence.get(tier) || 0) + 1;
  const epoch = state.epoch;
  state.sequence.set(tier, run);
  state.running.add(tier);
  setBusy(card, true, tier);
  $('[data-role="error"]', card).hidden = true;
  $('[data-role="explain-panel"]', card).hidden = true;
  clearInline();

  const started = performance.now();
  try {
    const payload = { text: value };
    if (state.allowOverride) payload.tier = tier;
    const result = await request("/v1/predict", {
      method: "POST",
      body: payload,
      timeout: PREDICT_TIMEOUT,
    });
    if (state.sequence.get(tier) !== run || state.epoch !== epoch) return;
    state.results.set(tier, result);
    state.calls += 1;
    renderResult(tier, result, performance.now() - started);
    if (options.explain) await runExplain(tier);
  } catch (error) {
    if (state.sequence.get(tier) === run && state.epoch === epoch) {
      state.results.delete(tier);
      showError(tier, error.message);
    }
  } finally {
    if (state.sequence.get(tier) === run) {
      state.running.delete(tier);
      setBusy(card, false, tier);
    }
    updateSummary();
  }
}

async function runAll() {
  if (state.running.size) return;
  const value = requireText();
  if (!value) return;
  await ensureReady();
  const ready = state.tiers.filter((info) => info.ready).map((info) => info.tier);
  if (!ready.length) {
    showInline("No desk is installed, so there is nothing to run. Train a tier first.");
    return;
  }
  clearInline();
  resetResults();
  el.runAll.disabled = true;
  const label = $(".btn-label", el.runAll);
  const original = label.textContent;
  label.textContent = `Running ${ready.length} desks…`;
  try {
    await Promise.allSettled(ready.map((tier) => runTier(tier)));
  } finally {
    label.textContent = original;
    el.runAll.disabled = false;
  }
}

/* ---------------------------------------------------------------- explain */

function explainValue(token) {
  if (typeof token.contribution === "number") return token.contribution;
  if (typeof token.importance === "number") return token.importance;
  if (typeof token.score === "number") return token.score;
  return 0;
}

function renderExplanation(tier, result) {
  const card = cardFor(tier);
  const panel = $('[data-role="explain-panel"]', card);
  const method = $('[data-role="method"]', card);
  const tokens = $('[data-role="tokens"]', card);
  panel.hidden = false;

  const all = Array.isArray(result.tokens) ? result.tokens : [];
  const ranked = all
    .slice()
    .sort((a, b) => Math.abs(explainValue(b)) - Math.abs(explainValue(a)))
    .slice(0, 12);
  const max = Math.max(...ranked.map((token) => Math.abs(explainValue(token))), 1e-9);

  method.textContent = result.explanation_method || "attribution";
  method.title = result.explanation_context || "";
  if (!ranked.length) {
    tokens.innerHTML = '<span class="hint mono">no token attribution returned</span>';
    return;
  }
  tokens.innerHTML = ranked
    .map((token) => {
      const value = explainValue(token);
      const negative = typeof token.contribution === "number" && token.contribution < 0;
      const width = (Math.abs(value) / max).toFixed(3);
      const title = `${token.token}: ${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
      return `<span class="token-pill${negative ? " is-negative" : ""}" style="--w:${width}" title="${esc(title)}">${esc(token.token)}</span>`;
    })
    .join("");
}

async function runExplain(tier) {
  const value = text();
  const card = cardFor(tier);
  if (!value || !card || !state.results.has(tier)) return;
  const button = $('[data-role="explain"]', card);
  const previous = button.textContent;
  button.disabled = true;
  button.textContent = "Explaining…";
  clearInline();
  try {
    const payload = { text: value };
    if (state.allowOverride) payload.tier = tier;
    const result = await request("/v1/explain", {
      method: "POST",
      body: payload,
      timeout: EXPLAIN_TIMEOUT,
    });
    state.explanations.set(tier, result);
    renderExplanation(tier, result);
    button.textContent = "Refresh";
  } catch (error) {
    const panel = $('[data-role="explain-panel"]', card);
    panel.hidden = false;
    $('[data-role="method"]', card).textContent = "explanation unavailable";
    $('[data-role="tokens"]', card).innerHTML = `<span class="hint mono">${esc(error.message)}</span>`;
    button.textContent = previous;
  } finally {
    button.disabled = false;
  }
}

function toggleExplain(tier) {
  const card = cardFor(tier);
  const panel = $('[data-role="explain-panel"]', card);
  if (!panel.hidden && state.explanations.has(tier)) {
    panel.hidden = true;
    return;
  }
  if (state.explanations.has(tier)) {
    panel.hidden = false;
    return;
  }
  runExplain(tier);
}

/* ---------------------------------------------------------------- summary */

function updateSummary() {
  const entries = [...state.results.entries()];
  if (!entries.length) {
    el.agreement.textContent = "—";
    el.agreement.classList.remove("is-split");
    el.fastest.textContent = "—";
    el.spread.textContent = "—";
    el.calls.textContent = String(state.calls);
    return;
  }

  const labels = entries.map(([, result]) => result.label);
  const unique = [...new Set(labels)];
  if (unique.length === 1) {
    const count = labels.length;
    if (count === 1) {
      el.agreement.textContent = `${cap(entries[0][0])} answered ${unique[0]}`;
    } else {
      const subject = count === 2 ? "Both desks" : `All ${count} desks`;
      el.agreement.textContent = `${subject} agree on ${unique[0]}`;
    }
    el.agreement.classList.remove("is-split");
  } else {
    const tally = unique
      .map((label) => `${label} × ${labels.filter((item) => item === label).length}`)
      .join(" · ");
    el.agreement.textContent = `Split — ${tally}`;
    el.agreement.classList.add("is-split");
  }

  const fastest = entries.reduce((best, current) =>
    current[1].latency_ms <= best[1].latency_ms ? current : best,
  );
  el.fastest.textContent = `${cap(fastest[0])} · ${fmtMs(fastest[1].latency_ms)}`;

  if (entries.length < 2) {
    el.spread.textContent = "—";
  } else {
    const confidences = entries.map(([, result]) => Number(result.confidence));
    const spread = (Math.max(...confidences) - Math.min(...confidences)) * 100;
    el.spread.textContent = `${spread.toFixed(1)} pts`;
  }
  el.calls.textContent = String(state.calls);
}

/* ------------------------------------------------------------------- boot */

let readyPromise = null;

function ensureReady() {
  if (!readyPromise) {
    readyPromise = (async () => {
      try {
        const [tiers, model, health] = await Promise.all([
          request("/v1/tiers"),
          request("/v1/model"),
          request("/healthz"),
        ]);
        state.tiers = Array.isArray(tiers) ? tiers : [];
        state.allowOverride = model?.allow_tier_override !== false;
        state.defaultTier = model?.default_tier || null;
        renderCards(state.tiers);
        setStatus(true, state.defaultTier || "");
        if (health?.version) {
          el.version.textContent = `v${health.version}`;
          el.backend.textContent = `api: ${health.version}`;
        }
        if (!state.allowOverride) {
          const note = document.createElement("p");
          note.className = "hint mono";
          note.id = "lab-override-note";
          note.textContent =
            `Tier override is disabled on this service; every desk will be answered by the house default (${state.defaultTier}).`;
          el.runAll.closest(".lab-controls")?.after(note);
        }
      } catch (error) {
        setStatus(false);
        el.grid.setAttribute("aria-busy", "false");
        el.grid.innerHTML = `<p class="lab-loading mono">${esc(error.message)}</p>`;
        showInline(`Could not load the desks: ${error.message}`);
        state.tiers = [];
      }
    })();
  }
  return readyPromise;
}

function fillSamples() {
  el.samples.innerHTML = SAMPLES.map(
    ([name, copy]) =>
      `<button class="sample-chip" type="button" data-text="${esc(copy)}" title="${esc(copy)}">${esc(name)}</button>`,
  ).join("");
  el.samples.addEventListener("click", (event) => {
    const chip = event.target.closest(".sample-chip");
    if (!chip) return;
    el.copy.value = chip.dataset.text || "";
    el.copy.dispatchEvent(new Event("input", { bubbles: true }));
    runAll();
  });
}

function wire() {
  el.copy.addEventListener("input", () => {
    state.epoch += 1;
    syncCount();
    clearInline();
    if (state.results.size || state.explanations.size || state.running.size) resetResults();
  });
  el.copy.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      runAll();
    }
  });
  el.runAll.addEventListener("click", runAll);
  el.clear.addEventListener("click", () => {
    el.copy.value = "";
    el.copy.dispatchEvent(new Event("input", { bubbles: true }));
    resetResults();
    clearInline();
    el.copy.focus();
  });
  window.addEventListener("scroll", syncNav, { passive: true });
}

function animateIn() {
  if (!window.gsap || reduceMotion) return;
  window.gsap.from([".lab-hero > *", ".lab-bench"], {
    y: 26,
    opacity: 0,
    duration: 0.8,
    stagger: 0.07,
    ease: "power3.out",
  });
}

function boot() {
  fillSamples();
  syncCount();
  syncNav();
  wire();
  animateIn();
  ensureReady().then(() => {
    if (state.allowOverride && state.defaultTier) {
      setStatus(true, `${state.defaultTier} default`);
    }
  });
}

boot();
