/* innards view: talks to the JSON API only. Vanilla JS + the vendored Plotly. Untrusted text goes in via textContent. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ state and helpers
  const S = {
    info: null,
    state: null,
    runs: [],
    lineup: { pairs: [], models: [] },
    sets: [],
    viewing: null, // {run, conversation} shown on the Conversation tab
    viewingTurns: [],
    live: null, // {run, conversation} the loaded model is writing to
    liveTurns: -1,
    job: null, // streaming job text
    streaming: false,
    report: null,
    docId: null,
    docName: null,
    colors: new Map(),
    turnCharts: [],
    analysisKey: "",
    pollTimer: null,
    lastRunsFetch: 0,
  };
  const $ = (sel) => document.querySelector(sel);
  const nn = (v) => v !== null && v !== undefined && !Number.isNaN(v);
  const short = (id) => String(id || "").split("/").pop();
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); return node; };
  function fmtBytes(n) {
    if (!nn(n)) return "n/a";
    let v = Number(n);
    for (const unit of ["B", "KB", "MB", "GB", "TB"]) {
      if (Math.abs(v) < 1000 || unit === "TB") return unit === "B" ? `${Math.round(v)} B` : `${v.toFixed(v < 10 ? 2 : 1)} ${unit}`;
      v /= 1000;
    }
    return `${v.toFixed(1)} TB`;
  }
  const fmtInt = (n) => (nn(n) ? Math.round(Number(n)).toLocaleString() : "n/a");
  const fmtNum = (n, d = 1) => (nn(n) ? Number(n).toFixed(d) : "n/a");
  function fmtUnit(v, unit) {
    if (!nn(v)) return "n/a";
    if (unit === "bytes") return fmtBytes(v);
    if (unit === "tokens" || unit === "count") return fmtInt(v);
    if (unit === "tok/s") return `${fmtNum(v, 1)} tok/s`;
    if (unit === "ms") return `${fmtInt(v)} ms`;
    if (unit === "%") return `${Number(v) < 1 ? fmtNum(v, 2) : fmtNum(v, 1)}%`;
    if (unit === "ratio") return fmtNum(v, 3);
    return Math.abs(v) >= 100 ? fmtInt(v) : fmtNum(v, 3);
  }
  function storageGet(k) { try { return window.localStorage.getItem(k); } catch { return null; } }
  function storageSet(k, v) { try { window.localStorage.setItem(k, v); } catch { /* private mode */ } }
  function token() {
    const m = window.location.hash.match(/token=([^&]+)/);
    if (m) { try { window.sessionStorage.setItem("innards-token", m[1]); } catch { /* ignore */ } return m[1]; }
    try { return window.sessionStorage.getItem("innards-token"); } catch { return null; }
  }
  async function api(path, opts = {}) {
    const headers = { Accept: "application/json", ...(opts.headers || {}) };
    const t = token();
    if (t) headers.Authorization = `Bearer ${t}`;
    if (opts.method === "POST") { headers["X-Innards"] = "1"; headers["Content-Type"] = "application/json"; }
    const res = await fetch(path, { ...opts, headers });
    if (!res.ok) {
      let detail = res.statusText;
      try { const body = await res.json(); detail = body.detail !== undefined ? body.detail : detail; } catch { /* not json */ }
      const err = new Error(typeof detail === "string" ? detail : (detail.message || JSON.stringify(detail)));
      err.status = res.status; err.detail = detail;
      throw err;
    }
    return opts.raw ? res : res.json();
  }
  function banner(text, kind) {
    const node = $("#banner");
    if (!text) { node.hidden = true; return; }
    node.hidden = false; node.className = `banner${kind === "info" ? " info" : ""}`; node.textContent = text;
  }

  // ------------------------------------------------------------------ fixed model colors
  const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"];
  function colorFor(modelId) {
    if (!modelId) return SLOTS[0];
    if (S.colors.has(modelId)) return S.colors.get(modelId);
    const entry = S.lineup.models.find((m) => m.id === modelId);
    let color = null;
    if (entry && entry.role === "newer") color = SLOTS[0];
    else if (entry && entry.role === "previous") color = SLOTS[1];
    if (!color || [...S.colors.values()].includes(color) && !entry) {
      const used = new Set(S.colors.values());
      color = SLOTS.find((c) => !used.has(c)) || SLOTS[S.colors.size % SLOTS.length];
    }
    S.colors.set(modelId, color);
    return color;
  }

  // ------------------------------------------------------------------ header
  function currentWatch() { return S.state && S.state.watch; }
  function runnerState() { return S.state && S.state.runner; }
  function renderHeader() {
    const runner = runnerState();
    const watch = currentWatch();
    const live = $("#live");
    const text = $("#live-text");
    const generating = (runner && runner.state === "generating") || (watch && watch.live && watch.live.generating) || S.streaming;
    const turn = watch ? (watch.live && watch.live.turn) || watch.turns : null;
    live.className = "live";
    if (generating) { live.classList.add("on"); text.textContent = `Live · turn ${turn || 1}`; }
    else if (S.pending) { live.classList.add("loading"); text.textContent = "Checking…"; }
    else if (runner && ["loading", "validating"].includes(runner.state)) { live.classList.add("loading"); text.textContent = runner.state === "loading" ? `Loading ${short(runner.request && runner.request.model)}…` : "Checking…"; }
    else if (runner && runner.state === "error") { live.classList.add("error"); text.textContent = "Error"; }
    else if (watch) { live.classList.add("ready"); text.textContent = `Ready · ${watch.turns} turn${watch.turns === 1 ? "" : "s"}`; }
    else { live.classList.add("idle"); text.textContent = runner ? "No model loaded" : "Idle"; }
    // gauge
    const mem = watch && watch.memory;
    const fill = $("#gauge-fill");
    const meter = $("#gauge-meter");
    if (mem && nn(mem.budget_bytes)) {
      const used = nn(mem.allocated_bytes) ? mem.allocated_bytes : mem.weights_bytes;
      const pct = Math.min(100, (used / mem.budget_bytes) * 100);
      fill.style.width = `${pct}%`;
      meter.className = `meter${pct > 90 ? " crit" : pct > 70 ? " warn" : ""}`;
      meter.setAttribute("aria-valuenow", String(Math.round(pct)));
      $("#gauge-value").textContent = `${(used / 1e9).toFixed(1)}/${Math.round(mem.budget_bytes / 1e9)} GB`;
      $("#gauge-label").textContent = String(mem.device || "").startsWith("cpu") ? "RAM" : String(mem.device || "").startsWith("mps") ? "MPS" : "GPU";
    } else {
      fill.style.width = "0"; $("#gauge-value").textContent = "n/a";
    }
    // load controls
    const busy = S.pending || (runner && ["loading", "validating", "generating"].includes(runner.state));
    $("#load-btn").disabled = !runner || busy;
    $("#load-btn").title = runner ? "" : "Loading needs a runner: start the view with innards-exp serve";
    $("#check-btn").disabled = busy;
    if (runner && runner.state === "error" && runner.error) banner(runner.error);
    if (runner && runner.report && runner.report !== S.report) { S.report = runner.report; renderChecks(runner.report); }
    const mode = runner ? `runner: ${runner.name || "attached"}` : watch ? "attached to your script" : "run folder";
    $("#mode-note").textContent = mode;
  }

  function renderFit(report) {
    const pill = $("#fit-pill");
    if (!report) { pill.hidden = true; return; }
    pill.hidden = false;
    const kind = report.status === "Blocked" ? "critical" : report.status === "Ready" ? "good" : "warning";
    const icon = kind === "critical" ? "✕" : kind === "good" ? "✓" : "!";
    pill.className = `pill ${kind}`;
    pill.textContent = `${icon} ${report.status}`;
  }

  function renderChecks(report) {
    const box = $("#checks");
    renderFit(report);
    if (!report) { box.hidden = true; return; }
    box.hidden = false;
    clear(box);
    const icons = { ok: "✓", warning: "!", blocked: "✕", skipped: "–" };
    box.append(el("h3", {}, `Checks for ${report.model}`, el("span", { class: `pill ${report.status === "Blocked" ? "critical" : report.status === "Ready" ? "good" : "warning"}`, text: report.status }),
      el("span", { class: "muted small", text: `${report.device || ""} · ${String(report.precision || "").toUpperCase()}` })));
    const list = el("ul");
    for (const c of report.checks || []) {
      list.append(el("li", { class: c.status }, el("span", { class: "icon", text: icons[c.status] || "?", title: c.status }),
        el("span", { text: c.title }), el("span", {}, c.message, c.fix && c.status !== "ok" ? el("span", { class: "fix", text: `Fix: ${c.fix}` }) : null)));
    }
    box.append(list);
  }

  // ------------------------------------------------------------------ sidebar
  function renderSidebar() {
    const watch = currentWatch();
    const last = watch && watch.last;
    const mon = clear($("#monitor"));
    const rows = [
      ["KV cache", last ? fmtBytes(last.kv.measured_bytes) : "–"],
      ["Context", last ? fmtInt(last.tokens.context_after) : "–"],
      ["Memory", watch && watch.memory ? fmtBytes(watch.memory.allocated_bytes ?? watch.memory.weights_bytes) : "–"],
      ["Timing", last && nn(last.timing.decode_tokens_per_s) ? `${fmtNum(last.timing.decode_tokens_per_s, 0)} t/s` : "–"],
    ];
    for (const [k, v] of rows) {
      mon.append(el("li", {}, el("span", { class: `status-dot${watch ? "" : " off"}` }), el("span", { class: "k", text: k }), el("span", { class: "v", text: v })));
    }
    const feats = clear($("#features"));
    const list = watch ? watch.features : null;
    if (!list) {
      feats.append(el("li", { class: "muted", text: runnerState() ? "Load a model to see its features" : "No model attached" }));
    } else {
      const order = { analyzed: 0, detected: 1, not_analyzed: 2, not_run: 3, absent: 4 };
      const entries = Object.entries(list).sort((a, b) => (order[a[1].status] - order[b[1].status]) || (a[1].light - b[1].light));
      const absent = entries.filter(([, f]) => f.status === "absent");
      for (const [id, f] of entries.filter(([, x]) => x.status !== "absent")) {
        const status = f.status;
        const dot = status === "analyzed" ? "status-dot" : status === "not_analyzed" ? "status-dot partial" : "status-dot off";
        const tag = status === "analyzed" ? (f.light ? "cache layout" : `${f.layers || ""} layers`.trim())
          : status === "absent" ? "not in this model" : status === "not_analyzed" ? "not analyzed yet"
            : status === "not_run" ? "not run by transformers" : "detected, probes off";
        feats.append(el("li", { class: status === "absent" ? "absent" : "", title: f.shows || f.note || "", "data-feature": id },
          el("span", { class: "dot-wrap" }, el("span", { class: dot })), el("span", {}, f.name, el("span", { class: "tag", text: tag }))));
      }
      if (absent.length) {
        feats.append(el("li", { class: "absent", title: absent.map(([, f]) => f.name).join(", ") },
          el("span", { class: "dot-wrap" }, el("span", { class: "status-dot off" })),
          el("span", {}, `${absent.length} more not in this model`, el("span", { class: "tag", text: absent.map(([, f]) => f.name.split(" (")[0]).join(" · ") }))));
      }
    }
    const health = clear($("#health"));
    const h = watch && watch.health;
    if (!h) { health.textContent = "No turns yet"; health.className = "health muted"; return; }
    health.className = "health";
    const over = h.overhead_pct;
    health.append(el("div", { class: "row" }, el("span", { text: "Overhead" }), el("span", { text: nn(over) ? fmtUnit(over, "%") : "–" })));
    health.append(el("div", { class: "row" }, el("span", { text: "Dropped events" }), el("span", { text: fmtInt(h.dropped) })));
    health.append(el("div", { class: "row" }, el("span", { text: "Errors" }), el("span", { text: fmtInt(h.errors) })));
    for (const p of h.probes || []) {
      health.append(el("div", { class: "probe" }, `${p.feature}: ${fmtUnit(p.overhead_pct, "%")} · ${p.hooks} hooks${p.errors ? ` · ${p.errors} errors` : ""}`));
    }
  }

  function convLabel(c) {
    return `${short(c.model)}${c.set ? ` · ${c.set}` : ""} · ${c.turns || 0} turn${c.turns === 1 ? "" : "s"}`;
  }
  function renderRuns() {
    const list = clear($("#runs"));
    if (!S.runs.length) { list.append(el("li", { class: "muted small", text: "No runs yet" })); return; }
    for (const run of S.runs.slice(0, 12)) {
      const item = el("li", {}, el("div", { class: "run-name", text: run.name }));
      for (const c of run.conversations || []) {
        const key = `${run.name}/${c.id}`;
        const isCurrent = S.viewing && `${S.viewing.run}/${S.viewing.conversation}` === key;
        item.append(el("button", { class: isCurrent ? "current" : "", title: `${c.id}${c.status ? ` (${c.status})` : ""}`,
          onclick: () => { openConversation(run.name, c.id); } },
        el("span", { class: "swatch", style: `background:${colorFor(c.model)}` }), convLabel(c)));
      }
      list.append(item);
    }
  }

  // ------------------------------------------------------------------ polling
  function schedulePoll(ms) { clearTimeout(S.pollTimer); S.pollTimer = setTimeout(poll, ms); }
  async function poll() {
    let busy = false;
    try {
      S.state = await api("/api/state");
      const runner = runnerState();
      const watch = currentWatch();
      busy = S.streaming || (runner && ["loading", "validating", "generating"].includes(runner.state)) || (watch && watch.live && watch.live.generating);
      const liveKey = watch ? { run: watch.run, conversation: watch.conversation } : null;
      const changed = liveKey && (!S.live || S.live.conversation !== liveKey.conversation);
      if (changed) { S.live = liveKey; S.liveTurns = -1; }
      if (watch && watch.turns !== S.liveTurns) {
        S.liveTurns = watch.turns;
        await refreshRuns();
        if (!S.viewing || changed || (S.viewing.conversation === watch.conversation)) {
          if (watch.turns > 0 || changed) await openConversation(watch.run, watch.conversation, { quiet: true });
        }
        if ($("#an-conv").value === `${watch.run}/${watch.conversation}`) renderAnalysis(true);
      } else if (Date.now() - S.lastRunsFetch > 15000) {
        await refreshRuns();
      }
      renderHeader();
      renderSidebar();
      renderComposer();
      if (watch && watch.live && watch.live.generating && !S.streaming) renderMessages();
    } catch (err) {
      banner(`Lost contact with the view server: ${err.message}`);
    }
    schedulePoll(busy ? 800 : 3000);
  }
  async function refreshRuns() {
    S.runs = await api("/api/runs");
    S.lastRunsFetch = Date.now();
    renderRuns();
    fillViewing();
    fillAnalysisSelects();
  }

  // ------------------------------------------------------------------ model picker, checks, load
  function fillModels() {
    const list = clear($("#model-list"));
    for (const m of S.lineup.models) {
      const pair = S.lineup.pairs.find((p) => p.id === m.pair);
      list.append(el("option", { value: m.id, label: `${pair ? pair.feature : ""} · ${m.role || ""} · ~${m.weights_gb} GB ${String(m.precision).toUpperCase()}` }));
    }
    for (const d of S.lineup.downloaded || []) list.append(el("option", { value: d.id, label: "downloaded" }));
    const saved = storageGet("innards-model");
    $("#model-input").value = saved || (S.lineup.pairs[0] ? S.lineup.pairs[0].newer : "");
  }
  function pickedSet() { return $("#set-select").value || null; }
  async function checkModel() {
    const model = $("#model-input").value.trim();
    if (!model) return;
    storageSet("innards-model", model);
    banner("");
    $("#check-btn").disabled = true;
    try {
      const report = await api("/api/validate", { method: "POST", body: JSON.stringify({ model, precision: $("#precision").value, set: pickedSet(), document: S.docId }) });
      S.report = report; renderChecks(report);
    } catch (err) { banner(err.message); } finally { $("#check-btn").disabled = false; }
  }
  async function loadModel(ev) {
    if (ev) ev.preventDefault();
    const model = $("#model-input").value.trim();
    if (!model) return;
    storageSet("innards-model", model);
    banner("");
    S.pending = true; renderHeader(); renderComposer();
    try {
      const status = await api("/api/load", { method: "POST", body: JSON.stringify({ model, precision: $("#precision").value, set: pickedSet(), document: S.docId }) });
      if (S.state) S.state.runner = status;
      S.report = status.report; renderChecks(status.report);
      showTab("conversation");
    } catch (err) {
      if (err.detail && err.detail.report) { S.report = err.detail.report; renderChecks(err.detail.report); banner(err.detail.message); }
      else banner(err.message);
    } finally {
      S.pending = false; renderHeader(); renderComposer();
    }
    schedulePoll(300);
  }

  // ------------------------------------------------------------------ conversation tab
  function fillSets() {
    const sel = clear($("#set-select"));
    sel.append(el("option", { value: "", text: "Free chat" }));
    for (const s of S.sets) sel.append(el("option", { value: s.name, text: s.title }));
    sel.value = storageGet("innards-set") || "with_context";
    onSetChange();
  }
  function onSetChange() {
    const name = pickedSet();
    storageSet("innards-set", name || "");
    const s = S.sets.find((x) => x.name === name);
    const needsDoc = s && s.needs_document;
    $("#doc-btn").hidden = !needsDoc;
    $("#doc-note").textContent = needsDoc ? (S.docName ? `${S.docName}` : "no document yet") : "";
    $("#set-note").textContent = s ? `${s.contents}. Shows: ${s.shows}. ${s.turns} turns${s.needs_document ? "" : `, about ${fmtInt(s.estimated_tokens)} tokens of input`}.` : "Type your own messages.";
    renderComposer();
  }
  async function onDocument(ev) {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    if (file.size > 20e6) { banner("Documents are limited to 20 MB"); return; }
    const buf = new Uint8Array(await file.arrayBuffer());
    let bin = "";
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    try {
      const doc = await api("/api/documents", { method: "POST", body: JSON.stringify({ name: file.name, data_base64: btoa(bin) }) });
      S.docId = doc.id; S.docName = doc.name;
      $("#doc-note").textContent = `${doc.name}: ${fmtInt(doc.characters)} characters, ${doc.exact ? "" : "about "}${fmtInt(doc.tokens)} tokens`;
      banner("");
    } catch (err) { banner(err.message); }
    ev.target.value = "";
  }
  function canRun() { const r = runnerState(); return !!(r && r.state === "ready" && !S.streaming && !S.pending); }
  function renderComposer() {
    const r = runnerState();
    const box = $("#message");
    const ok = canRun();
    box.disabled = !ok;
    $("#send-btn").disabled = !ok;
    $("#run-set").disabled = !ok || !pickedSet() || (S.sets.find((s) => s.name === pickedSet()) || {}).needs_document && !S.docId;
    $("#new-conv").disabled = !ok;
    $("#stop-btn").hidden = !S.streaming;
    const status = $("#composer-status");
    if (!r) { box.placeholder = "This view monitors your script: turns appear as your code calls generate()"; status.textContent = ""; }
    else if (r.state === "ready") { box.placeholder = "Message… (Enter to send, Shift+Enter for a new line)"; status.textContent = r.model ? `${short(r.model.id)} · ${String(r.model.precision || "").toUpperCase()} · ${r.model.device || ""}` : ""; }
    else if (r.state === "generating") { box.placeholder = "Generating…"; }
    else { box.placeholder = "Load a model to start"; status.textContent = r.state === "error" ? "" : r.state; }
  }
  function fillViewing() {
    const sel = clear($("#viewing"));
    const convs = S.runs.flatMap((r) => (r.conversations || []).map((c) => ({ run: r.name, ...c })));
    $("#viewing-wrap").hidden = !convs.length;
    for (const c of convs) sel.append(el("option", { value: `${c.run}/${c.id}`, text: `${c.run} · ${convLabel(c)}` }));
    if (S.viewing) sel.value = `${S.viewing.run}/${S.viewing.conversation}`;
  }
  async function openConversation(run, conversation, opts = {}) {
    S.viewing = { run, conversation };
    try {
      S.viewingTurns = await api(`/api/runs/${encodeURIComponent(run)}/${encodeURIComponent(conversation)}/turns`);
    } catch (err) { S.viewingTurns = []; if (!opts.quiet) banner(err.message); }
    renderMessages();
    renderRuns();
    $("#viewing").value = `${run}/${conversation}`;
    if (!opts.quiet && !$("#tab-analysis").hidden) { $("#an-conv").value = `${run}/${conversation}`; renderAnalysis(); }
  }
  function turnMeta(t) {
    const kv = t.kv || {}; const tok = t.tokens || {}; const tm = t.timing || {};
    const parts = [
      ["turn", fmtInt(t.turn)],
      ["reused", fmtInt(tok.cached)],
      ["prefilled", fmtInt(tok.recomputed)],
      ["generated", fmtInt(tok.generated)],
      ["KV", `${nn(kv.growth_bytes) && kv.growth_bytes >= 0 ? "+" : ""}${fmtBytes(kv.growth_bytes)} → ${fmtBytes(kv.measured_bytes)}`],
      ["TTFT", nn(tm.ttft_ms) ? `${fmtInt(tm.ttft_ms)} ms` : "n/a"],
      ["", nn(tm.decode_tokens_per_s) ? `${fmtNum(tm.decode_tokens_per_s, 1)} t/s` : ""],
    ];
    return el("div", { class: "meta" }, parts.filter((p) => p[1]).map(([k, v]) => el("span", {}, k ? `${k} ` : "", el("b", { text: v }))));
  }
  function bubble(role, text, extra) {
    const body = el("div", { class: "body", text: text || "" });
    const wrap = el("div", {}, body);
    if ((text || "").length > 900) {
      body.classList.add("long");
      wrap.append(el("button", { class: "expand", type: "button", onclick: (e) => { body.classList.toggle("long"); e.target.textContent = body.classList.contains("long") ? "Show all" : "Show less"; }, text: "Show all" }));
    }
    if (extra) wrap.append(extra);
    return el("div", { class: `msg ${role}` }, el("div", { class: "role", text: role === "user" ? "user" : "model" }), wrap);
  }
  function renderMessages() {
    const box = clear($("#messages"));
    const turns = S.viewingTurns || [];
    const watch = currentWatch();
    if (!turns.length && !S.job && !(watch && watch.live && watch.live.generating)) {
      box.append(el("div", { class: "empty", text: runnerState() ? "Load a model, pick a conversation set and press Run set, or type a message." : "No turns yet. They appear here as generate() is called." }));
      return;
    }
    for (const t of turns) {
      const step = t.set && t.set.step ? ` (${t.set.name} ${t.set.step}/${t.set.of})` : "";
      box.append(bubble("user", (t.message || "(no message text)") + step));
      box.append(bubble("model", t.output && t.output.text !== null ? t.output.text : "(no reply text)", turnMeta(t)));
    }
    if (S.job) {
      if (S.job.message) box.append(bubble("user", S.job.message));
      const b = bubble("model", S.job.text || "", el("div", { class: "meta" }, el("span", {}, "generating ", el("b", { text: fmtInt(S.job.generated || 0) }), " tokens")));
      b.classList.add("streaming");
      box.append(b);
    } else if (watch && watch.live && watch.live.generating && S.viewing && S.viewing.conversation === watch.conversation) {
      const b = bubble("model", "", el("div", { class: "meta" }, el("span", {}, "turn ", el("b", { text: fmtInt(watch.live.turn) }), " · generated ", el("b", { text: fmtInt(watch.live.generated) }), " tokens")));
      b.classList.add("streaming");
      box.append(b);
    }
    box.scrollTop = box.scrollHeight;
  }

  // server-sent events over a POST
  async function streamPost(path, body) {
    const t = token();
    const headers = { "Content-Type": "application/json", "X-Innards": "1", Accept: "text/event-stream" };
    if (t) headers.Authorization = `Bearer ${t}`;
    const res = await fetch(path, { method: "POST", headers, body: JSON.stringify(body) });
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail; } catch { /* ignore */ }
      throw Object.assign(new Error(typeof detail === "string" ? detail : (detail && detail.message) || "request failed"), { detail });
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let event = "message"; const data = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data.push(line.slice(6));
        }
        if (data.length) handleEvent(event, JSON.parse(data.join("\n")));
      }
    }
  }
  function handleEvent(event, data) {
    if (event === "turn_start") { S.job = { message: data.message, text: "", generated: 0, step: data.step, of: data.of }; $("#composer-status").textContent = data.step ? `Set step ${data.step} of ${data.of}` : ""; renderMessages(); }
    else if (event === "token") { if (!S.job) S.job = { text: "", generated: 0 }; S.job.text += data.text; S.job.generated = data.generated || S.job.generated + 1; throttledRender(); }
    else if (event === "turn") {
      S.job = null;
      const rec = data.record;
      if (rec && S.viewing && rec.conversation === S.viewing.conversation) { S.viewingTurns = S.viewingTurns.filter((t) => t.turn !== rec.turn).concat([rec]); }
      else if (rec) { S.viewing = { run: rec.run, conversation: rec.conversation }; S.viewingTurns = [rec]; }
      renderMessages();
    } else if (event === "error") { S.job = null; banner(data.message || "The turn failed"); renderMessages(); }
    else if (event === "blocked") { S.job = null; if (data.report) { S.report = data.report; renderChecks(data.report); } banner(data.message); }
  }
  let renderPending = false;
  function throttledRender() { if (renderPending) return; renderPending = true; requestAnimationFrame(() => { renderPending = false; renderMessages(); }); }
  async function runJob(path, body) {
    banner("");
    S.streaming = true; renderComposer(); renderHeader();
    schedulePoll(500);
    try { await streamPost(path, body); } catch (err) {
      if (err.detail && err.detail.report) { S.report = err.detail.report; renderChecks(err.detail.report); }
      banner(err.message);
    } finally {
      S.streaming = false; S.job = null;
      await refreshRuns();
      if (S.viewing) await openConversation(S.viewing.run, S.viewing.conversation, { quiet: true });
      renderComposer(); schedulePoll(200);
    }
  }
  async function sendMessage(ev) {
    if (ev) ev.preventDefault();
    const box = $("#message");
    const text = box.value.trim();
    if (!text || !canRun()) return;
    box.value = "";
    await runJob("/api/chat", { message: text });
  }
  async function runSet() {
    const name = pickedSet();
    if (!name || !canRun()) return;
    await runJob("/api/set/run", { set: name, document: S.docId });
  }
  async function newConversation() {
    try { await api("/api/conversation", { method: "POST", body: JSON.stringify({ set: pickedSet() }) }); S.viewingTurns = []; renderMessages(); } catch (err) { banner(err.message); }
    schedulePoll(200);
  }
  async function stopTurn() { try { await api("/api/stop", { method: "POST", body: "{}" }); } catch (err) { banner(err.message); } }

  // ------------------------------------------------------------------ analysis tab
  function fillAnalysisSelects() {
    const conv = $("#an-conv");
    const cmp = $("#an-compare");
    // The live conversation wins until the viewer picks one in this page; then their pick stays.
    const live = S.live ? `${S.live.run}/${S.live.conversation}` : "";
    const keep = S.anPicked || live || storageGet("innards-an-conv") || "";
    const keepCmp = cmp.value || storageGet("innards-an-compare") || "";
    clear(conv); clear(cmp);
    cmp.append(el("option", { value: "", text: "Nothing (this conversation only)" }));
    for (const run of S.runs) {
      const group = el("optgroup", { label: run.name });
      const group2 = el("optgroup", { label: run.name });
      for (const c of run.conversations || []) {
        group.append(el("option", { value: `${run.name}/${c.id}`, text: convLabel(c) }));
        group2.append(el("option", { value: `${run.name}/${c.id}`, text: convLabel(c) }));
      }
      conv.append(group); cmp.append(group2);
    }
    const predicted = el("optgroup", { label: "Lineup, predicted from config (not run)" });
    for (const m of S.lineup.models) predicted.append(el("option", { value: `predicted:${m.id}`, text: `${short(m.id)} (predicted)` }));
    conv.append(predicted);
    const values = [...conv.options].map((o) => o.value);
    const measured = values.find((v) => v && !v.startsWith("predicted:"));
    const before = conv.dataset.shown || "";
    conv.value = values.includes(keep) ? keep : (measured || values[0] || "");
    conv.dataset.shown = conv.value;
    const values2 = [...cmp.options].map((o) => o.value);
    cmp.value = values2.includes(keepCmp) && keepCmp !== conv.value ? keepCmp : "";
    if (before !== conv.value && (!cmp.value || before)) { cmp.value = ""; suggestCompare(); }
  }
  function suggestCompare() {
    // default: the other model of the pair, same set, latest conversation
    const [run, id] = $("#an-conv").value.split("/");
    const r = S.runs.find((x) => x.name === run);
    const c = r && (r.conversations || []).find((x) => x.id === id);
    if (!c) return;
    const entry = S.lineup.models.find((m) => m.id === c.model);
    if (!entry || !entry.partner) return;
    for (const run2 of S.runs) {
      const hit = (run2.conversations || []).slice().reverse().find((x) => x.model === entry.partner && (!c.set || x.set === c.set));
      if (hit) { $("#an-compare").value = `${run2.name}/${hit.id}`; return; }
    }
  }
  async function renderAnalysis(quiet) {
    const value = $("#an-conv").value;
    storageSet("innards-an-conv", value);
    storageSet("innards-an-compare", $("#an-compare").value);
    const head = $("#an-head"); const tiles = $("#tiles"); const grid = $("#an-grid");
    if (!value) { clear(head).append(el("p", { class: "muted", text: "No conversations yet. Run a conversation set, or pick a lineup model to see its predicted view." })); clear(tiles); clear(grid); return; }
    if (value.startsWith("predicted:")) { await renderPredicted(value.slice(10)); return; }
    const [run, conversation] = value.split("/");
    const params = new URLSearchParams();
    if ($("#an-compare").value && $("#an-compare").value !== value) params.set("compare", $("#an-compare").value);
    if ($("#an-first").value) params.set("first", $("#an-first").value);
    if ($("#an-last").value) params.set("last", $("#an-last").value);
    let data;
    try {
      grid.style.opacity = "0.55";
      data = await api(`/api/runs/${encodeURIComponent(run)}/${encodeURIComponent(conversation)}/analysis?${params}`);
    } catch (err) { if (!quiet) banner(err.message); grid.style.opacity = ""; return; } finally { grid.style.opacity = ""; }
    const key = JSON.stringify([value, params.toString(), data.turns]);
    if (quiet && key === S.analysisKey) return;
    S.analysisKey = key;
    colorFor(data.model.id);
    clear(head).append(el("h2", {}, el("span", { class: "swatch-key", style: `background:${colorFor(data.model.id)}` }), data.model.id),
      el("p", { text: `${data.model.architecture || ""}${data.set ? ` · set: ${data.set}` : ""} · ${data.turns} turns · ${data.status || ""} · run ${run}` }));
    renderTiles(data.tiles || []);
    S.turnCharts = [];
    clear(grid);
    if (data.changed) grid.append(changedCard(data.changed));
    for (const s of data.sections || []) grid.append(sectionCard(s, s.kind === "light" || s.kind === "compare"));
    if (data.compare) grid.append(compareCard(data.compare));
    if (!data.turns) grid.append(el("div", { class: "card wide" }, el("div", { class: "card-body muted", text: "This conversation has no turns yet." })));
  }

  function spark(values, unit) {
    const pts = values.map((v, i) => [i, v]).filter((p) => nn(p[1]));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 100 28"); svg.setAttribute("preserveAspectRatio", "none");
    if (pts.length < 2) return svg;
    const n = Math.max(values.length - 1, 1);
    const ys = pts.map((p) => p[1]); const lo = Math.min(...ys); const hi = Math.max(...ys);
    const y = (v) => (hi === lo ? 14 : 25 - ((v - lo) / (hi - lo)) * 22);
    const line = document.createElementNS(svg.namespaceURI, "polyline");
    line.setAttribute("points", pts.map(([i, v]) => `${(i / n) * 100},${y(v)}`).join(" "));
    line.setAttribute("fill", "none"); line.setAttribute("stroke", "#b8bec8"); line.setAttribute("stroke-width", "1.5");
    line.setAttribute("vector-effect", "non-scaling-stroke");
    svg.append(line);
    const [li, lv] = pts[pts.length - 1];
    const dot = document.createElementNS(svg.namespaceURI, "circle");
    dot.setAttribute("cx", String((li / n) * 100)); dot.setAttribute("cy", String(y(lv))); dot.setAttribute("r", "2.2");
    dot.setAttribute("fill", "#2a78d6"); dot.setAttribute("class", "spark-dot");
    svg.append(dot);
    svg.dataset.n = String(n); svg.dataset.lo = String(lo); svg.dataset.hi = String(hi);
    svg._values = values; svg._y = y; svg._unit = unit;
    return svg;
  }
  function renderTiles(tiles) {
    const box = clear($("#tiles"));
    for (const t of tiles) {
      let value = fmtUnit(t.value, t.unit);
      let unit = "";
      const m = value.match(/^([\d.,]+)\s+(.*)$/);
      if (m && t.unit !== "tokens") { value = m[1]; unit = m[2]; }
      const sub = t.id === "memory" && nn(t.budget) ? `of ${Math.round(t.budget / 1e9)} GB budget` : t.id === "overhead" ? "budget 1%" : `${t.spark.filter(nn).length} turns`;
      box.append(el("div", { class: "tile", "data-tile": t.id }, el("div", { class: "label", text: t.label }),
        el("div", { class: "value" }, value, unit ? el("small", { text: unit }) : null), el("div", { class: "sub", text: sub }), spark(t.spark || [], t.unit)));
    }
  }
  function highlightTiles(turnIndex) {
    for (const svg of document.querySelectorAll(".tile svg")) {
      const dot = svg.querySelector(".spark-dot");
      if (!dot || !svg._values) continue;
      const idx = turnIndex === null ? svg._values.length - 1 : turnIndex;
      const v = svg._values[idx];
      if (!nn(v)) continue;
      dot.setAttribute("cx", String((idx / Number(svg.dataset.n)) * 100)); dot.setAttribute("cy", String(svg._y(v)));
      const tile = svg.closest(".tile");
      const sub = tile.querySelector(".sub");
      if (turnIndex !== null) sub.textContent = `turn ${idx + 1}: ${fmtUnit(v, svg._unit)}`;
    }
  }

  function card(title, headline, wide, collapsedDefault, id) {
    const key = `innards-collapsed-${id || title}`;
    const saved = storageGet(key);
    const collapsed = saved === null ? !!collapsedDefault && false : saved === "1";
    const node = el("div", { class: `card${wide ? " wide" : ""}${collapsed ? " collapsed" : ""}` });
    const btn = el("button", { class: "collapse", type: "button", "aria-label": "Collapse", text: "▾",
      onclick: () => { node.classList.toggle("collapsed"); storageSet(key, node.classList.contains("collapsed") ? "1" : "0"); if (!node.classList.contains("collapsed")) resizeCharts(node); } });
    node.append(el("div", { class: "card-head" }, el("h3", { text: title }), headline ? el("span", { class: "pill neutral", text: headline }) : null, btn));
    const body = el("div", { class: "card-body" });
    node.append(body);
    return [node, body];
  }
  function changedCard(changed) {
    const newer = short(changed.newer_id); const previous = short(changed.previous_id);
    const [node, body] = card(`What changed: ${previous} → ${newer}`, changed.pair ? changed.pair.feature : null, true, false, "changed");
    if (changed.pair && changed.pair.change) body.append(el("p", { class: "findings", style: "padding:0", text: changed.pair.change }));
    const chips = el("div", { class: "chips" });
    for (const a of changed.added || []) chips.append(el("span", { class: "chip added", text: `+ ${a.name}` }));
    for (const d of changed.dropped || []) chips.append(el("span", { class: "chip dropped", text: d.name }));
    if (chips.childNodes.length) body.append(chips);
    const table = el("table", { class: "facts" });
    table.append(el("thead", {}, el("tr", {}, el("th", { text: "" }),
      el("th", {}, el("span", { class: "swatch-key", style: `background:${colorFor(changed.previous_id)}` }), previous),
      el("th", {}, el("span", { class: "swatch-key", style: `background:${colorFor(changed.newer_id)}` }), newer))));
    const tb = el("tbody");
    const show = (key, v) => {
      if (v === null || v === undefined) return "–";
      if (key && key.startsWith("kv_at")) return fmtBytes(v);
      if (key && key.startsWith("kv_per")) return `${fmtBytes(v)}/token`;
      if (typeof v === "object") return Object.entries(v).map(([k, n]) => `${n} ${k.replace(/_/g, " ")}`).join(", ");
      return typeof v === "number" ? fmtInt(v) : String(v);
    };
    for (const r of changed.facts || []) tb.append(el("tr", {}, el("td", { text: r.fact }), el("td", { text: show(r.key, r.previous) }), el("td", { text: show(r.key, r.newer) })));
    table.append(tb);
    body.append(table);
    return node;
  }
  function sectionCard(s, wide) {
    const [node, body] = card(s.title, s.headline, true, false, s.id);
    if (s.findings && s.findings.length) body.append(el("ul", { class: "findings" }, s.findings.map((f) => el("li", { text: f }))));
    const single = (s.charts || []).length === 1 && s.id && s.id.startsWith("experts-");
    const charts = el("div", { class: `charts${single ? " single" : ""}` });
    for (const spec of s.charts || []) {
      const box = el("div", { class: "chart-box" }, el("h4", { text: spec.title }));
      const rows = spec.type === "heatmap" ? (spec.y || []).length : 0;
      const size = rows > 40 ? " taller" : rows > 16 ? " tall" : "";
      const plot = el("div", { class: `chart${size}${single && rows > 16 ? " taller" : ""}`, role: "img", "aria-label": spec.title });
      box.append(plot);
      charts.append(box);
      requestAnimationFrame(() => drawChart(plot, spec, box));
    }
    body.append(charts);
    void wide;
    return node;
  }
  function compareCard(c) {
    const [node, body] = card(c.title, c.headline, true, false, "compare");
    body.append(el("ul", { class: "findings" }, (c.findings || []).map((f) => el("li", { text: f }))));
    const [a, b] = c.models;
    const table = el("table", { class: "facts" });
    table.append(el("thead", {}, el("tr", {}, ["Turn", "Message", `${short(a)} context`, `${short(b)} context`, `${short(a)} KV`, `${short(b)} KV`, `${short(a)} t/s`, `${short(b)} t/s`].map((h) => el("th", { text: h })))));
    const tb = el("tbody");
    for (const r of c.rows || []) {
      tb.append(el("tr", {}, el("td", { class: "num", text: r.turn }), el("td", { text: (r.message || "").slice(0, 80) + ((r.message || "").length > 80 ? "…" : "") }),
        el("td", { class: "num", text: fmtInt(r.a.context) }), el("td", { class: "num", text: fmtInt(r.b.context) }),
        el("td", { class: "num", text: fmtBytes(r.a.kv) }), el("td", { class: "num", text: fmtBytes(r.b.kv) }),
        el("td", { class: "num", text: fmtNum(r.a.decode, 1) }), el("td", { class: "num", text: fmtNum(r.b.decode, 1) })));
    }
    table.append(tb);
    body.append(table);
    const charts = el("div", { class: "charts", style: "margin-top:10px" });
    for (const spec of c.charts || []) {
      const box = el("div", { class: "chart-box" }, el("h4", { text: spec.title }));
      const plot = el("div", { class: "chart" });
      box.append(plot); charts.append(box);
      requestAnimationFrame(() => drawChart(plot, spec, box));
    }
    body.append(charts);
    return node;
  }
  async function renderPredicted(modelId) {
    const head = $("#an-head"); const grid = clear($("#an-grid")); clear($("#tiles"));
    S.turnCharts = [];
    let pred; let changed = null;
    try { pred = await api(`/api/lineup/predicted?model=${encodeURIComponent(modelId)}`); } catch (err) { banner(err.message); return; }
    try { changed = await api(`/api/lineup/changed?model=${encodeURIComponent(modelId)}`); } catch { /* not a pair */ }
    clear(head).append(el("h2", {}, el("span", { class: "swatch-key", style: `background:${colorFor(modelId)}` }), modelId, " ", el("span", { class: "predicted-note", text: "predicted from config, not measured" })),
      el("p", { text: `${pred.architecture} · ${pred.formula}` }));
    if (changed) grid.append(changedCard(changed));
    const [node, body] = card("KV predicted from the config (bf16)", "predicted", true, false, "predicted");
    const groups = [...new Set(pred.points.flatMap((p) => Object.keys(p.by_layer_type)))].filter((g) => pred.points.some((p) => p.by_layer_type[g]));
    const spec = { type: "bar", title: "Predicted KV by part", x: pred.points.map((p) => `${fmtInt(p.tokens)} tokens`), series: groups.map((g) => ({ name: g, values: pred.points.map((p) => p.by_layer_type[g] || 0) })), unit: "bytes", stacked: true, x_label: "context", y_label: "bytes" };
    body.append(el("ul", { class: "findings" }, pred.points.map((p) => el("li", { text: `At ${fmtInt(p.tokens)} tokens: ${fmtBytes(p.total_bytes)}` })), (pred.notes || []).map((n) => el("li", { class: "muted", text: n }))));
    const box = el("div", { class: "chart-box" }); const plot = el("div", { class: "chart" }); box.append(plot);
    body.append(box);
    grid.append(node);
    requestAnimationFrame(() => drawChart(plot, spec, box));
  }

  // ------------------------------------------------------------------ experts (MoE routing)
  async function refreshTraces() {
    try { S.traces = await api("/api/experts"); } catch { S.traces = []; }
    const sel = $("#ex-trace");
    const keep = sel.value || storageGet("innards-ex-trace") || "";
    clear(sel);
    for (const t of S.traces) sel.append(el("option", { value: `${t.run}/${t.name}`, text: `${t.run} · ${t.name} (${short(t.model)})` }));
    const values = S.traces.map((t) => `${t.run}/${t.name}`);
    sel.value = values.includes(keep) ? keep : (values[0] || "");
  }
  function fillExpertSelects(data) {
    const topic = $("#ex-topic"); const layer = $("#ex-layer");
    clear(topic); clear(layer);
    for (const t of data.topics || []) topic.append(el("option", { value: t, text: `${t} (${fmtInt(data.tokens[t])} tokens)` }));
    for (const l of data.layers || []) layer.append(el("option", { value: String(l), text: String(l) }));
    if (data.topic !== null && data.topic !== undefined) topic.value = data.topic;
    if (data.layer !== null && data.layer !== undefined) layer.value = String(data.layer);
  }
  function expertParams() {
    const params = new URLSearchParams();
    if ($("#ex-topic").value) params.set("topic", $("#ex-topic").value);
    if ($("#ex-layer").value) params.set("layer", $("#ex-layer").value);
    params.set("by", $("#ex-by").value || "selection");
    const cov = Number($("#ex-coverage").value);
    params.set("coverage", String(cov > 0 && cov <= 1 ? cov : 0.9));
    return params;
  }
  async function renderExperts() {
    const value = $("#ex-trace").value;
    const grid = $("#ex-grid"); const tokens = $("#ex-tokens");
    storageSet("innards-ex-trace", value);
    if (!value) {
      clear(grid).append(el("div", { class: "card wide" }, el("div", { class: "card-body muted",
        text: "No expert traces yet. Record one with innards.experts.trace(model, tokenizer, run=...) and trace.topic(...) around generate() or a training step." })));
      clear(tokens);
      return;
    }
    const [run, name] = value.split("/");
    let data;
    try {
      grid.style.opacity = "0.55";
      data = await api(`/api/experts/${encodeURIComponent(run)}/${encodeURIComponent(name)}?${expertParams()}`);
    } catch (err) { banner(err.message); return; } finally { grid.style.opacity = ""; }
    S.experts = { run, name, data };
    fillExpertSelects(data);
    clear(grid);
    for (const s of data.sections || []) grid.append(sectionCard(s, true));
    await renderTokens();
  }
  async function renderTokens() {
    const box = clear($("#ex-tokens"));
    const ex = S.experts;
    if (!ex || !ex.data.has_tokens || !ex.data.topic) return;
    const [node, body] = card("Tokens and their experts", null, true, false, "ex-tokens");
    box.append(node);
    let seqs;
    try {
      seqs = await api(`/api/experts/${encodeURIComponent(ex.run)}/${encodeURIComponent(ex.name)}/sequences?topic=${encodeURIComponent(ex.data.topic)}`);
    } catch (err) { body.append(el("p", { class: "muted", text: err.message })); return; }
    if (!seqs.length) { body.append(el("p", { class: "muted", text: "No recorded sequences for this topic." })); return; }
    const pick = el("select", { "aria-label": "Sequence" });
    for (const q of seqs.slice(0, 500)) pick.append(el("option", { value: String(q.sequence), text: `#${q.sequence} · ${q.tokens} tokens · ${q.preview}` }));
    const chart = el("div");
    body.append(el("div", { class: "toolbar" }, el("label", { class: "field-inline grow" }, "Sequence ", pick)), chart);
    const draw = async () => {
      try {
        const s = await api(`/api/experts/${encodeURIComponent(ex.run)}/${encodeURIComponent(ex.name)}/sequence/${pick.value}?layer=${$("#ex-layer").value}`);
        clear(chart).append(sectionCard(s, true));
      } catch (err) { clear(chart).append(el("p", { class: "muted", text: err.message })); }
    };
    pick.addEventListener("change", draw);
    await draw();
  }
  async function showExpert(layer, expert, topic) {
    const ex = S.experts;
    if (!ex || !ex.data.has_tokens) { banner("This trace kept counts only (tokens=False), so there are no per-token records to show.", "info"); return; }
    const params = new URLSearchParams({ layer: String(layer), expert: String(expert) });
    if (topic) params.set("topic", topic);
    try {
      const s = await api(`/api/experts/${encodeURIComponent(ex.run)}/${encodeURIComponent(ex.name)}/expert?${params}`);
      const old = document.getElementById("ex-expert-card");
      const node = sectionCard(s, true);
      node.id = "ex-expert-card";
      if (old) old.replaceWith(node); else $("#ex-grid").prepend(node);
      node.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) { banner(err.message); }
  }
  function onChartClick(spec, pt) {
    if (spec.click !== "expert" || !S.experts) return;
    const expert = spec.x[pt.x];
    if (spec.layer !== undefined) showExpert(spec.layer, expert, pt.y);
    else showExpert(Number(pt.y), expert, S.experts.data.topic);
  }
  async function downloadSelection() {
    const ex = S.experts;
    if (!ex || !ex.data.topic) return;
    const params = expertParams();
    params.delete("layer");
    if (params.get("by") === "lift") params.set("by", "selection");
    try {
      const res = await api(`/api/experts/${encodeURIComponent(ex.run)}/${encodeURIComponent(ex.name)}/selection?${params}`, { raw: true });
      const blob = await res.blob();
      const a = el("a", { href: URL.createObjectURL(blob), download: `${ex.name}-${ex.data.topic}-experts.json` });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (err) { banner(err.message); }
  }

  // ------------------------------------------------------------------ charts
  const GRID = "#eceef1"; const AXIS = "#e4e6eb"; const INK2 = "#4b5160";
  const HEAT = [[0, "#eef4fc"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#256abf"], [1, "#0d366b"]];
  function bytesAxis(values, fromZero) {
    const vals = values.filter(nn);
    if (!vals.length) return {};
    const hi = Math.max(...vals); const lo = fromZero ? Math.min(0, ...vals) : Math.min(...vals);
    const span = hi - lo || Math.abs(hi) || 1;
    const raw = span / 4;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const start = Math.floor(lo / step) * step; const end = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = start; v <= end + step * 0.01; v += step) ticks.push(v);
    const pad = step * 0.15;
    return { tickvals: ticks, ticktext: ticks.map(fmtBytes), range: [start - (fromZero && start === 0 ? 0 : pad), end + pad] };
  }
  function baseLayout(spec, extra) {
    return Object.assign({
      margin: { l: 56, r: 10, t: 6, b: 38 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: getComputedStyle(document.body).fontFamily, size: 11, color: INK2 }, showlegend: false,
      hoverlabel: { bgcolor: "#ffffff", bordercolor: AXIS, font: { color: "#111318", size: 12 } },
      xaxis: { title: { text: spec.x_label || "", standoff: 6 }, gridcolor: GRID, linecolor: AXIS, zeroline: false, automargin: true },
      yaxis: { title: { text: spec.y_label || "", standoff: 6 }, gridcolor: GRID, linecolor: AXIS, zeroline: false, automargin: true },
    }, extra || {});
  }
  function legend(box, series, colors, dashed) {
    if (series.length < 2) return;
    const lg = el("div", { class: "legend" });
    series.forEach((s, i) => lg.append(el("span", {}, el("span", { class: `key${dashed[i] ? " dot" : ""}`, style: dashed[i] ? `color:${colors[i]}` : `background:${colors[i]}` }), s.name)));
    box.append(lg);
  }
  function seriesColors(series) {
    const models = [...new Set(series.map((s) => s.model).filter(Boolean))];
    return series.map((s, i) => {
      if (s.model && models.length > 1) return colorFor(s.model);
      if (s.model && models.length === 1 && series.every((x) => x.model === s.model)) {
        // one model, several measures: first measure in its color, the rest in the next slots
        const base = colorFor(s.model);
        const rest = SLOTS.filter((c) => c !== base);
        return i === 0 ? base : rest[(i - 1) % rest.length];
      }
      return SLOTS[i % SLOTS.length];
    });
  }
  function drawChart(node, spec, box) {
    if (typeof Plotly === "undefined") { node.textContent = "Charts are loading…"; setTimeout(() => drawChart(node, spec, box), 300); return; }
    const config = { displayModeBar: false, responsive: true };
    if (spec.type === "heatmap") {
      const xs = spec.x.map((_, i) => i);
      const text = spec.x.map((v) => String(v));
      const custom = spec.z.map((row) => row.map((v) => fmtUnit(v, spec.unit)));
      const ylabels = spec.y.map((v) => String(v));
      const tickEvery = Math.max(1, Math.ceil(xs.length / 24));
      const trace = { type: "heatmap", x: xs, y: ylabels.map((_, i) => i), z: spec.z, colorscale: HEAT, zmin: spec.zmin ?? undefined, zmax: spec.zmax ?? undefined,
        customdata: custom, text: spec.z.map((row) => row.map((_, j) => text[j])), xgap: xs.length < 60 ? 1 : 0, ygap: ylabels.length < 40 ? 1 : 0,
        hovertemplate: `${spec.y_label || "row"} %{y}<br>${spec.x_label || "x"} %{text}<br><b>%{customdata}</b><extra></extra>`,
        colorbar: { thickness: 8, outlinewidth: 0, tickfont: { size: 10 }, len: 0.9 } };
      trace.y = ylabels;
      trace.hovertemplate = `${spec.y_label || "row"} %{y}<br>${spec.x_label || "x"} %{text}<br><b>%{customdata}</b><extra></extra>`;
      const layout = baseLayout(spec, {});
      layout.xaxis = Object.assign(layout.xaxis, { tickvals: xs.filter((i) => i % tickEvery === 0), ticktext: text.filter((_, i) => i % tickEvery === 0), showgrid: false, tickangle: spec.x_label === "token" ? -50 : 0 });
      layout.yaxis = Object.assign(layout.yaxis, { type: "category", showgrid: false, autorange: "reversed", dtick: ylabels.length <= 32 ? 1 : undefined });
      if (spec.x_label === "token") layout.margin.b = 70;
      Plotly.newPlot(node, [trace], layout, config);
      if (spec.x_label === "turn") registerTurnChart(node, spec, true);
      if (spec.click) node.on("plotly_click", (ev) => { const pt = ev.points && ev.points[0]; if (pt) onChartClick(spec, pt); });
      return;
    }
    const series = spec.series || [];
    const colors = seriesColors(series);
    const dashed = series.map((s) => s.dash === "dot");
    const layout = baseLayout(spec, { hovermode: "x unified" });
    const allValues = series.flatMap((s) => s.values || []);
    if (spec.unit === "bytes") {
      const stackedTotals = spec.x.map((_, i) => series.reduce((a, s) => a + (s.values[i] || 0), 0));
      layout.yaxis = Object.assign(layout.yaxis, bytesAxis(spec.stacked ? stackedTotals : allValues, spec.type === "bar"));
    }
    if (spec.x_label === "turn") layout.xaxis.dtick = spec.x.length > 16 ? undefined : 1;
    let traces;
    if (spec.type === "bar") {
      layout.barmode = spec.stacked ? "stack" : "group";
      layout.bargap = 0.35;
      traces = series.map((s, i) => ({ type: "bar", name: s.name, x: spec.x, y: s.values, marker: { color: colors[i], line: { color: "#ffffff", width: spec.stacked ? 1 : 0 } },
        customdata: (s.values || []).map((v) => fmtUnit(v, spec.unit)), hovertemplate: `%{customdata}<extra>${s.name}</extra>` }));
    } else {
      traces = series.map((s, i) => ({ type: "scatter", mode: spec.mode || "lines+markers", name: s.name, x: spec.x, y: s.values,
        line: { color: colors[i], width: 2, dash: s.dash === "dot" ? "dot" : "solid", shape: "linear" },
        marker: { size: 7, color: colors[i], line: { color: "#ffffff", width: 1.5 } }, connectgaps: false,
        customdata: (s.values || []).map((v) => fmtUnit(v, spec.unit)), hovertemplate: `%{customdata}<extra>${s.name}</extra>` }));
    }
    Plotly.newPlot(node, traces, layout, config);
    legend(box, series, colors, dashed);
    if (spec.x_label === "turn") registerTurnChart(node, spec, false);
  }
  // shared crosshair: hovering turn N in one chart highlights turn N everywhere
  function registerTurnChart(node, spec, heat) {
    S.turnCharts.push({ node, spec, heat });
    node.on("plotly_hover", (ev) => {
      if (node._syncing) return;
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      const turn = heat ? spec.x[pt.x] : pt.x;
      syncHover(node, turn);
    });
    node.on("plotly_unhover", () => { if (!node._syncing) syncHover(node, null); });
  }
  function syncHover(source, turn) {
    for (const c of S.turnCharts) {
      if (c.node === source || !c.node.isConnected) continue;
      c.node._syncing = true;
      try {
        if (turn === null) Plotly.Fx.unhover(c.node);
        else if (c.heat) {
          const idx = c.spec.x.indexOf(turn);
          if (idx >= 0) Plotly.relayout(c.node, { shapes: [{ type: "rect", xref: "x", yref: "paper", x0: idx - 0.5, x1: idx + 0.5, y0: 0, y1: 1, line: { color: "#111318", width: 1 }, fillcolor: "rgba(0,0,0,0)" }] });
        } else Plotly.Fx.hover(c.node, { xval: turn });
        if (turn === null && c.heat) Plotly.relayout(c.node, { shapes: [] });
      } catch { /* chart not ready */ } finally { c.node._syncing = false; }
    }
    const firstTurnChart = S.turnCharts[0];
    const index = turn === null || !firstTurnChart ? null : Number(turn) - 1;
    highlightTiles(index === null || Number.isNaN(index) ? null : index);
  }
  function resizeCharts(root) { for (const n of root.querySelectorAll(".js-plotly-plot")) { try { Plotly.Plots.resize(n); } catch { /* ignore */ } } }

  // ------------------------------------------------------------------ tabs, bindings, start
  function showTab(tab) {
    for (const b of document.querySelectorAll("[role=tab]")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    for (const p of document.querySelectorAll("[role=tabpanel]")) p.hidden = p.dataset.panel !== tab;
    storageSet("innards-tab", tab);
    if (tab === "analysis") { if (!$("#an-compare").value) suggestCompare(); renderAnalysis(); }
    if (tab === "experts") refreshTraces().then(renderExperts);
  }
  function bind() {
    for (const b of document.querySelectorAll("[role=tab]")) b.addEventListener("click", () => showTab(b.dataset.tab));
    $("#load-form").addEventListener("submit", loadModel);
    $("#check-btn").addEventListener("click", checkModel);
    $("#set-select").addEventListener("change", onSetChange);
    $("#doc-btn").addEventListener("click", () => $("#doc-input").click());
    $("#doc-input").addEventListener("change", onDocument);
    $("#run-set").addEventListener("click", runSet);
    $("#new-conv").addEventListener("click", newConversation);
    $("#composer").addEventListener("submit", sendMessage);
    $("#message").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
    $("#stop-btn").addEventListener("click", stopTurn);
    $("#viewing").addEventListener("change", (e) => { const [r, c] = e.target.value.split("/"); openConversation(r, c); });
    $("#an-conv").addEventListener("change", () => { S.anPicked = $("#an-conv").value; $("#an-conv").dataset.shown = S.anPicked; $("#an-compare").value = ""; suggestCompare(); renderAnalysis(); });
    for (const id of ["#an-compare", "#an-first", "#an-last"]) $(id).addEventListener("change", () => renderAnalysis());
    $("#ex-trace").addEventListener("change", () => { $("#ex-topic").value = ""; $("#ex-layer").value = ""; renderExperts(); });
    for (const id of ["#ex-topic", "#ex-by", "#ex-layer", "#ex-coverage"]) $(id).addEventListener("change", () => renderExperts());
    $("#ex-download").addEventListener("click", downloadSelection);
    window.addEventListener("resize", () => { clearTimeout(S.resizeTimer); S.resizeTimer = setTimeout(() => resizeCharts(document), 150); });
  }
  async function init() {
    bind();
    try {
      [S.info, S.lineup, S.sets] = await Promise.all([api("/api/info"), api("/api/lineup"), api("/api/sets")]);
    } catch (err) { banner(`Could not reach the view server: ${err.message}`); return; }
    $("#store-url").textContent = S.info.store;
    for (const m of S.lineup.models) colorFor(m.id);
    fillModels();
    fillSets();
    await refreshRuns();
    await poll();
    if (!S.viewing) {
      const first = S.runs[0] && (S.runs[0].conversations || []).slice(-1)[0];
      if (first) await openConversation(S.runs[0].name, first.id, { quiet: true });
      else renderMessages();
    }
    const tab = storageGet("innards-tab");
    showTab(tab === "analysis" || tab === "experts" ? tab : "conversation");
  }
  document.addEventListener("DOMContentLoaded", init);
})();
