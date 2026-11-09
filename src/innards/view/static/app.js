/* innards view: Conversation and MoE layers. Talks to the JSON API only. Vanilla JS + the vendored Plotly.
   Untrusted text goes in via textContent. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ state and helpers
  const S = {
    info: null,
    runner: null, // the runner's status, or null when the page cannot chat
    traces: [],
    convs: [], // stored conversations of the selected trace
    convPicked: null, // a conversation the viewer picked; null follows the live one
    job: null, // the reply streaming in
    streaming: false,
    experts: null, // the MoE layers data on screen
    focus: null, // a conversation turn opened on the MoE layers tab
    moeKey: "",
    pollTimer: null,
    lastSeen: null,
    lastTraces: 0,
    traceStamp: "",
    models: [],
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
  const fmtInt = (n) => (nn(n) ? Math.round(Number(n)).toLocaleString() : "n/a");
  const fmtNum = (n, d = 1) => (nn(n) ? Number(n).toFixed(d) : "n/a");
  function fmtUnit(v, unit) {
    if (!nn(v)) return "n/a";
    if (unit === "count") return fmtInt(v);
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
  function headers(extra) {
    const h = { Accept: "application/json", ...(extra || {}) };
    const t = token();
    if (t) h.Authorization = `Bearer ${t}`;
    return h;
  }
  async function api(path, opts = {}) {
    const h = headers(opts.headers);
    if (opts.method === "POST") { h["X-Innards"] = "1"; h["Content-Type"] = "application/json"; }
    const res = await fetch(path, { ...opts, headers: h });
    if (!res.ok) {
      let detail = res.statusText;
      try { const body = await res.json(); detail = body.detail !== undefined ? body.detail : detail; } catch { /* not json */ }
      const err = new Error(typeof detail === "string" ? detail : (detail.message || JSON.stringify(detail)));
      err.status = res.status;
      throw err;
    }
    return opts.raw ? res : res.json();
  }
  function banner(text, kind) {
    const node = $("#banner");
    if (!text) { node.hidden = true; return; }
    node.hidden = false; node.className = `banner${kind === "info" ? " info" : ""}`; node.textContent = text;
  }
  const traceUrl = (path) => {
    const [run, name] = (S.trace || "").split("/");
    return `/api/experts/${encodeURIComponent(run)}/${encodeURIComponent(name)}${path || ""}`;
  };
  const runnerTrace = () => (S.runner && S.runner.trace ? `${S.runner.trace.run}/${S.runner.trace.name}` : null);
  const seenKey = (r) => (r ? [r.model, r.state === "generating" ? "ready" : r.state, r.conversation, r.turns].join("|") : "");
  const fmtGB = (n) => (nn(n) ? `${(n / 1e9).toFixed(1)} GB` : "n/a");
  const onRunnerTrace = () => !!S.runner && S.trace === runnerTrace();

  // ------------------------------------------------------------------ header and polling
  function renderLive() {
    const r = S.runner;
    const live = $("#live"); const text = $("#live-text");
    live.className = "live";
    if (!r) { live.classList.add("idle"); text.textContent = "Run folder"; live.title = "Run innards view, or trace.view(chat=True) from your script, to chat"; return; }
    live.title = r.model || "";
    if (r.state === "loading") { live.classList.add("loading"); text.textContent = `Loading ${short(r.model)}…`; }
    else if (r.state === "idle") { live.classList.add("idle"); text.textContent = "No model loaded"; }
    else if (r.state === "generating" || S.streaming) { live.classList.add("on"); text.textContent = `Generating · ${short(r.model)}`; }
    else if (r.state === "error") { live.classList.add("error"); text.textContent = "Error"; live.title = r.error || ""; }
    else { live.classList.add("ready"); text.textContent = `Ready · ${short(r.model)}`; }
  }
  function schedulePoll(ms) { clearTimeout(S.pollTimer); S.pollTimer = setTimeout(poll, ms); }
  async function poll() {
    let busy = false;
    try {
      S.runner = (await api("/api/state")).runner;
      busy = S.streaming || (S.runner && ["generating", "loading"].includes(S.runner.state));
      const seen = seenKey(S.runner);
      if (seen !== S.lastSeen) { // a model loaded, a turn finished, or a conversation started somewhere else
        const first = S.lastSeen === null;
        S.lastSeen = seen;
        const rt = runnerTrace();
        if (rt && rt !== S.lastRunnerTrace) { S.trace = rt; S.convPicked = null; S.focus = null; }
        S.lastRunnerTrace = rt;
        if (!first && !S.streaming) { await refreshTraces(); await loadConversations(); markMoeStale(); }
      } else if (Date.now() - S.lastTraces > 15000 && !busy) {
        if (await refreshTraces()) markMoeStale();
      }
      renderLive();
      renderComposer();
      if (S.runner && S.runner.job && !S.streaming && onRunnerTrace()) renderMessages();
    } catch (err) {
      banner(`Lost contact with the view server: ${err.message}`);
    }
    schedulePoll(busy ? 1000 : 3000);
  }

  // ------------------------------------------------------------------ traces
  async function refreshTraces() {
    try { S.traces = await api("/api/experts"); } catch (err) { S.traces = []; banner(err.message); }
    S.lastTraces = Date.now();
    const sel = clear($("#trace"));
    for (const t of S.traces) {
      const total = Object.values(t.tokens || {}).reduce((a, b) => a + b, 0);
      sel.append(el("option", { value: `${t.run}/${t.name}`, text: `${t.run} · ${t.name} · ${short(t.model)} · ${fmtInt(total)} tokens` }));
    }
    const values = S.traces.map((t) => `${t.run}/${t.name}`);
    const keep = S.trace || runnerTrace() || storageGet("innards-trace") || "";
    S.trace = values.includes(keep) ? keep : (values[0] || "");
    sel.value = S.trace;
    sel.disabled = !values.length;
    const shown = S.traces.find((t) => `${t.run}/${t.name}` === S.trace);
    const stamp = shown ? `${S.trace}@${shown.updated_at}` : "";
    const changed = stamp !== S.traceStamp;
    S.traceStamp = stamp;
    return changed; // true when the trace on screen has new measurements
  }
  async function onTraceChange() {
    S.trace = $("#trace").value;
    storageSet("innards-trace", S.trace);
    S.focus = null; S.convPicked = null;
    $("#ex-topic").value = ""; $("#ex-layer").value = "";
    await loadConversations();
    markMoeStale();
  }

  // ------------------------------------------------------------------ conversation tab
  function convStart(c) {
    const at = c.turns.length ? c.turns[0].started_at : null;
    return at ? new Date(at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "new";
  }
  async function loadConversations() {
    S.convs = [];
    if (S.trace) {
      try { S.convs = await api(traceUrl("/conversations")); } catch (err) { banner(err.message); }
    }
    if (onRunnerTrace() && !S.convs.some((c) => c.id === S.runner.conversation)) {
      S.convs.push({ id: S.runner.conversation, turns: [] });
    }
    const sel = clear($("#conv"));
    for (const c of S.convs.slice().reverse()) {
      const live = onRunnerTrace() && c.id === S.runner.conversation;
      const preview = c.turns.length ? ` · ${c.turns[0].message.slice(0, 48)}` : "";
      sel.append(el("option", { value: c.id, text: `${live ? "Live · " : ""}${convStart(c)} · ${c.turns.length} turn${c.turns.length === 1 ? "" : "s"}${preview}` }));
    }
    const ids = S.convs.map((c) => c.id);
    const live = onRunnerTrace() ? S.runner.conversation : null;
    const pick = S.convPicked && ids.includes(S.convPicked) ? S.convPicked : live || ids[ids.length - 1] || "";
    sel.value = pick;
    sel.disabled = !ids.length;
    renderMessages();
    renderComposer();
  }
  function currentConv() { return S.convs.find((c) => c.id === $("#conv").value) || null; }
  function isLiveConv() { return onRunnerTrace() && $("#conv").value === S.runner.conversation; }

  function turnMeta(t) {
    const tok = t.tokens || {};
    const parts = [
      ["topic", t.topic],
      ["reused", fmtInt(tok.reused)],
      ["prefilled", fmtInt(tok.prefilled)],
      ["generated", fmtInt(tok.generated)],
    ];
    const meta = el("div", { class: "meta" }, parts.map(([k, v]) => el("span", {}, `${k} `, el("b", { text: v }))));
    if (t.stopped) meta.append(el("span", { class: "flag", text: "stopped" }));
    if (nn(t.sequence)) {
      meta.append(el("button", { class: "link", type: "button", title: "This turn's tokens and the experts each chose, per layer",
        onclick: () => openTurn(t), text: "Experts →" }));
    }
    return meta;
  }
  function bubble(role, text, extra) {
    const body = el("div", { class: "body", text: text || "" });
    const wrap = el("div", {}, body);
    if ((text || "").length > 1200) {
      body.classList.add("long");
      wrap.append(el("button", { class: "link", type: "button", onclick: (e) => { body.classList.toggle("long"); e.target.textContent = body.classList.contains("long") ? "Show all" : "Show less"; }, text: "Show all" }));
    }
    if (extra) wrap.append(extra);
    return el("div", { class: `msg ${role}` }, el("div", { class: "role", text: role === "user" ? "you" : "model" }), wrap);
  }
  function renderMessages() {
    const box = clear($("#messages"));
    const conv = currentConv();
    const turns = conv ? conv.turns : [];
    const remote = S.runner && S.runner.job && !S.streaming && isLiveConv() ? S.runner.job : null;
    if (!turns.length && !S.job && !remote) {
      let text;
      const r = S.runner;
      if (r && r.loadable && !r.trace) text = r.state === "loading" ? `Loading ${r.model}: the before-download checks, then the weights.` : "Pick an MoE model above and press Load. Checkpoints already downloaded are listed first; anything else downloads from the Hugging Face Hub.";
      else if (!S.traces.length) text = `No expert traces in ${S.info ? S.info.store : "this folder"}. Trace a model with innards.experts.trace(model, tokenizer, run=...) and serve this page with trace.view(chat=True) to talk to it.`;
      else if (isLiveConv()) text = `Talk to ${S.runner.model}. Each turn's tokens are recorded under the topic next to Send; open MoE layers to see the experts they chose.`;
      else text = "No stored conversations for this trace. They appear here when the page talks to the model: trace.view(chat=True).";
      box.append(el("div", { class: "empty", text }));
      return;
    }
    for (const t of turns) {
      box.append(bubble("user", t.message));
      box.append(bubble("model", t.reply, turnMeta(t)));
    }
    const pending = S.job || (remote ? { message: remote.message, topic: remote.topic, text: remote.text } : null);
    if (pending && isLiveConv()) {
      if (pending.message) box.append(bubble("user", pending.message));
      const b = bubble("model", pending.text || "", el("div", { class: "meta" }, el("span", {}, "topic ", el("b", { text: pending.topic || "" })), el("span", {}, "generating…")));
      b.classList.add("streaming");
      box.append(b);
    }
    box.scrollTop = box.scrollHeight;
  }
  function renderComposer() {
    renderLoadBar();
    const r = S.runner;
    const box = $("#message");
    const live = isLiveConv();
    const ok = !!(r && live && r.state !== "generating" && !S.streaming);
    box.disabled = !ok;
    $("#send-btn").disabled = !ok;
    $("#topic").disabled = !r || !live;
    $("#new-conv").disabled = !r || !r.trace || S.streaming || r.state === "generating";
    $("#stop-btn").hidden = !(S.streaming || (r && r.state === "generating"));
    const note = $("#conv-note");
    if (!r) { box.placeholder = "This page shows stored traces only"; note.textContent = "To chat, run innards view, or call trace.view(chat=True) from the script that traced the model."; }
    else if (!r.trace) { box.placeholder = r.state === "loading" ? `Loading ${short(r.model)}…` : "Load an MoE model above to start"; note.textContent = ""; }
    else if (!onRunnerTrace()) { box.placeholder = `Pick the trace ${runnerTrace()} to talk to ${short(r.model)}`; note.textContent = `This page talks to ${r.model}, traced as ${runnerTrace()}.`; }
    else if (!live) { box.placeholder = "Stored conversation: pick the live one, or start a new one, to send messages"; note.textContent = ""; }
    else if (r.mode === "text") {
      box.placeholder = "Text for the model to continue… (Enter to send, Shift+Enter for a new line)";
      note.textContent = `${short(r.model)} has no chat template, so this is text mode: your text is added to the text so far exactly as typed (start it with a space or a new line to separate it), and the model continues it.`;
    }
    else { box.placeholder = "Message… (Enter to send, Shift+Enter for a new line)"; note.textContent = ""; }
    if (r && !$("#topic").value && r.topic) $("#topic").value = r.topic;
  }

  // server-sent events over a POST
  async function streamPost(path, body) {
    const res = await fetch(path, { method: "POST", headers: headers({ "Content-Type": "application/json", "X-Innards": "1", Accept: "text/event-stream" }), body: JSON.stringify(body) });
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail; } catch { /* ignore */ }
      throw new Error(typeof detail === "string" ? detail : (detail && detail.message) || "request failed");
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
  let renderPending = false;
  function throttledRender() { if (renderPending) return; renderPending = true; requestAnimationFrame(() => { renderPending = false; renderMessages(); }); }
  function handleEvent(event, data) {
    if (event === "turn_start") { S.job = { message: data.message, topic: data.topic, text: "" }; renderMessages(); }
    else if (event === "token") { if (S.job) { S.job.text += data.text; throttledRender(); } }
    else if (event === "turn") {
      S.job = null;
      const rec = data.record;
      const conv = S.convs.find((c) => c.id === rec.conversation);
      if (conv) conv.turns.push(rec);
      renderMessages();
    } else if (event === "error") { S.job = null; banner(data.message || "The turn failed"); renderMessages(); }
  }
  async function sendMessage(ev) {
    if (ev) ev.preventDefault();
    const box = $("#message");
    const text = box.value.trim();
    const topic = $("#topic").value.trim();
    if (!text || box.disabled) return;
    if (!topic) { banner("Give the turn a topic: its tokens are recorded under it."); return; }
    banner("");
    box.value = "";
    storageSet("innards-topic", topic);
    S.streaming = true; renderComposer(); renderLive();
    try { await streamPost("/api/chat", { message: text, topic }); } catch (err) { banner(err.message); box.value = text; } finally {
      S.streaming = false; S.job = null;
      try {
        S.runner = (await api("/api/state")).runner;
        S.lastSeen = seenKey(S.runner); // this page already shows the turn
      } catch { /* the next poll reports it */ }
      renderLive();
      await refreshTraces();
      await loadConversations();
      markMoeStale();
      schedulePoll(3000);
    }
  }
  async function newConversation() {
    banner("");
    try { S.runner = await api("/api/conversation", { method: "POST", body: "{}" }); } catch (err) { banner(err.message); return; }
    S.convPicked = null;
    S.lastSeen = seenKey(S.runner);
    if (!onRunnerTrace()) { S.trace = runnerTrace(); await refreshTraces(); }
    await loadConversations();
    $("#message").focus();
  }
  async function stopTurn() { try { await api("/api/stop", { method: "POST", body: "{}" }); } catch (err) { banner(err.message); } }

  // ------------------------------------------------------------------ model (innards view loads it)
  function fillModels() {
    const list = clear($("#model-list"));
    for (const m of S.models) {
      const where = m.downloaded ? `downloaded, ${fmtGB(m.size_bytes)}` : `${fmtGB(m.size_bytes)} to download${m.released ? `, released ${m.released}` : ""}`;
      const text = m.chat_template === false ? " · no chat template: text mode" : "";
      list.append(el("option", { value: m.id, label: `${where} · ${m.experts} experts, top ${nn(m.top_k) ? m.top_k : "?"}${text}` }));
    }
    const saved = storageGet("innards-model");
    const first = S.models.find((m) => m.downloaded && m.chat_template !== false);
    if (!$("#model-input").value) $("#model-input").value = (S.runner && S.runner.model) || saved || (first ? first.id : "");
  }
  function renderLoadBar() {
    const r = S.runner;
    const form = $("#load-form");
    form.hidden = !(r && r.loadable);
    if (form.hidden) return;
    const busy = S.streaming || ["loading", "generating"].includes(r.state);
    $("#load-btn").disabled = busy;
    $("#model-input").disabled = r.state === "loading";
    const status = $("#load-status");
    status.className = "small";
    if (r.state === "loading") { status.classList.add("muted"); status.textContent = `Loading ${r.model}…`; }
    else if (r.trace) {
      status.classList.add("muted");
      status.textContent = `${short(r.model)} on ${r.device || "?"}, ${r.remote_code ? "running the checkpoint's own code" : "transformers code"}${r.mode === "text" ? ", text mode (no chat template)" : ""}`;
    }
    else if (r.state === "error") { status.classList.add("error-text"); status.textContent = r.error || "Loading failed"; }
    else {
      const picked = S.models.find((m) => m.id === $("#model-input").value.trim());
      status.className = "small muted";
      status.textContent = picked && !picked.downloaded
        ? `${short(picked.id)} downloads ${fmtGB(picked.size_bytes)} from the Hugging Face Hub when you press Load (checked with transformers ${picked.checked_with || "?"}).`
        : "";
    }
    $("#load-btn").textContent = r.trace && r.model === $("#model-input").value.trim() ? "Reload" : "Load";
  }
  async function loadModel(ev) {
    if (ev) ev.preventDefault();
    const model = $("#model-input").value.trim();
    if (!model) return;
    banner("");
    storageSet("innards-model", model);
    try { S.runner = await api("/api/load", { method: "POST", body: JSON.stringify({ model }) }); } catch (err) { banner(err.message); return; }
    renderLive(); renderLoadBar(); renderComposer(); renderMessages();
    schedulePoll(500);
  }

  // ------------------------------------------------------------------ MoE layers tab
  function markMoeStale() { S.moeKey = ""; if (!$("#tab-moe").hidden) renderExperts(); }
  function expertParams() {
    const params = new URLSearchParams();
    const topic = S.focus ? S.focus.topic : $("#ex-topic").value;
    if (topic) params.set("topic", topic);
    if ($("#ex-layer").value) params.set("layer", $("#ex-layer").value);
    params.set("by", $("#ex-by").value || "selection");
    const cov = Number($("#ex-coverage").value);
    params.set("coverage", String(cov > 0 && cov <= 1 ? cov : 0.9));
    return params;
  }
  function fillExpertSelects(data) {
    const topic = clear($("#ex-topic")); const layer = clear($("#ex-layer"));
    for (const t of data.topics || []) topic.append(el("option", { value: t, text: `${t} (${fmtInt(data.tokens[t])} tokens)` }));
    for (const l of data.layers || []) layer.append(el("option", { value: String(l), text: String(l) }));
    if (nn(data.topic)) topic.value = data.topic;
    if (nn(data.layer)) layer.value = String(data.layer);
  }
  async function renderExperts() {
    const grid = $("#ex-grid"); const tokens = $("#ex-tokens");
    if (!S.trace) {
      clear(grid).append(el("div", { class: "card wide" }, el("div", { class: "card-body muted",
        text: "No expert traces yet. Record one with innards.experts.trace(model, tokenizer, run=...) and trace.topic(...) around generate() or a training step, or talk to a traced model on the Conversation tab." })));
      clear(tokens);
      return;
    }
    const params = expertParams();
    const key = `${S.trace}?${params}|${S.focus ? JSON.stringify(S.focus) : ""}`;
    if (key === S.moeKey) return;
    S.moeKey = key;
    let data;
    try {
      grid.style.opacity = "0.55";
      data = await api(`${traceUrl()}?${params}`);
    } catch (err) { banner(err.message); return; } finally { grid.style.opacity = ""; }
    S.experts = data;
    fillExpertSelects(data);
    clear(grid);
    for (const s of data.sections || []) grid.append(sectionCard(s));
    $("#tab-moe").insertBefore(tokens, S.focus ? grid : null);
    await renderTokens();
  }
  async function renderTokens() {
    const box = clear($("#ex-tokens"));
    const data = S.experts;
    if (!data || !data.topic) return;
    const focus = S.focus;
    const title = focus ? `Turn ${focus.turn}: its tokens and their experts` : "Tokens and their experts";
    const [node, body] = card(title, focus ? focus.topic : null, "ex-tokens");
    box.append(node);
    if (!data.has_tokens) { body.append(el("p", { class: "muted", text: "This trace kept counts only (tokens=False), so there are no per-token records to show." })); return; }
    let seqs;
    try { seqs = await api(`${traceUrl("/sequences")}?topic=${encodeURIComponent(data.topic)}`); } catch (err) { body.append(el("p", { class: "muted", text: err.message })); return; }
    if (focus && !seqs.some((q) => q.sequence === focus.sequence)) seqs.unshift({ sequence: focus.sequence, tokens: null, preview: "this conversation" });
    if (!seqs.length) { body.append(el("p", { class: "muted", text: "No recorded sequences for this topic." })); return; }
    const pick = el("select", { "aria-label": "Sequence" });
    for (const q of seqs.slice(0, 500)) pick.append(el("option", { value: String(q.sequence), text: `#${q.sequence}${nn(q.tokens) ? ` · ${q.tokens} tokens` : ""} · ${q.preview}` }));
    const scope = el("select", { "aria-label": "Tokens shown" });
    if (focus) {
      pick.value = String(focus.sequence);
      scope.append(el("option", { value: "turn", text: `Turn ${focus.turn} (positions ${focus.first}–${focus.last - 1})` }), el("option", { value: "all", text: "Whole sequence" }));
    }
    const scopeWrap = el("label", { class: "field-inline", hidden: !focus }, "Tokens ", scope);
    const chart = el("div");
    body.append(el("div", { class: "toolbar" }, el("label", { class: "field-inline grow" }, "Sequence ", pick), scopeWrap,
      focus ? el("button", { class: "ghost", type: "button", text: "Back to all turns", onclick: () => { S.focus = null; markMoeStale(); } }) : null), chart);
    const draw = async () => {
      const onFocus = focus && pick.value === String(focus.sequence);
      scopeWrap.hidden = !onFocus;
      const params = new URLSearchParams({ layer: $("#ex-layer").value });
      if (onFocus && scope.value === "turn") { params.set("first", String(focus.first)); params.set("last", String(focus.last)); }
      try {
        const s = await api(`${traceUrl(`/sequence/${pick.value}`)}?${params}`);
        clear(chart).append(sectionCard(s, true));
      } catch (err) { clear(chart).append(el("p", { class: "muted", text: err.message })); }
    };
    pick.addEventListener("change", draw);
    scope.addEventListener("change", draw);
    await draw();
  }
  function openTurn(t) {
    const tok = t.tokens || {};
    S.focus = { sequence: t.sequence, first: t.positions ? t.positions[0] : tok.reused, last: t.positions ? t.positions[1] : null, turn: t.turn, topic: t.topic };
    S.moeKey = "";
    showTab("moe");
  }
  async function showExpert(layer, expert, topic) {
    if (!S.experts || !S.experts.has_tokens) { banner("This trace kept counts only (tokens=False), so there are no per-token records to show.", "info"); return; }
    const params = new URLSearchParams({ layer: String(layer), expert: String(expert) });
    if (topic) params.set("topic", topic);
    try {
      const s = await api(`${traceUrl("/expert")}?${params}`);
      const old = document.getElementById("ex-expert-card");
      const node = sectionCard(s);
      node.id = "ex-expert-card";
      if (old) old.replaceWith(node); else $("#ex-grid").prepend(node);
      node.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) { banner(err.message); }
  }
  function onChartClick(spec, pt) {
    if (spec.click !== "expert" || !S.experts) return;
    const expert = spec.x[pt.x];
    if (spec.layer !== undefined) showExpert(spec.layer, expert, pt.y);
    else showExpert(Number(pt.y), expert, S.experts.topic);
  }
  async function downloadSelection() {
    if (!S.experts || !S.experts.topic) return;
    const params = expertParams();
    params.set("topic", S.experts.topic);
    params.delete("layer");
    if (params.get("by") === "lift") params.set("by", "selection");
    try {
      const res = await api(`${traceUrl("/selection")}?${params}`, { raw: true });
      const blob = await res.blob();
      const name = (S.trace || "").split("/")[1];
      const a = el("a", { href: URL.createObjectURL(blob), download: `${name}-${S.experts.topic}-experts.json` });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (err) { banner(err.message); }
  }

  // ------------------------------------------------------------------ cards and charts
  function card(title, headline, id) {
    const key = `innards-collapsed-${id || title}`;
    const collapsed = storageGet(key) === "1";
    const node = el("div", { class: `card wide${collapsed ? " collapsed" : ""}` });
    const btn = el("button", { class: "collapse", type: "button", "aria-label": "Collapse", text: "▾",
      onclick: () => { node.classList.toggle("collapsed"); storageSet(key, node.classList.contains("collapsed") ? "1" : "0"); if (!node.classList.contains("collapsed")) resizeCharts(node); } });
    node.append(el("div", { class: "card-head" }, el("h3", { text: title }), headline ? el("span", { class: "pill", text: headline }) : null, btn));
    const body = el("div", { class: "card-body" });
    node.append(body);
    return [node, body];
  }
  function sectionCard(s, bare) {
    const [node, body] = bare ? [el("div"), null] : card(s.title, s.headline, s.id);
    const target = body || node;
    if (s.findings && s.findings.length) target.append(el("ul", { class: "findings" }, s.findings.map((f) => el("li", { text: f }))));
    const single = (s.charts || []).length === 1;
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
    target.append(charts);
    return node;
  }

  const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"];
  const GRID = "#eceef1"; const AXIS = "#e4e6eb"; const INK2 = "#4b5160";
  const HEAT = [[0, "#eef4fc"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#256abf"], [1, "#0d366b"]];
  function baseLayout(spec, extra) {
    return Object.assign({
      margin: { l: 56, r: 10, t: 6, b: 38 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: getComputedStyle(document.body).fontFamily, size: 11, color: INK2 }, showlegend: false,
      hoverlabel: { bgcolor: "#ffffff", bordercolor: AXIS, font: { color: "#111318", size: 12 } },
      xaxis: { title: { text: spec.x_label || "", standoff: 6 }, gridcolor: GRID, linecolor: AXIS, zeroline: false, automargin: true },
      yaxis: { title: { text: spec.y_label || "", standoff: 6 }, gridcolor: GRID, linecolor: AXIS, zeroline: false, automargin: true },
    }, extra || {});
  }
  function legend(box, series, dashed) {
    if (series.length < 2) return;
    const lg = el("div", { class: "legend" });
    series.forEach((s, i) => {
      const color = SLOTS[i % SLOTS.length];
      lg.append(el("span", {}, el("span", { class: `key${dashed[i] ? " dot" : ""}`, style: dashed[i] ? `color:${color}` : `background:${color}` }), s.name));
    });
    box.append(lg);
  }
  function drawChart(node, spec, box) {
    if (typeof Plotly === "undefined") { node.textContent = "Charts are loading…"; setTimeout(() => drawChart(node, spec, box), 300); return; }
    const config = { displayModeBar: false, responsive: true };
    if (spec.type === "heatmap") {
      const xs = spec.x.map((_, i) => i);
      const text = spec.x.map((v) => String(v));
      const ylabels = spec.y.map((v) => String(v));
      const tickEvery = Math.max(1, Math.ceil(xs.length / 24));
      const trace = { type: "heatmap", x: xs, y: ylabels, z: spec.z, colorscale: HEAT, zmin: spec.zmin ?? undefined, zmax: spec.zmax ?? undefined,
        customdata: spec.z.map((row) => row.map((v) => fmtUnit(v, spec.unit))), text: spec.z.map((row) => row.map((_, j) => text[j])),
        xgap: xs.length < 60 ? 1 : 0, ygap: ylabels.length < 40 ? 1 : 0,
        hovertemplate: `${spec.y_label || "row"} %{y}<br>${spec.x_label || "x"} %{text}<br><b>%{customdata}</b><extra></extra>`,
        colorbar: { thickness: 8, outlinewidth: 0, tickfont: { size: 10 }, len: 0.9 } };
      const layout = baseLayout(spec, {});
      layout.xaxis = Object.assign(layout.xaxis, { tickvals: xs.filter((i) => i % tickEvery === 0), ticktext: text.filter((_, i) => i % tickEvery === 0), showgrid: false, tickangle: spec.x_label === "token" ? -50 : 0 });
      layout.yaxis = Object.assign(layout.yaxis, { type: "category", showgrid: false, autorange: "reversed", dtick: ylabels.length <= 32 ? 1 : undefined });
      if (spec.x_label === "token") layout.margin.b = 70;
      Plotly.newPlot(node, [trace], layout, config);
      if (spec.click) node.on("plotly_click", (ev) => { const pt = ev.points && ev.points[0]; if (pt) onChartClick(spec, pt); });
      return;
    }
    const series = spec.series || [];
    const dashed = series.map((s) => s.dash === "dot");
    const layout = baseLayout(spec, { hovermode: "x unified" });
    let traces;
    if (spec.type === "bar") {
      layout.barmode = spec.stacked ? "stack" : "group";
      layout.bargap = 0.35;
      traces = series.map((s, i) => ({ type: "bar", name: s.name, x: spec.x, y: s.values, marker: { color: SLOTS[i % SLOTS.length], line: { color: "#ffffff", width: spec.stacked ? 1 : 0 } },
        customdata: (s.values || []).map((v) => fmtUnit(v, spec.unit)), hovertemplate: `%{customdata}<extra>${s.name}</extra>` }));
    } else {
      traces = series.map((s, i) => ({ type: "scatter", mode: spec.mode || "lines+markers", name: s.name, x: spec.x, y: s.values,
        line: { color: SLOTS[i % SLOTS.length], width: 2, dash: s.dash === "dot" ? "dot" : "solid", shape: "linear" },
        marker: { size: 7, color: SLOTS[i % SLOTS.length], line: { color: "#ffffff", width: 1.5 } }, connectgaps: false,
        customdata: (s.values || []).map((v) => fmtUnit(v, spec.unit)), hovertemplate: `%{customdata}<extra>${s.name}</extra>` }));
    }
    Plotly.newPlot(node, traces, layout, config);
    legend(box, series, dashed);
  }
  function resizeCharts(root) { for (const n of root.querySelectorAll(".js-plotly-plot")) { try { Plotly.Plots.resize(n); } catch { /* ignore */ } } }

  // ------------------------------------------------------------------ tabs, bindings, start
  function showTab(tab) {
    for (const b of document.querySelectorAll("[role=tab]")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    for (const p of document.querySelectorAll("[role=tabpanel]")) p.hidden = p.dataset.panel !== tab;
    storageSet("innards-tab", tab);
    if (tab === "moe") renderExperts(); else resizeCharts(document);
  }
  function bind() {
    for (const b of document.querySelectorAll("[role=tab]")) b.addEventListener("click", () => showTab(b.dataset.tab));
    $("#trace").addEventListener("change", onTraceChange);
    $("#load-form").addEventListener("submit", loadModel);
    $("#model-input").addEventListener("input", renderLoadBar);
    $("#conv").addEventListener("change", () => { S.convPicked = $("#conv").value; renderMessages(); renderComposer(); });
    $("#new-conv").addEventListener("click", newConversation);
    $("#composer").addEventListener("submit", sendMessage);
    $("#message").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
    $("#stop-btn").addEventListener("click", stopTurn);
    $("#ex-topic").addEventListener("change", () => { S.focus = null; renderExperts(); });
    for (const id of ["#ex-by", "#ex-layer", "#ex-coverage"]) $(id).addEventListener("change", () => renderExperts());
    $("#ex-download").addEventListener("click", downloadSelection);
    window.addEventListener("resize", () => { clearTimeout(S.resizeTimer); S.resizeTimer = setTimeout(() => resizeCharts(document), 150); });
  }
  async function init() {
    bind();
    try { S.info = await api("/api/info"); } catch (err) { banner(`Could not reach the view server: ${err.message}`); return; }
    $("#store-url").textContent = S.info.store;
    const topic = storageGet("innards-topic");
    if (topic) $("#topic").value = topic;
    await poll();
    if (runnerTrace()) S.trace = runnerTrace();
    if (S.info.load) { try { S.models = await api("/api/models"); } catch { S.models = []; } fillModels(); renderLoadBar(); }
    await refreshTraces();
    await loadConversations();
    const tab = storageGet("innards-tab");
    showTab(tab === "moe" ? "moe" : "conversation");
  }
  document.addEventListener("DOMContentLoaded", init);
})();
