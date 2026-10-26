"use strict";

// Streams answers from POST /api/chat (Server-Sent Events over fetch, since
// EventSource can't send a POST body) and renders each turn's answer, route,
// timeline, and sources. No framework, no build step.

const AGENT_NAMES = { supervisor: "Supervisor", research: "Research", summary: "Summary", data: "Data" };
const PROVIDER_NAMES = { google: "Google Gemini", nvidia: "NVIDIA Nemotron" };
const SHORT_PROVIDER = { google: "Gemini", nvidia: "Nemotron" };
const BASIS = { cost: "cost", capability: "capability", hybrid: "cost and capability", failover: "failover" };
const SESSION_KEY = "hka-session";

const el = {
  empty: document.getElementById("empty"),
  turns: document.getElementById("turns"),
  form: document.getElementById("ask"),
  query: document.getElementById("query"),
  submit: document.getElementById("submit"),
  reset: document.getElementById("new-conversation"),
  inspectorEmpty: document.getElementById("inspector-empty"),
  route: document.getElementById("route"),
  routeReason: document.getElementById("route-reason"),
  routeSteps: document.getElementById("route-steps"),
  timeline: document.getElementById("timeline"),
  waterfall: document.getElementById("waterfall"),
  sources: document.getElementById("sources"),
  sourceList: document.getElementById("source-list"),
};

const turns = [];
let selected = -1;
let busy = false;
let sessionId = loadSession();

// ---- Session ---------------------------------------------------------------

function freshId() {
  return crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function loadSession() {
  try {
    const saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) return saved;
  } catch { /* storage unavailable: a new session per page load is fine */ }
  return saveSession(freshId());
}

function saveSession(id) {
  try { sessionStorage.setItem(SESSION_KEY, id); } catch { /* ignore */ }
  return id;
}

// ---- Markdown (escaped first, so model output can never inject HTML) -------

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(text) {
  const codes = [];
  let html = escapeHtml(text).replace(/`([^`]+)`/g, (_, code) => `\u0000${codes.push(code) - 1}\u0000`);
  html = html
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_([^_\s][^_]*?)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\[(\d{1,3})\]/g, '<a class="cite" href="#source-$1" data-n="$1" aria-label="Source $1">$1</a>');
  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const isTableRow = (line) => /^\s*\|/.test(line);
const isRule = (line) => /^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line);
const isHeading = (line) => /^#{1,4}\s/.test(line);
const startsBlock = (line) => isHeading(line) || isRule(line) || isTableRow(line) || LIST_ITEM.test(line);
const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

function renderList(items) {
  let html = "";
  const stack = [];
  for (const item of items) {
    while (stack.length && item.indent < stack[stack.length - 1].indent) html += `</li></${stack.pop().tag}>`;
    const top = stack[stack.length - 1];
    if (!top || item.indent > top.indent) {
      const tag = item.ordered ? "ol" : "ul";
      stack.push({ indent: item.indent, tag });
      html += `<${tag}><li>`;
    } else {
      html += "</li><li>";
    }
    html += inline(item.text);
  }
  while (stack.length) html += `</li></${stack.pop().tag}>`;
  return html;
}

function markdown(source) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    if (isHeading(line)) {
      const [, hashes, text] = line.match(/^(#{1,4})\s+(.*)$/);
      const level = Math.min(hashes.length + 1, 4);
      out.push(`<h${level}>${inline(text)}</h${level}>`);
      i += 1;
    } else if (isRule(line)) {
      out.push("<hr>");
      i += 1;
    } else if (isTableRow(line)) {
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) rows.push(lines[i++]);
      const body = rows.filter((row, n) => !(n === 1 && /^[\s|:-]+$/.test(row)));
      const [head, ...rest] = body;
      out.push(`<table><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${
        rest.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    } else if (LIST_ITEM.test(line)) {
      const items = [];
      while (i < lines.length) {
        const match = lines[i].match(LIST_ITEM);
        if (match) {
          items.push({ indent: match[1].replace(/\t/g, "    ").length, ordered: /\d/.test(match[2]), text: match[3] });
        } else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
          items[items.length - 1].text += ` ${lines[i].trim()}`;
        } else {
          break;
        }
        i += 1;
      }
      out.push(renderList(items));
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !(para.length && startsBlock(lines[i]))) para.push(lines[i++]);
      out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    }
  }
  return out.join("");
}

// ---- Server-Sent Events over fetch -----------------------------------------

function parseFrame(raw) {
  let event = "message";
  const data = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

async function streamChat(body, onFrame) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(response.status === 422
      ? "The question must be between 1 and 2,000 characters."
      : `The server returned ${response.status}.`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = parseFrame(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (frame) onFrame(frame);
    }
  }
}

// ---- Turns -----------------------------------------------------------------

function seconds(ms) {
  return ms == null ? "" : `${(ms / 1000).toFixed(1)} s`;
}

function statusFor(event) {
  switch (event.type) {
    case "llm_route":
      if (event.agent === "supervisor" && event.trigger) return `Also asking ${PROVIDER_NAMES[event.provider]} to choose agents`;
      return event.agent === "supervisor"
        ? `Choosing agents with ${PROVIDER_NAMES[event.provider] || event.provider}`
        : `${AGENT_NAMES[event.agent] || event.agent} agent is working on ${PROVIDER_NAMES[event.provider] || event.provider}`;
    case "rag_retrieval":
      if (event.query.startsWith("read chapter")) return "Combining the section digests";
      return event.query.startsWith("read ") ? "Reading the textbook section" : "Searching the textbook";
    case "llm_call": return event.stage === "digest" ? `Read and condensed ${event.section}` : null;
    case "mcp_tool_call": return event.tool.includes("extract") ? "Reading a web page" : "Searching the web";
    case "sql_query": return "Querying the patient database";
    case "answer_reset": return "Retrying on the other provider";
    case "error": return `${SHORT_PROVIDER[event.provider] || "A provider"} failed; switching providers`;
    default: return null;
  }
}

function createTurn(question) {
  const turn = {
    question, text: "", events: [], citations: [], done: null, error: null,
    started: performance.now(), renderQueued: false,
  };
  const item = document.createElement("li");
  item.className = "turn";
  item.innerHTML = `
    <p class="question"></p>
    <p class="status" role="status">Starting</p>
    <div class="answer" aria-busy="true"></div>
    <p class="turn-meta" hidden></p>`;
  item.querySelector(".question").textContent = question;
  turn.node = item;
  turn.statusNode = item.querySelector(".status");
  turn.answerNode = item.querySelector(".answer");
  turn.metaNode = item.querySelector(".turn-meta");
  item.addEventListener("click", (event) => {
    const cite = event.target.closest(".cite");
    if (cite) {
      event.preventDefault();
      select(turns.indexOf(turn));
      flashSource(cite.dataset.n);
    }
  });
  el.turns.append(item);
  turns.push(turn);
  return turn;
}

// Sources are numbered as agents gather them, so an answer may cite only [9]-[14]. Readers see
// 1, 2, 3... in order of first appearance; the source cards use the same labels.
function citationLabels(turn) {
  const known = turn.done ? new Set(turn.citations.map((c) => String(c.n))) : null;
  const labels = new Map();
  for (const [, n] of turn.text.matchAll(/\[(\d{1,3})\]/g)) {
    if (!labels.has(n) && (!known || known.has(n))) labels.set(n, labels.size + 1);
  }
  return labels;
}

function renderAnswer(turn) {
  turn.renderQueued = false;
  const labels = citationLabels(turn);
  const text = turn.text
    .replace(/\[(\d{1,3})\]/g, (match, n) => (labels.has(n) ? `[${labels.get(n)}]` : match))
    .replace(/(?:\[\d{1,3}\]){2,}/g, (run) => run.match(/\d+/g).map(Number).sort((a, b) => a - b).map((n) => `[${n}]`).join(""));
  turn.answerNode.innerHTML = markdown(text);
}

function queueAnswerRender(turn) {
  if (turn.renderQueued) return;
  turn.renderQueued = true;
  requestAnimationFrame(() => renderAnswer(turn));
}

function finishTurn(turn) {
  turn.statusNode.remove();
  turn.answerNode.removeAttribute("aria-busy");
  if (turn.error && !turn.text) {
    turn.answerNode.classList.add("is-error");
    turn.answerNode.textContent = turn.error;
  } else {
    renderAnswer(turn);
  }
  if (turn.done) {
    turn.metaNode.hidden = false;
    turn.metaNode.innerHTML = `<span>First words in ${seconds(turn.done.ttft_ms)}</span>
      <span>Finished in ${seconds(turn.done.total_ms)}</span>
      <button type="button" class="show-how">Show how it was answered</button>`;
    turn.metaNode.querySelector(".show-how").addEventListener("click", () => select(turns.indexOf(turn)));
  }
}

function handleFrame(turn, { event, data }) {
  if (event === "token") {
    turn.statusNode.hidden = true;
    turn.text += data.text;
    queueAnswerRender(turn);
  } else if (event === "trace") {
    turn.events.push(data);
    if (data.type === "answer_reset") {
      turn.text = "";
      queueAnswerRender(turn);
      turn.statusNode.hidden = false;
    }
    const status = statusFor(data);
    if (status) turn.statusNode.textContent = status;
  } else if (event === "final") {
    turn.text = data.text;
    queueAnswerRender(turn);
  } else if (event === "citation") {
    turn.citations.push(data);
  } else if (event === "error") {
    turn.error = data.message;
  } else if (event === "done") {
    turn.done = data;
    queueAnswerRender(turn);  // final citation labels: only numbers that have a source
  }
  if (turns.indexOf(turn) === selected && event !== "token") renderInspector();
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  el.submit.disabled = true;
  el.empty.hidden = true;
  el.query.value = "";
  const turn = createTurn(question.trim());
  select(turns.length - 1);
  turn.node.scrollIntoView({ block: "nearest" });
  try {
    await streamChat({ query: turn.question, session_id: sessionId }, (frame) => handleFrame(turn, frame));
  } catch (error) {
    turn.error = error instanceof TypeError
      ? "Couldn't reach the server. Check that it's running, then try again."
      : error.message;
  } finally {
    finishTurn(turn);
    renderInspector();
    busy = false;
    el.submit.disabled = false;
    el.query.focus();
  }
}

// ---- Inspector: route, timeline, sources -----------------------------------

function select(index) {
  selected = index;
  turns.forEach((turn, i) => turn.node.classList.toggle("is-selected", i === index));
  renderInspector();
}

function routeSteps(turn) {
  const steps = [];
  const latest = (event) => [...steps].reverse().find((s) => s.agent === event.agent && (!event.provider || s.provider === event.provider));
  for (const event of turn.events) {
    if (event.type === "llm_route") steps.push({ ...event, failed: false, cancelled: false });
    if (event.type === "error") Object.assign(latest(event) || {}, { failed: true, error: event.message });
    if (event.type === "llm_cancelled") Object.assign(latest(event) || {}, { cancelled: true, error: event.reason, ms: event.ms });
  }
  return steps;
}

function stepNote(step) {
  if (step.failed) return `Failed: ${step.error}.`;
  if (step.cancelled) return `Not used: ${step.error}, so this call was cancelled after ${seconds(step.ms)}.`;
  return step.trigger ? `Started because ${step.trigger}.` : step.reason;
}

function renderRoute(turn) {
  const plan = turn.events.find((e) => e.type === "agent_plan");
  el.routeReason.textContent = plan ? plan.reason : "";
  el.routeReason.hidden = !plan;
  el.routeSteps.replaceChildren(...routeSteps(turn).map((step) => {
    const li = document.createElement("li");
    li.className = `route-step${step.fallback ? " is-fallback" : ""}${step.failed ? " is-failed" : ""}${step.cancelled ? " is-cancelled" : ""}`;
    li.dataset.provider = step.provider;
    li.innerHTML = `<span class="route-head"><span class="route-agent"></span><span class="route-tag"></span></span>
      <span class="route-model"></span><span class="route-why"></span>`;
    li.querySelector(".route-agent").textContent =
      `${AGENT_NAMES[step.agent] || step.agent} on ${PROVIDER_NAMES[step.provider] || step.provider}`;
    li.querySelector(".route-tag").textContent = step.fallback ? "Fallback" : `Routed by ${BASIS[step.dimension] || step.dimension}`;
    li.querySelector(".route-model").textContent = step.model;
    li.querySelector(".route-why").textContent = stepNote(step);
    return li;
  }));
}

function niceStep(total) {
  const rough = total / 4;
  return [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000].find((s) => s >= rough) || 60000;
}

function detailList(pairs) {
  const dl = document.createElement("dl");
  for (const [term, value, code] of pairs) {
    if (value == null || value === "") continue;
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    if (code) {
      const pre = document.createElement("pre");
      pre.textContent = value;
      dd.append(pre);
    } else {
      dd.textContent = value;
    }
    dl.append(dt, dd);
  }
  return dl;
}

function tokenSummary(e) {
  if (e.input_tokens == null && e.output_chars == null) return null;
  const parts = [];
  if (e.input_tokens != null) parts.push(`${e.input_tokens.toLocaleString()} in`);
  if (e.output_tokens != null) parts.push(`${e.output_tokens.toLocaleString()} out`);
  const tokens = parts.length ? `${parts.join(", ")} tokens` : null;
  const chars = e.output_chars != null ? `${e.output_chars.toLocaleString()} characters written` : null;
  return [tokens, chars].filter(Boolean).join("; ");
}

function timelineRows(turn) {
  const rows = [];
  for (const e of turn.events) {
    const end = e.t_ms;
    const start = e.ms != null ? Math.max(0, end - e.ms) : end;
    if (e.type === "agent_plan") {
      const provider = e.planner.split(" ")[0];
      rows.push({ label: `Plan on ${SHORT_PROVIDER[provider]}`, start, end, provider,
        detail: [["Agents", e.plan.map((a) => AGENT_NAMES[a] || a).join(", ") || "none (direct reply)"],
                 ["Reason", e.reason], ["Read as", e.standalone],
                 ["Router", e.hedged ? `${PROVIDER_NAMES[provider]}, first of two routers to answer` : PROVIDER_NAMES[provider]],
                 ["Usage", tokenSummary(e)], ["Took", `${e.ms} ms`]] });
    } else if (e.type === "llm_cancelled") {
      rows.push({ label: `${SHORT_PROVIDER[e.provider]} call cancelled`, start, end, provider: e.provider, cancelled: true,
        detail: [["Outcome", `Cancelled: ${e.reason}`], ["Model", e.model, true], ["Ran for", `${e.ms} ms`]] });
    } else if (e.type === "llm_call" && e.stage === "digest") {
      rows.push({ label: `${AGENT_NAMES[e.agent]}: ${e.section}`, start, end, provider: e.provider,
        detail: [["Step", `Reading ${e.section} in full and condensing it for the chapter summary`],
                 ["Provider", PROVIDER_NAMES[e.provider]], ["Model", e.model, true], ["Usage", tokenSummary(e)],
                 ["Took", `${e.ms} ms`]] });
    } else if (e.type === "llm_call") {
      const doing = { tools: "tools", answer: "answer", intermediate: "draft" }[e.stage] || e.stage;
      rows.push({ label: `${AGENT_NAMES[e.agent] || e.agent}: ${doing}`, start, end, provider: e.provider,
        detail: [["Step", { tools: "Choosing which tools to call", answer: "Writing the answer",
                            intermediate: "Drafting findings for the next agent" }[e.stage] || e.stage],
                 ["Provider", PROVIDER_NAMES[e.provider]], ["Model", e.model, true],
                 ["Tool calls requested", e.stage === "tools" ? String(e.tool_calls) : null], ["Usage", tokenSummary(e)],
                 ["Took", `${e.ms} ms`]] });
    } else if (e.type === "llm_route" && e.agent !== "supervisor") {
      rows.push({ label: `${AGENT_NAMES[e.agent]} starts`, start, end, provider: e.provider, tick: true,
        detail: [["Provider", PROVIDER_NAMES[e.provider]], ["Model", e.model, true],
                 ["Routing basis", BASIS[e.dimension] || e.dimension], ["Why", e.reason]] });
    } else if (e.type === "rag_retrieval") {
      const reading = e.query.startsWith("read ");
      const chapter = e.query.startsWith("read chapter");
      rows.push({ label: chapter ? "Read chapter" : reading ? "Read section" : "Textbook search", start, end,
        detail: [["Agent", AGENT_NAMES[e.agent]],
                 [chapter ? "Chapter" : reading ? "Section" : "Query", chapter ? e.query.slice(13) : reading ? e.query.slice(5) : e.query],
                 ["Results", e.results.map((r) => r.chunk_id || r.source_id).join(", ")], ["Took", `${e.ms} ms`]] });
    } else if (e.type === "mcp_tool_call") {
      rows.push({ label: e.tool.includes("extract") ? "Web page (MCP)" : "Web search (MCP)", start, end,
        error: e.status !== "ok",
        detail: [["Agent", AGENT_NAMES[e.agent]], ["MCP server", e.server], ["Tool", e.tool, true],
                 ["Arguments", JSON.stringify(e.args), true], ["Status", e.status], ["Results", String(e.results)],
                 ["Took", `${e.ms} ms`]] });
    } else if (e.type === "sql_query") {
      rows.push({ label: "SQL query", start, end, error: e.status !== "ok",
        detail: [["Query", e.sql, true], ["Status", e.status], ["Rows", String(e.rows)], ["Took", `${e.ms} ms`]] });
    } else if (e.type === "error") {
      const what = e.agent === "supervisor" ? "Plan" : AGENT_NAMES[e.agent] || e.agent;
      rows.push({ label: `${what} on ${SHORT_PROVIDER[e.provider] || e.provider} failed`, start, end,
        tick: e.ms == null, error: true, provider: e.provider,
        detail: [["Provider", PROVIDER_NAMES[e.provider]], ["What happened", e.message],
                 ["Raw error", e.detail, true], ["Failed after", e.ms != null ? `${e.ms} ms` : null]] });
    } else if (e.type === "answer_reset") {
      rows.push({ label: "Answer restarted", start, end, tick: true, error: true, detail: [["Why", e.reason]] });
    }
  }
  return rows;
}

function renderTimeline(turn) {
  const live = Math.round(performance.now() - turn.started);
  const total = Math.max(turn.done ? turn.done.total_ms : live, ...turn.events.map((e) => e.t_ms || 0), 1);
  const pct = (ms) => `${Math.min(100, (ms / total) * 100).toFixed(2)}%`;
  const step = niceStep(total);

  const axis = document.createElement("div");
  axis.className = "axis";
  axis.setAttribute("aria-hidden", "true");
  for (let t = 0; t <= total; t += step) {
    const mark = document.createElement("span");
    mark.style.left = pct(t);
    mark.textContent = t === 0 ? "0" : `${t / 1000} s`;
    if (t > 0 && t / total > 0.9) mark.classList.add("is-last");
    axis.append(mark);
  }

  const ttft = turn.done && turn.done.ttft_ms != null ? pct(turn.done.ttft_ms) : "-10%";
  const rows = timelineRows(turn).map((row) => {
    const details = document.createElement("details");
    details.className = `wf-row${row.error ? " is-error" : ""}${row.cancelled ? " is-cancelled" : ""}`;
    const summary = document.createElement("summary");
    const label = document.createElement("span");
    label.className = "wf-label";
    label.textContent = row.label;
    const track = document.createElement("span");
    track.className = "wf-track";
    track.style.setProperty("--ttft", ttft);
    const mark = document.createElement("span");
    mark.className = row.tick ? "wf-tick" : "wf-bar";
    mark.style.left = pct(row.start);
    if (!row.tick) mark.style.width = pct(Math.max(row.end - row.start, 0));
    if (row.provider) mark.dataset.provider = row.provider;
    const time = document.createElement("span");
    time.className = "wf-time";
    const duration = row.end - row.start;
    time.textContent = row.tick ? seconds(row.end) : duration >= 1000 ? seconds(duration) : `${duration} ms`;
    time.style.left = `calc(${pct(row.end)} + 6px)`;
    if (row.end / total > 0.8) { time.style.left = "auto"; time.style.right = `calc(${(100 - (row.start / total) * 100).toFixed(2)}% + 6px)`; }
    track.append(mark, time);
    summary.append(label, track);
    const detail = document.createElement("div");
    detail.className = "wf-detail";
    detail.append(detailList(row.detail));
    details.append(summary, detail);
    return details;
  });

  const note = document.createElement("p");
  note.className = "wf-summary";
  note.textContent = turn.done
    ? `Dashed line: first words at ${seconds(turn.done.ttft_ms)}. Select a row for details.`
    : "Running. Select a row for details.";
  el.waterfall.replaceChildren(axis, ...rows, note);
}

function renderSources(turn) {
  const labels = citationLabels(turn);
  const label = (source) => labels.get(String(source.n)) ?? source.n;
  const ordered = [...turn.citations].sort((a, b) => label(a) - label(b));
  el.sourceList.replaceChildren(...ordered.map((source) => {
    const li = document.createElement("li");
    li.className = "source";
    li.id = `source-${label(source)}`;
    li.dataset.n = label(source);
    const kind = { textbook: "Textbook", web: "Web page", sql: "Database query" }[source.kind] || source.kind;
    li.innerHTML = `<span class="source-n"></span><div><div class="source-title"></div><div class="source-kind"></div></div>
      <p class="source-snippet"></p>`;
    li.querySelector(".source-n").textContent = label(source);
    const title = li.querySelector(".source-title");
    if (source.url) {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title;
      title.append(link);
    } else {
      title.textContent = source.title;
    }
    li.querySelector(".source-kind").textContent = source.kind === "textbook" ? `${kind}, ${source.key}` : kind;
    const snippet = li.querySelector(".source-snippet");
    if (source.kind === "sql") {
      snippet.classList.add("is-sql");
      snippet.textContent = source.key;
    } else {
      snippet.textContent = source.snippet.replace(/\s+/g, " ").slice(0, 220) + (source.snippet.length > 220 ? "…" : "");
    }
    return li;
  }));
}

function renderInspector() {
  const turn = turns[selected];
  el.inspectorEmpty.hidden = Boolean(turn);
  el.route.hidden = el.timeline.hidden = !turn;
  el.sources.hidden = !turn || !turn.citations.length;
  if (!turn) return;
  renderRoute(turn);
  renderTimeline(turn);
  renderSources(turn);
}

function flashSource(n) {
  const node = el.sourceList.querySelector(`[data-n="${CSS.escape(String(n))}"]`);
  if (!node) return;
  node.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  node.classList.add("is-flashed");
  setTimeout(() => node.classList.remove("is-flashed"), 1400);
}

// ---- Wiring ----------------------------------------------------------------

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(el.query.value);
});

el.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.form.requestSubmit();
  }
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.query));
});

el.reset.addEventListener("click", () => {
  if (busy) return;
  sessionId = saveSession(freshId());
  turns.length = 0;
  selected = -1;
  el.turns.replaceChildren();
  el.empty.hidden = false;
  renderInspector();
  el.query.focus();
});
