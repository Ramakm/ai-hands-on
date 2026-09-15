"use strict";

const MODELS = ["rnn", "lstm"];
const LABEL = { rnn: "RNN", lstm: "LSTM" };
const T = 8, D = 8;

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const zeros = () => Array.from({ length: T }, () => Array(D).fill(0));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const pct = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  pixels: zeros(),
  label: null,        // true digit when the input came from the test set
  source: "",
  mode: "draw",
  meta: null,
  compare: null,
  last: null,
};

// ------------------------------------------------------------------ API
async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

function showError(message) {
  const el = $("#error");
  el.textContent = message;
  el.hidden = !message;
}

// ------------------------------------------------------------------ tooltip
const tip = $("#tip");
function showTip(text, x, y) {
  tip.textContent = text;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${clamp(x + 14, 8, window.innerWidth - w - 8)}px`;
  tip.style.top = `${clamp(y - h - 10, 8, window.innerHeight - h - 8)}px`;
}
const hideTip = () => { tip.hidden = true; };

// ------------------------------------------------------------------ drawing pad
const pad = $("#pad");
const padCtx = pad.getContext("2d", { willReadFrequently: true });
const PAD = pad.width;
const INK = "17,17,17";
const INK_GAIN = 1.6;   // thin strokes -> intensities comparable to the 0..16 training scans

const work = document.createElement("canvas");
work.width = work.height = 64;
const workCtx = work.getContext("2d", { willReadFrequently: true });

let drawing = false, lastPt = null, syncQueued = false;

function padPoint(e) {
  const r = pad.getBoundingClientRect();
  return { x: (e.clientX - r.left) * PAD / r.width, y: (e.clientY - r.top) * PAD / r.height };
}

function strokeTo(a, b) {
  padCtx.globalCompositeOperation = state.mode === "erase" ? "destination-out" : "source-over";
  padCtx.strokeStyle = `rgb(${INK})`;
  padCtx.lineWidth = state.mode === "erase" ? 34 : 22;
  padCtx.lineCap = "round";
  padCtx.beginPath();
  padCtx.moveTo(a.x, a.y);
  padCtx.lineTo(b.x, b.y);
  padCtx.stroke();
  padCtx.globalCompositeOperation = "source-over";
  if (!syncQueued) {
    syncQueued = true;
    requestAnimationFrame(() => { syncQueued = false; syncFromPad(); });
  }
}

pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  state.label = null;
  state.source = "hand-drawn";
  lastPt = padPoint(e);
  strokeTo(lastPt, lastPt);
});
pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = padPoint(e);
  strokeTo(lastPt, p);
  lastPt = p;
});
for (const ev of ["pointerup", "pointercancel"]) {
  pad.addEventListener(ev, () => { drawing = false; });
}

/** Downsample the pad to the 8x8 grid the models were trained on. */
function syncFromPad() {
  const { data } = padCtx.getImageData(0, 0, PAD, PAD);
  let x0 = PAD, y0 = PAD, x1 = -1, y1 = -1;
  for (let y = 0; y < PAD; y++) {
    for (let x = 0; x < PAD; x++) {
      if (data[(y * PAD + x) * 4 + 3] > 16) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }

  workCtx.clearRect(0, 0, 64, 64);
  if (x1 < 0) {
    state.pixels = zeros();
  } else {
    if ($("#center").checked) {
      const w = x1 - x0 + 1, h = y1 - y0 + 1;
      const s = 56 / Math.max(w, h);
      workCtx.drawImage(pad, x0, y0, w, h, (64 - w * s) / 2, (64 - h * s) / 2, w * s, h * s);
    } else {
      workCtx.drawImage(pad, 0, 0, PAD, PAD, 0, 0, 64, 64);
    }
    const d = workCtx.getImageData(0, 0, 64, 64).data;
    state.pixels = Array.from({ length: T }, (_, r) => Array.from({ length: D }, (_, c) => {
      let sum = 0;
      for (let y = r * 8; y < r * 8 + 8; y++) {
        for (let x = c * 8; x < c * 8 + 8; x++) sum += d[(y * 64 + x) * 4 + 3];
      }
      return Math.round(Math.min(1, (sum / (64 * 255)) * INK_GAIN) * 1e4) / 1e4;
    }));
  }
  inputChanged();
}

/** Paint an 8x8 grid onto the pad (inverse of syncFromPad without centering). */
function paintPad(pixels) {
  padCtx.clearRect(0, 0, PAD, PAD);
  const cell = PAD / 8;
  pixels.forEach((row, r) => row.forEach((v, c) => {
    if (v <= 0) return;
    padCtx.fillStyle = `rgba(${INK},${Math.min(1, v / INK_GAIN)})`;
    padCtx.fillRect(c * cell, r * cell, cell, cell);
  }));
}

function paintGrid(canvas, pixels) {
  const ctx = canvas.getContext("2d");
  const cell = canvas.width / 8;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  pixels.forEach((row, r) => row.forEach((v, c) => {
    ctx.fillStyle = `rgba(${INK},${v})`;
    ctx.fillRect(c * cell, r * cell, cell, cell);
  }));
}

const isBlank = () => state.pixels.every((row) => row.every((v) => v === 0));

function inputChanged() {
  paintGrid($("#preview"), state.pixels);
  $("#pad-hint").hidden = !isBlank();
  $("#input-source").textContent = state.label !== null
    ? `${state.source} · true digit ${state.label}` : state.source;
  schedulePredict();
}

function loadPixels(pixels, label, source) {
  state.pixels = pixels.map((row) => row.slice());
  state.label = label;
  state.source = source;
  paintPad(state.pixels);
  inputChanged();
}

// ------------------------------------------------------------------ controls
$$(".segmented button").forEach((btn) => btn.addEventListener("click", () => {
  state.mode = btn.dataset.mode;
  $$(".segmented button").forEach((b) => b.classList.toggle("active", b === btn));
}));

$("#clear").addEventListener("click", () => loadPixels(zeros(), null, ""));

$("#random").addEventListener("click", loadRandom);
async function loadRandom() {
  try {
    const digit = $("#digit").value;
    const s = await api(`/api/samples/random${digit ? `?digit=${digit}` : ""}`);
    loadPixels(s.pixels, s.label, `test #${s.index}`);
  } catch (e) { showError(e.message); }
}

$$("[data-shift]").forEach((btn) => btn.addEventListener("click", () => {
  const [dr, dc] = btn.dataset.shift.split(",").map(Number);
  const next = zeros();
  for (let r = 0; r < T; r++) for (let c = 0; c < D; c++) {
    const sr = r - dr, sc = c - dc;
    if (sr >= 0 && sr < T && sc >= 0 && sc < D) next[r][c] = state.pixels[sr][sc];
  }
  loadPixels(next, state.label, withTag(state.source, "shifted"));
}));

$("#noise").addEventListener("click", () => {
  const next = state.pixels.map((row) => row.map((v) =>
    Math.round(clamp(v + (Math.random() - 0.5) * 0.3, 0, 1) * 1e4) / 1e4));
  loadPixels(next, state.label, withTag(state.source, "noisy"));
});

$("#flip").addEventListener("click", () => {
  // Reversed time is a different input, so the original label no longer applies.
  loadPixels(state.pixels.slice().reverse(), null, withTag(state.source, "rows reversed"));
});

$("#center").addEventListener("change", () => { if (state.source === "hand-drawn") syncFromPad(); });

function withTag(source, tag) {
  if (!source) return tag;
  return source.includes(tag) ? source : `${source} · ${tag}`;
}

// ------------------------------------------------------------------ prediction
let predictTimer = null, predictSeq = 0;
function schedulePredict() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(predict, 80);
}

async function predict() {
  const seq = ++predictSeq;
  if (isBlank()) {
    state.last = null;
    renderResults();
    return;
  }
  try {
    const body = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: state.pixels }),
    });
    if (seq !== predictSeq) return;          // a newer request superseded this one
    state.last = body;
    showError("");
    renderResults();
  } catch (e) { showError(e.message); }
}

// ------------------------------------------------------------------ results
function renderResults() {
  const res = state.last?.results;
  const banner = $("#banner");
  const truth = state.label !== null ? `<span class="tag">true digit ${state.label}</span>` : "";

  if (!res) {
    banner.innerHTML = "Load a test digit or draw one.";
  } else if (res.rnn && res.lstm) {
    banner.innerHTML = state.last.agree
      ? `Both models predict <b>${res.rnn.prediction}</b>${truth}`
      : `The models disagree: RNN says <b>${res.rnn.prediction}</b>, LSTM says <b>${res.lstm.prediction}</b>${truth}`;
  } else {
    banner.innerHTML = Object.values(res).map((r) => `${LABEL[r.model]} predicts <b>${r.prediction}</b>`).join(" · ") + truth;
  }

  for (const name of MODELS) renderCard(name, res?.[name]);
  renderStepsChart();
}

function cardHead(name) {
  const m = state.meta?.[name];
  const info = m ? `${m.parameters.toLocaleString()} params · test acc ${pct(m.metrics.test_acc)}` : "not loaded";
  return `<header class="card-head"><span class="sw"></span><h2>${LABEL[name]}</h2><span class="muted small">${info}</span></header>`;
}

function renderCard(name, r) {
  const el = $(`#card-${name}`);
  if (!r) {
    const msg = state.meta && !state.meta[name] ? "Model not loaded." : "Waiting for input.";
    el.innerHTML = `${cardHead(name)}<div class="empty">${msg}</div>`;
    return;
  }

  const label = state.label;
  let status = "";
  if (label !== null) {
    status = r.prediction === label
      ? `<div class="status good">✓ correct</div>`
      : `<div class="status bad">✗ wrong, true digit is ${label}</div>`;
  }

  const bars = r.probabilities.map((p, d) => `
    <div class="bar-row${d === r.prediction ? " is-top" : ""}">
      <span class="bar-digit">${d}${d === label ? "<small>true</small>" : ""}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(2)}%"></span></span>
      <span class="bar-val">${pct(p)}</span>
    </div>`).join("");

  const strip = r.steps.map((s) => `
    <div class="step" title="After reading row ${s.row}: predicts ${s.prediction} (${pct(s.confidence)})">
      <span class="step-row">t${s.row}</span>
      <span class="step-digit">${s.prediction}</span>
      <span class="step-meter"><span style="height:${(s.confidence * 100).toFixed(1)}%"></span></span>
      <span class="step-conf">${Math.round(s.confidence * 100)}</span>
    </div>`).join("");

  el.innerHTML = `
    ${cardHead(name)}
    <div class="verdict">
      <div class="big">${r.prediction}</div>
      <div>
        <div class="conf">${pct(r.confidence)}</div>
        <div class="muted small">confidence · ${r.latency_ms.toFixed(2)} ms</div>
        ${status}
      </div>
    </div>
    <div class="bars">${bars}</div>
    <h3>Prediction after each row</h3>
    <div class="strip">${strip}</div>
    <h3>${name === "lstm" ? "Gates &amp; memory per row (mean over units)" : "Memory per row"}</h3>
    ${internalsTable(r)}
    <h3>Hidden state h<sub>t</sub> (rows × ${r.hidden[0].length} units)</h3>
    <div class="heat-wrap"><canvas class="heat"></canvas></div>
    <div class="div-legend">−1 <i></i> +1</div>`;

  drawHeatmap($("canvas.heat", el), r.hidden);
}

function internalsTable(r) {
  const head = `<thead><tr><th></th>${r.steps.map((s) => `<th>t${s.row}</th>`).join("")}</tr></thead>`;
  const shade = (v) => `background:color-mix(in srgb, var(--series) ${Math.round(clamp(v, 0, 1) * 75)}%, transparent)`;
  const row = (title, values, scale = 1, fmt = (v) => v.toFixed(2)) =>
    `<tr><th>${title}</th>${values.map((v) => `<td style="${shade(v / scale)}">${fmt(v)}</td>`).join("")}</tr>`;
  const maxOf = (vs) => Math.max(1e-6, ...vs);

  let rows = "";
  if (r.gates) {
    rows += row("input i", r.gates.input);
    rows += row("forget f", r.gates.forget);
    rows += row("output o", r.gates.output);
    rows += row("‖c<sub>t</sub>‖", r.cell_norm, maxOf(r.cell_norm), (v) => v.toFixed(1));
  }
  rows += row("‖h<sub>t</sub>‖", r.hidden_norm, maxOf(r.hidden_norm), (v) => v.toFixed(1));
  const note = r.gates ? "" :
    `<p class="note">No gates: h<sub>t</sub> = tanh(x<sub>t</sub>W<sub>xh</sub> + h<sub>t−1</sub>W<sub>hh</sub> + b). Every row rewrites the whole memory through one squashing matrix.</p>`;
  return `<table class="internals">${head}<tbody>${rows}</tbody></table>${note}`;
}

function hexToRgb(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function drawHeatmap(canvas, hidden) {
  const rows = hidden.length, cols = hidden[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  const neg = hexToRgb(cssVar("--div-neg")), mid = hexToRgb(cssVar("--div-mid")), pos = hexToRgb(cssVar("--div-pos"));
  hidden.forEach((row, t) => row.forEach((v, j) => {
    const a = Math.min(1, Math.abs(v)), end = v < 0 ? neg : pos, k = (t * cols + j) * 4;
    for (let ch = 0; ch < 3; ch++) img.data[k + ch] = Math.round(mid[ch] + (end[ch] - mid[ch]) * a);
    img.data[k + 3] = 255;
  }));
  ctx.putImageData(img, 0, 0);

  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const j = clamp(Math.floor((e.clientX - rect.left) / rect.width * cols), 0, cols - 1);
    const t = clamp(Math.floor((e.clientY - rect.top) / rect.height * rows), 0, rows - 1);
    showTip(`row t${t + 1} · unit ${j}\nh = ${hidden[t][j].toFixed(3)}`, e.clientX, e.clientY);
  };
  canvas.onmouseleave = hideTip;
}

// ------------------------------------------------------------------ line chart
function niceTicks(lo, hi, count = 4) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

/**
 * series: [{ key, name, color, values }], xs: x values shared by all series.
 * One y-axis, recessive grid, direct end labels, crosshair tooltip.
 */
function lineChart(container, { xs, series, yMin, yMax, yFormat, xFormat = String, xTicks, height = 220 }) {
  const width = Math.max(260, container.clientWidth);
  const m = { t: 10, r: 52, b: 26, l: 44 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const all = series.flatMap((s) => s.values);
  const lo = yMin ?? Math.min(...all), hi = yMax ?? Math.max(...all);
  const yTicks = niceTicks(lo, hi);
  const y0 = Math.min(lo, yTicks[0]), y1 = Math.max(hi, yTicks[yTicks.length - 1]);
  const xOf = (i) => m.l + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const yOf = (v) => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;

  const grid = yTicks.map((v) => `
    <line class="gridline" x1="${m.l}" x2="${m.l + iw}" y1="${yOf(v)}" y2="${yOf(v)}"/>
    <text class="axis-text" x="${m.l - 8}" y="${yOf(v) + 4}" text-anchor="end">${yFormat(v)}</text>`).join("");
  const xt = (xTicks ?? xs.map((_, i) => i)).map((i) =>
    `<text class="axis-text" x="${xOf(i)}" y="${height - 6}" text-anchor="middle">${xFormat(xs[i])}</text>`).join("");
  const lines = series.map((s) =>
    `<path class="series" stroke="${s.color}" d="${s.values.map((v, i) => `${i ? "L" : "M"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join("")}"/>`).join("");

  // direct labels at the line ends, nudged apart when they would collide
  const ends = series.map((s) => ({ name: s.name, y: yOf(s.values[s.values.length - 1]) })).sort((a, b) => a.y - b.y);
  const GAP = 13, bottom = m.t + ih - 6;
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + GAP);
  for (let i = ends.length - 1; i >= 0; i--) {         // keep clear of the x-axis labels
    const limit = i === ends.length - 1 ? bottom : ends[i + 1].y - GAP;
    ends[i].y = Math.min(ends[i].y, limit);
  }
  const labels = ends.map((e) => `<text class="end-label" x="${m.l + iw + 8}" y="${e.y + 4}">${e.name}</text>`).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" height="${height}" role="img">
      ${grid}${xt}${lines}${labels}
      <g class="hover" visibility="hidden">
        <line class="crosshair" y1="${m.t}" y2="${m.t + ih}"/>
        ${series.map((s) => `<circle class="dot" r="4.5" fill="${s.color}"/>`).join("")}
      </g>
      <rect x="${m.l}" y="0" width="${iw}" height="${height}" fill="transparent"/>
    </svg>`;

  const svg = $("svg", container), hover = $(".hover", svg), rect = $("rect", svg);
  rect.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = (e.clientX - box.left) * (width / box.width);
    const i = clamp(Math.round(((px - m.l) / iw) * (xs.length - 1)), 0, xs.length - 1);
    hover.setAttribute("visibility", "visible");
    $("line", hover).setAttribute("x1", xOf(i));
    $("line", hover).setAttribute("x2", xOf(i));
    $$("circle", hover).forEach((c, k) => {
      c.setAttribute("cx", xOf(i));
      c.setAttribute("cy", yOf(series[k].values[i]));
    });
    showTip([xFormat(xs[i]), ...series.map((s) => `${s.name}  ${yFormat(s.values[i], true)}`)].join("\n"), e.clientX, e.clientY);
  });
  rect.addEventListener("mouseleave", () => { hover.setAttribute("visibility", "hidden"); hideTip(); });
}

function dataTable(container, header, rows) {
  container.innerHTML = `<table><thead><tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

const colorOf = (name) => cssVar(`--${name}`);

function renderStepsChart() {
  const res = state.last?.results;
  const el = $("#steps-chart");
  if (!res) {
    el.innerHTML = `<div class="empty">No input yet.</div>`;
    $("#steps-caption").textContent = "";
    $("#steps-table").innerHTML = "";
    return;
  }
  const names = MODELS.filter((n) => res[n]);
  const known = state.label !== null;
  const target = (n) => (known ? state.label : res[n].prediction);
  const series = names.map((n) => ({
    name: LABEL[n], color: colorOf(n),
    values: res[n].step_probabilities.map((p) => p[target(n)]),
  }));
  $("#steps-caption").textContent = known
    ? `Probability each model gives the true digit ${state.label} if the image stopped after row t.`
    : "Probability each model gives its own final answer if the image stopped after row t.";
  const xs = Array.from({ length: T }, (_, i) => i + 1);
  lineChart(el, { xs, series, yMin: 0, yMax: 1, yFormat: (v, exact) => exact ? pct(v) : `${Math.round(v * 100)}%`, xFormat: (x) => `row ${x}` });
  dataTable($("#steps-table"), ["row", ...series.map((s) => s.name)],
    xs.map((x, i) => [x, ...series.map((s) => pct(s.values[i]))]));
}

// ------------------------------------------------------------------ test-set section
function renderTestSet() {
  const meta = state.meta, cmp = state.compare;
  if (!meta || !cmp) return;
  const names = MODELS.filter((n) => meta[n]);
  const any = meta[names[0]];
  $("#test-meta").textContent =
    `${cmp.test_samples} digits never seen in training · ${any.training.epochs} epochs · hidden size ${any.config.hidden_size}`;

  const tiles = names.map((n) => ({
    label: `${LABEL[n]} test accuracy`, value: pct(meta[n].metrics.test_acc),
    hint: `${meta[n].parameters.toLocaleString()} params · trained in ${meta[n].training.seconds}s`,
  }));
  if (cmp.agreement) {
    const a = cmp.agreement;
    tiles.push(
      { label: "Both correct", value: a.both_correct, hint: `same answer on ${a.same_prediction} digits` },
      { label: "Only RNN correct", value: a.only_rnn_correct, hint: "LSTM wrong" },
      { label: "Only LSTM correct", value: a.only_lstm_correct, hint: "RNN wrong" },
      { label: "Both wrong", value: a.both_wrong, hint: "" },
    );
  }
  $("#tiles").innerHTML = tiles.map((t) =>
    `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}</div><div class="hint">${t.hint}</div></div>`).join("");

  renderCurves();

  const gallery = $("#gallery");
  const cases = cmp.hard_cases ?? [];
  gallery.innerHTML = cases.length ? "" : `<div class="empty">Both models are perfect on the test set.</div>`;
  for (const c of cases) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "case";
    const mark = (p) => p === c.label ? `${p}` : `<span class="x">${p} ✗</span>`;
    btn.innerHTML = `<canvas width="8" height="8"></canvas>
      <span class="cap">true <b>${c.label}</b><br>RNN ${mark(c.rnn)} · LSTM ${mark(c.lstm)}</span>`;
    paintGrid($("canvas", btn), c.pixels);
    btn.addEventListener("click", () => {
      loadPixels(c.pixels, c.label, `test #${c.index}`);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    gallery.appendChild(btn);
  }
}

function renderCurves() {
  const meta = state.meta;
  if (!meta) return;
  const names = MODELS.filter((n) => meta[n]);
  const epochs = meta[names[0]].history.loss.length;
  const xs = Array.from({ length: epochs }, (_, i) => i + 1);
  const xTicks = xs.map((_, i) => i).filter((i) => i === 0 || (i + 1) % 5 === 0);
  const build = (key) => names.map((n) => ({ name: LABEL[n], color: colorOf(n), values: meta[n].history[key] }));

  const loss = build("loss");
  lineChart($("#loss-chart"), { xs, xTicks, series: loss, yMin: 0, yFormat: (v, exact) => v.toFixed(exact ? 4 : 1), xFormat: (x) => `epoch ${x}` });
  const acc = build("test_acc");
  const lo = Math.floor(Math.min(...acc.flatMap((s) => s.values)) * 10) / 10;
  lineChart($("#acc-chart"), { xs, xTicks, series: acc, yMin: lo, yMax: 1, yFormat: (v, exact) => exact ? pct(v) : `${Math.round(v * 100)}%`, xFormat: (x) => `epoch ${x}` });

  dataTable($("#loss-table"), ["epoch", ...loss.map((s) => s.name)], xs.map((x, i) => [x, ...loss.map((s) => s.values[i].toFixed(4))]));
  dataTable($("#acc-table"), ["epoch", ...acc.map((s) => s.name)], xs.map((x, i) => [x, ...acc.map((s) => pct(s.values[i]))]));
}

// ------------------------------------------------------------------ boot
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { renderCurves(); renderStepsChart(); }, 150);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  renderResults();
  renderCurves();
});

async function boot() {
  renderResults();
  const healthEl = $("#health");
  try {
    const health = await api("/api/health");
    healthEl.textContent = `${health.status} · ${health.models.map((n) => LABEL[n]).join(" + ") || "no models"} · v${health.version}`;
    healthEl.classList.add(health.status === "ok" ? "ok" : "bad");
    if (!health.models.length) {
      showError("No trained models found. Run `make train`, then restart the server.");
      return;
    }
    [state.meta, state.compare] = await Promise.all([api("/api/models"), api("/api/compare")]);
    renderTestSet();
    await loadRandom();
  } catch (e) {
    healthEl.textContent = "offline";
    healthEl.classList.add("bad");
    showError(`Cannot reach the API: ${e.message}`);
  }
}

boot();
