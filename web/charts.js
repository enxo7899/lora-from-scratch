// Hand-drawn SVG charts. Every number comes from docs/results.json (written by scripts/experiments.py).
import { h, s } from "./dom.js";

const INK = "#14213d", ACCENT = "#1f4fbf", MUTED = "#5b6478", RULE = "#d5dae4", OCHRE = "#7d8aa5";

function frame(W, H, label) {
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label });
  const text = (x, y, str, o = {}) => {
    const t = s("text", { x, y, "font-size": 12.5, fill: "#3a4560", ...o });
    t.textContent = str;
    svg.append(t);
    return t;
  };
  const line = (x1, y1, x2, y2, o = {}) => svg.append(s("line", { x1, y1, x2, y2, stroke: RULE, "stroke-width": 0.9, ...o }));
  return { svg, text, line };
}

/** Round a range outward to a tidy step so the axis labels are short. */
function niceRange(values, step) {
  const lo = Math.floor(Math.min(...values) / step) * step;
  const hi = Math.ceil(Math.max(...values) / step) * step;
  return [lo, hi === lo ? lo + step : hi];
}

function yAxis(c, m, W, H, lo, hi, step, label) {
  const Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  for (let v = lo; v <= hi + 1e-9; v += step) {
    c.line(m.l, Y(v), W - m.r, Y(v));
    c.text(m.l - 7, Y(v) + 4, v.toFixed(step < 0.1 ? 2 : 1), { "text-anchor": "end" });
  }
  c.text(4, 13, label, { fill: MUTED });
  return Y;
}

const dot = (x, y, color, hollow = false) => s("circle", { cx: x, cy: y, r: 5, fill: hollow ? "#ffffff" : color, stroke: color, "stroke-width": 2 });

function refLine(c, m, W, Y, v, label, color, dash = "5 4", above = false) {
  c.svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: color, "stroke-width": 1.3, "stroke-dasharray": dash }));
  c.text(W - m.r - 4, Y(v) + (above ? -6 : 15), `${label} ${v.toFixed(3)}`, { "text-anchor": "end", fill: color, "font-size": 12 });
}

const methodsByName = (r) => Object.fromEntries(r.methods.map((x) => [x.name, x]));

/** New-domain loss against LoRA rank, with full fine-tuning and the untouched base as reference lines. */
export function drawRankChart(root, r) {
  const W = 560, H = 320, m = { l: 52, r: 16, t: 26, b: 46 };
  const c = frame(W, H, "Loss on the new domain against LoRA rank");
  const M = methodsByName(r);
  const ranks = ["lora_r1", "lora_r4", "lora_r16"];
  const vals = [...ranks.map((n) => M[n].val_loss.code), M.full.val_loss.code, r.base.code, M.lora_r4_attn.val_loss.code];
  const [lo, hi] = niceRange(vals, 0.25);
  const Y = yAxis(c, m, W, H, lo, hi, 0.25, "validation loss, new domain (nats per token)");
  const X = (rank) => m.l + 24 + (Math.log2(rank) / 4) * (W - m.l - m.r - 48);
  for (const k of [1, 2, 4, 8, 16]) {
    c.line(X(k), H - m.b, X(k), H - m.b + 5, { stroke: MUTED });
    c.text(X(k), H - m.b + 19, String(k), { "text-anchor": "middle" });
  }
  c.text((m.l + W - m.r) / 2, H - 6, "LoRA rank r (log scale)", { "text-anchor": "middle", fill: MUTED });
  refLine(c, m, W, Y, r.base.code, "no tuning", MUTED, "2 4", true);
  refLine(c, m, W, Y, M.full.val_loss.code, "full fine-tune", INK);
  c.svg.append(s("polyline", { points: ranks.map((n) => `${X(M[n].rank)},${Y(M[n].val_loss.code)}`).join(" "), fill: "none", stroke: ACCENT, "stroke-width": 2.4 }));
  for (const n of ranks) {
    c.svg.append(dot(X(M[n].rank), Y(M[n].val_loss.code), ACCENT));
    c.text(X(M[n].rank), Y(M[n].val_loss.code) - 11, M[n].val_loss.code.toFixed(3), { "text-anchor": "middle", fill: ACCENT, "font-weight": 700 });
  }
  const a = M.lora_r4_attn;
  c.svg.append(dot(X(4), Y(a.val_loss.code), ACCENT, true));
  c.text(X(4) + 10, Y(a.val_loss.code) + 4, `attention only ${a.val_loss.code.toFixed(3)}`, { fill: ACCENT });
  root.replaceChildren(c.svg);
}

/** Trainable parameters (log scale) against new-domain loss: what each extra parameter buys. */
export function drawParamsChart(root, r) {
  const W = 560, H = 320, m = { l: 52, r: 16, t: 26, b: 46 };
  const c = frame(W, H, "Trainable parameters against new-domain loss");
  const pts = r.methods.map((x) => ({ n: x.trainable_parameters, y: x.val_loss.code, label: x.kind === "full" ? "full" : `r=${x.rank}${x.name.endsWith("attn") ? " attn" : ""}`, full: x.kind === "full", hollow: x.name.endsWith("attn") }));
  const [lo, hi] = niceRange([...pts.map((p) => p.y), r.base.code], 0.25);
  const Y = yAxis(c, m, W, H, lo, hi, 0.25, "validation loss, new domain (nats per token)");
  const xmin = Math.log10(Math.min(...pts.map((p) => p.n)) / 1.6), xmax = Math.log10(Math.max(...pts.map((p) => p.n)) * 1.5);
  const X = (n) => m.l + 10 + ((Math.log10(n) - xmin) / (xmax - xmin)) * (W - m.l - m.r - 20);
  for (const k of [1e4, 1e5, 1e6]) {
    if (Math.log10(k) < xmin || Math.log10(k) > xmax) continue;
    c.line(X(k), H - m.b, X(k), H - m.b + 5, { stroke: MUTED });
    c.text(X(k), H - m.b + 19, k >= 1e6 ? "1M" : `${k / 1000}k`, { "text-anchor": "middle" });
  }
  c.text((m.l + W - m.r) / 2, H - 6, "trainable parameters (log scale)", { "text-anchor": "middle", fill: MUTED });
  refLine(c, m, W, Y, r.base.code, "no tuning", MUTED, "2 4", true);
  for (const p of pts) {
    c.svg.append(dot(X(p.n), Y(p.y), p.full ? INK : ACCENT, p.hollow));
    const left = p.full;
    const dy = left ? 4 : p.hollow ? 22 : -11;
    c.text(X(p.n) + (left ? -10 : 0), Y(p.y) + dy, `${p.label}  ${p.n.toLocaleString("en")}`, { "text-anchor": left ? "end" : "middle", fill: p.full ? INK : ACCENT, "font-size": 11.5 });
  }
  root.replaceChildren(c.svg);
}

/** How much worse the original domain gets after tuning, per method (bars start at the untouched base). */
export function drawForgettingChart(root, r) {
  const rows = [...r.methods].map((x) => ({ label: x.label.replace("LoRA ", "LoRA ").replace(", attention only", ", attn only"), v: x.val_loss.old - r.base.old, abs: x.val_loss.old, full: x.kind === "full" }));
  const W = 560, rowH = 34, m = { l: 150, r: 74, t: 30, b: 34 }, H = m.t + m.b + rows.length * rowH;
  const c = frame(W, H, "Increase in original-domain loss after each tuning method");
  const max = Math.max(...rows.map((x) => x.v)) * 1.08;
  const X = (v) => m.l + (v / max) * (W - m.l - m.r);
  c.text(m.l, 15, `increase over the untouched base loss of ${r.base.old.toFixed(3)}`, { fill: MUTED });
  const step = max > 1 ? 0.5 : max > 0.5 ? 0.25 : 0.1;
  for (let v = 0; v <= max + 1e-9; v += step) {
    c.line(X(v), m.t - 4, X(v), H - m.b + 2);
    c.text(X(v), H - m.b + 17, v.toFixed(2), { "text-anchor": "middle" });
  }
  rows.forEach((row, i) => {
    const y = m.t + i * rowH;
    c.text(m.l - 10, y + rowH / 2 + 4, row.label, { "text-anchor": "end", fill: INK });
    c.svg.append(s("rect", { x: m.l, y: y + 6, width: Math.max(1.5, X(row.v) - m.l), height: rowH - 14, fill: row.full ? INK : ACCENT }));
    c.text(X(row.v) + 7, y + rowH / 2 + 4, `+${row.v.toFixed(3)}`, { fill: INK, "font-weight": 700 });
  });
  root.replaceChildren(c.svg);
}

/** New-domain loss against the share of new-domain training text used (LoRA r=4, same number of steps). */
export function drawEfficiencyChart(root, r) {
  const W = 560, H = 320, m = { l: 52, r: 16, t: 26, b: 58 };
  const c = frame(W, H, "New-domain loss against the share of training data");
  const pts = r.data_efficiency;
  const [lo, hi] = niceRange([...pts.map((p) => p.val_loss.code), r.base.code], 0.25);
  const Y = yAxis(c, m, W, H, lo, hi, 0.25, "validation loss, new domain (nats per token)");
  const X = (f) => m.l + 30 + (Math.log(f / 0.1) / Math.log(10)) * (W - m.l - m.r - 60);
  for (const p of pts) {
    c.line(X(p.fraction), H - m.b, X(p.fraction), H - m.b + 5, { stroke: MUTED });
    c.text(X(p.fraction), H - m.b + 19, `${Math.round(p.fraction * 100)}%`, { "text-anchor": "middle" });
    c.text(X(p.fraction), H - m.b + 34, `${Math.round(p.train_tokens / 1000)}k tokens`, { "text-anchor": "middle", fill: MUTED, "font-size": 11.5 });
  }
  c.text((m.l + W - m.r) / 2, H - 6, "share of the new-domain training text (log scale)", { "text-anchor": "middle", fill: MUTED });
  refLine(c, m, W, Y, r.base.code, "no tuning", MUTED, "2 4", true);
  c.svg.append(s("polyline", { points: pts.map((p) => `${X(p.fraction)},${Y(p.val_loss.code)}`).join(" "), fill: "none", stroke: ACCENT, "stroke-width": 2.4 }));
  for (const p of pts) {
    c.svg.append(dot(X(p.fraction), Y(p.val_loss.code), ACCENT));
    c.text(X(p.fraction), Y(p.val_loss.code) - 11, p.val_loss.code.toFixed(3), { "text-anchor": "middle", fill: ACCENT, "font-weight": 700 });
  }
  root.replaceChildren(c.svg);
}

/** Held-out loss on three domains as the adapter scale moves from 0 (base model) to 2. `current` draws the slider position. */
export function drawScaleChart(root, r, current = null) {
  const W = 560, H = 320, m = { l: 52, r: 96, t: 26, b: 46 };
  const c = frame(W, H, "Held-out loss against adapter scale");
  const pts = r.scale_sweep.points;
  const series = [["code", "Python source", ACCENT, "none"], ["old", "Python docs", INK, "none"], ["js", "JavaScript", OCHRE, "5 4"]];
  const [lo, hi] = niceRange(pts.flatMap((p) => series.map(([k]) => p[k])), 0.5);
  const Y = yAxis(c, m, W, H, lo, hi, 0.5, "validation loss (nats per token)");
  const X = (v) => m.l + (v / 2) * (W - m.l - m.r);
  for (const v of [0, 0.5, 1, 1.5, 2]) {
    c.line(X(v), H - m.b, X(v), H - m.b + 5, { stroke: MUTED });
    c.text(X(v), H - m.b + 19, String(v), { "text-anchor": "middle" });
  }
  c.text((m.l + W - m.r) / 2, H - 6, `scale multiplier on the ${r.scale_sweep.adapter.replace("_", " ")} adapter (1 = as trained)`, { "text-anchor": "middle", fill: MUTED });
  if (current != null) {
    c.svg.append(s("line", { x1: X(current), x2: X(current), y1: m.t - 6, y2: H - m.b, stroke: INK, "stroke-width": 1.2, "stroke-dasharray": "3 3" }));
    c.text(X(current), m.t - 9, `now ${Number(current).toFixed(2)}`, { "text-anchor": "middle", fill: INK, "font-weight": 700 });
  }
  const ends = [];
  for (const [k, name, color, dash] of series) {
    c.svg.append(s("polyline", { points: pts.map((p) => `${X(p.scale)},${Y(p[k])}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.3, "stroke-dasharray": dash, "stroke-linejoin": "round" }));
    for (const p of pts) c.svg.append(s("circle", { cx: X(p.scale), cy: Y(p[k]), r: 2.6, fill: color }));
    ends.push([name, color, pts[pts.length - 1][k]]);
  }
  let lastY = -99;
  for (const [name, color, v] of ends.sort((a, b) => a[2] - b[2])) {
    const y = Math.max(Y(v), lastY + 15);
    lastY = y;
    c.text(W - m.r + 8, y + 4, name, { fill: color, "font-weight": 700, "font-size": 12 });
  }
  root.replaceChildren(c.svg);
}

/** The full numbers behind the charts. */
export function drawResultsTable(table, r) {
  const f = (v) => v.toFixed(3);
  const body = [["No tuning (base model)", "0", "-", "-", r.base]];
  for (const x of r.methods) body.push([x.label, x.trainable_parameters.toLocaleString("en"), `${x.trainable_percent}%`, `${Math.round(x.seconds)} s`, x.val_loss]);
  table.replaceChildren(
    h("thead", {}, h("tr", {}, ...["method", "trainable", "of base", "train time", "code (new)", "docs (old)", "JavaScript"].map((t, i) => h("th", { class: i > 0 ? "num" : "" }, t)))),
    h("tbody", {}, ...body.map(([label, n, pct, sec, v]) => h("tr", {}, h("td", {}, label), h("td", { class: "num" }, n), h("td", { class: "num" }, pct), h("td", { class: "num" }, sec), h("td", { class: "num" }, f(v.code)), h("td", { class: "num" }, f(v.old)), h("td", { class: "num" }, f(v.js))))),
  );
}

export { INK, ACCENT, MUTED };
