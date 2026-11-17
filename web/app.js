// Page wiring: loads the model and adapters, runs the real forward pass from model.js, and draws the figures.
import { BPETokenizer } from "./bpe.js";
import { parseWeights } from "./weights.js";
import { GPT } from "./model.js";
import { Adapter, AdapterShapeError, attachAdapter, mergeAdapter, unmergeAdapter, parseAdapter, validateAdapter, weightsDigest, snapshotWeights, maxWeightDifference } from "./adapter.js";
import { generate } from "./generate.js";
import { meanLoss, maxAbsDiff } from "./score.js";
import { probabilities } from "./sampling.js";
import { renderChips, markChips } from "./chips.js";
import { drawRankChart, drawParamsChart, drawForgettingChart, drawEfficiencyChart, drawScaleChart, drawResultsTable } from "./charts.js";
import { h, plainToken, pct } from "./dom.js";

const ADAPTERS = [
  ["code_r1", "Python source r=1"],
  ["code_r4", "Python source r=4"],
  ["code_r16", "Python source r=16"],
  ["js_r4", "JavaScript r=4"],
];
const EXAMPLES = [
  ["Python function", "def read_file(path):\n    with open(path) as f:\n        "],
  ["Python class", "class Cache:\n    def __init__(self, size):\n        "],
  ["JavaScript", "function readFile(path) {\n    const text = fs.readFileSync(path, \"utf8\");"],
  ["Docs sentence", "A decorator is a function that"],
];
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const runs = new Set((params.get("run") ?? "").split(",").filter(Boolean));
const wants = (name) => runs.has(name) || runs.has("all");

const S = { tok: null, model: null, parity: null, results: null, adapters: new Map(), current: "off", records: [], baseProbs: [], promptLen: 0, hover: null, pinned: null, busy: false };
const idle = () => new Promise((r) => setTimeout(r, 15));
const tokText = (id) => S.tok.decode([id]);
const label = (id) => (id === "off" ? "adapter off" : S.adapters.get(id)?.label ?? id);
const scaleNow = () => Number($("scale").value);

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = `status ${kind}`;
}

async function load() {
  const get = (u, t) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r[t](); });
  const [wbuf, tokSpec, parity, results, ...files] = await Promise.all([
    get("model/weights.bin", "arrayBuffer"), get("model/tokenizer.json", "json"), get("model/parity_lora.json", "json"), get("docs/results.json", "json"),
    ...ADAPTERS.map(([id]) => get(`model/adapters/${id}.lora`, "arrayBuffer")),
  ]);
  S.model = new GPT(parseWeights(wbuf));
  S.tok = new BPETokenizer(tokSpec);
  S.parity = parity;
  S.results = results;
  files.forEach((buf, i) => {
    const [id, text] = ADAPTERS[i];
    const a = parseAdapter(buf, id);
    a.label = text;
    a.bytes = buf.byteLength;
    validateAdapter(a, S.model.config);
    S.adapters.set(id, a);
  });
}

function showFacts() {
  const r = S.results, a = [...S.adapters.values()];
  const sizes = a.map((x) => `${x.label} ${(x.bytes / 1024).toFixed(0)} KB`).join(", ");
  $("facts").replaceChildren(h("b", {}, r.setup.base_parameters.toLocaleString("en")), " base parameters (", `${(S.model.t.size ? [...S.model.t.values()].reduce((n, t) => n + t.byteLength, 0) / 1e6 : 0).toFixed(1)} MB`, "). Adapter files: ", sizes, ". Each run: ", h("b", {}, String(r.setup.steps)), ` steps of ${r.setup.batch_size} x ${r.setup.context} tokens, CPU only.`);
}

/** Make `id` the live adapter ("off" for none) at the slider's scale. */
function use(id) {
  S.current = id;
  S.model.mergedAdapter = null;
  if (id === "off") S.model.adapter = null;
  else attachAdapter(S.model, S.adapters.get(id));
  S.model.scaleMult = scaleNow();
}

function withAdapter(id, scale, fn) {
  const keep = [S.model.adapter, S.model.scaleMult];
  S.model.adapter = id === "off" ? null : S.adapters.get(id);
  S.model.scaleMult = scale;
  try { return fn(); } finally { [S.model.adapter, S.model.scaleMult] = keep; }
}

function syncLabels() {
  $("tempv").textContent = Number($("temp").value) === 0 ? "0 (greedy)" : Number($("temp").value).toFixed(2);
  $("maxnewv").textContent = $("maxnew").value;
  $("scalev").textContent = scaleNow().toFixed(2);
}

function buildControls() {
  const seg = $("adapterSeg");
  const options = [["off", "Off"], ...ADAPTERS];
  seg.replaceChildren(...options.map(([id, text]) => h("button", { class: "small", "data-id": id, "aria-pressed": String(id === S.current), onclick: () => choose(id) }, text)));
  $("examples").replaceChildren(...EXAMPLES.map(([name, text]) => h("button", { class: "small", onclick: () => { $("prompt").value = text; } }, name)));
}

function markAdapterButtons() {
  for (const b of $("adapterSeg").children) b.setAttribute("aria-pressed", String(b.dataset.id === S.current));
  $("outTag").textContent = S.current === "off" ? "adapter off" : `${label(S.current)}, scale ${scaleNow().toFixed(2)}`;
  $("outTag").className = `tag${S.current === "off" ? " off" : ""}`;
}

async function choose(id) {
  use(id);
  markAdapterButtons();
  await generateMain();
}

function genOpts() {
  const promptIds = S.tok.encode($("prompt").value);
  const context = S.model.config.context;
  return { promptIds, maxNew: Math.max(0, Math.min(Number($("maxnew").value), context - promptIds.length)), temperature: Number($("temp").value), topK: 40, seed: Number($("seed").value) || 0, policy: "stop" };
}

function chipTokens(promptIds, generated) {
  return [...promptIds.map((id) => ({ text: tokText(id), prob: null, prompt: true })), ...generated.map((g) => ({ text: tokText(g.id), prob: g.prob, prompt: false }))];
}

async function generateMain() {
  if (S.busy) return;
  S.busy = true;
  $("go").disabled = true;
  setStatus($("genStatus"), "Generating...");
  await idle();
  try {
    const opts = genOpts();
    if (!opts.promptIds.length) throw new Error("The prompt is empty.");
    const t0 = performance.now();
    const out = generate(S.model, opts);
    const ms = performance.now() - t0;
    const ids = [...opts.promptIds, ...out.generated.map((g) => g.id)];
    S.promptLen = opts.promptIds.length;
    S.records = [...opts.promptIds.map(() => null), ...out.generated];
    S.baseProbs = withAdapter("off", 0, () => {
      const V = S.model.config.vocab_size;
      const { logits } = S.model.forward(ids.slice(0, -1));
      return ids.slice(1).map((_, t) => probabilities(logits.subarray(t * V, (t + 1) * V)));
    });
    S.hover = S.pinned = null;
    renderChips($("chipsOut"), chipTokens(opts.promptIds, out.generated), { onHover: (i) => { S.hover = i; showDist(); }, onPin: (i) => { S.pinned = S.pinned === i ? null : i; showDist(); } });
    showDist();
    setStatus($("genStatus"), `${out.generated.length} tokens in ${ms.toFixed(0)} ms${out.stopped ? ", stopped at the 128-token context limit" : ""}. Seed ${opts.seed}.`, "ok");
  } catch (err) {
    setStatus($("genStatus"), `${err.name}: ${err.message}`, "err");
  } finally {
    S.busy = false;
    $("go").disabled = false;
  }
}

function showDist() {
  const f = S.hover ?? S.pinned;
  markChips($("chipsOut"), S.hover, S.pinned);
  const root = $("dist");
  if (f == null) return root.replaceChildren(h("div", { class: "empty" }, "Hover or click a token above."));
  const rec = S.records[f];
  if (!rec) return root.replaceChildren(h("div", { class: "empty" }, "This token is part of your prompt; the model did not choose it. Hover a generated token."));
  const base = S.baseProbs[f - 1];
  const withA = rec.probs;
  const order = Array.from(withA, (p, id) => id).sort((a, b) => withA[b] - withA[a]).slice(0, 7);
  if (!order.includes(rec.id)) order.push(rec.id);
  const max = Math.max(...order.map((id) => Math.max(withA[id], base[id])));
  const w = (p) => `width:${Math.max(0.5, (p / max) * 100).toFixed(1)}%`;
  root.replaceChildren(h("div", { class: "bars" }, ...order.map((id) => h("div", { class: `bar${id === rec.id ? " chosen" : ""}` },
    h("span", { class: "t" }, `${id === rec.id ? "> " : ""}${plainToken(tokText(id))}`),
    h("span", { class: "track" }, h("i", { style: w(base[id]) }), h("i", { style: w(withA[id]) })),
    h("span", { class: "p" }, `${pct(base[id])} / ${pct(withA[id])}`)))));
}

async function runCompare() {
  $("compare").disabled = true;
  $("cmpWrap").hidden = false;
  const root = $("cmpOut");
  root.replaceChildren(h("div", { class: "empty" }, "Generating three continuations..."));
  await idle();
  const opts = genOpts();
  const cols = [["off", "Base model"], ["code_r4", "Python source r=4"], ["js_r4", "JavaScript r=4"]];
  const nodes = [];
  for (const [id, title] of cols) {
    const out = withAdapter(id, scaleNow(), () => generate(S.model, opts));
    const box = h("div", { class: "out" });
    renderChips(box, chipTokens(opts.promptIds, out.generated), { onHover: () => {}, onPin: () => {} });
    nodes.push(h("div", {}, h("h3", {}, title, h("span", { class: `tag${id === "off" ? " off" : ""}` }, id === "off" ? "off" : `scale ${scaleNow().toFixed(2)}`)), box));
    await idle();
  }
  root.replaceChildren(...nodes);
  $("compare").disabled = false;
}

// ---- held-out scoring ----------------------------------------------------------------------------------------------
const PASSAGES = [["docs", "Python docs"], ["code", "Python source"], ["js", "JavaScript"]];
const lossCache = new Map();
function passageLoss(id, scale, key) {
  const k = `${id}|${id === "off" ? 0 : scale}|${key}`;
  if (!lossCache.has(k)) lossCache.set(k, withAdapter(id, scale, () => meanLoss(S.model, S.parity.passages[key].ids)));
  return lossCache.get(k);
}

async function scoreRows(rows, table, statusEl) {
  const cells = rows.map(() => []);
  for (let i = 0; i < rows.length; i++) {
    for (const [key] of PASSAGES) {
      setStatus(statusEl, `Scoring ${rows[i].text} on ${key}...`);
      await idle();
      cells[i].push(passageLoss(rows[i].id, rows[i].scale, key));
    }
  }
  const trained = { code_r1: 1, code_r4: 1, code_r16: 1, js_r4: 2 };
  table.replaceChildren(
    h("thead", {}, h("tr", {}, h("th", {}, "model"), ...PASSAGES.map(([, t]) => h("th", { class: "num" }, t)))),
    h("tbody", {}, ...rows.map((r, i) => h("tr", {}, h("td", {}, r.text), ...cells[i].map((v, j) => h("td", { class: `num${r.id !== "off" && trained[r.id] === j && r.scale > 0 ? " hit" : ""}` }, v.toFixed(3)))))));
  return cells;
}

async function runScore() {
  $("scoreGo").disabled = true;
  const s = scaleNow();
  const rows = [{ id: "off", scale: 0, text: "Base model" }, { id: "code_r4", scale: s, text: `Python source r=4, scale ${s.toFixed(2)}` }, { id: "js_r4", scale: s, text: `JavaScript r=4, scale ${s.toFixed(2)}` }];
  await scoreRows(rows, $("scoreTable"), $("scoreStatus"));
  setStatus($("scoreStatus"), "Done. Move the scale in Figure 1 and press again.", "ok");
  $("scoreGo").disabled = false;
}

async function runScale2() {
  $("breakScale").disabled = true;
  $("scale").value = 2;
  syncLabels();
  if (S.current === "off") use("code_r4");
  else S.model.scaleMult = 2;
  markAdapterButtons();
  const id = S.current === "off" ? "code_r4" : S.current;
  const rows = [0, 1, 2].map((s) => ({ id: s === 0 ? "off" : id, scale: s, text: s === 0 ? "Base model (scale 0)" : `${label(id)}, scale ${s}` }));
  await scoreRows(rows, $("scaleTable"), $("scaleStatus"));
  setStatus($("scaleStatus"), "Scale 2 is worse than scale 1 on the passage the adapter was trained for.", "ok");
  drawScaleChart($("chScale"), S.results, 2);
  await generateMain();
  $("breakScale").disabled = false;
}

// ---- mismatched adapters ------------------------------------------------------------------------------------------
function tryAttach(adapter, title) {
  const before = S.model.adapter;
  const box = $("badOut");
  try {
    attachAdapter(S.model, adapter);
    S.model.adapter = before;
    box.replaceChildren(h("div", { class: "errbox" }, "This file was accepted, which it should not have been."));
  } catch (err) {
    box.replaceChildren(h("div", { class: "errbox" }, h("b", {}, `${title}: refused. `), `${err.name}: ${err.message}`),
      h("div", { class: "okbox" }, `Nothing changed. The model is still running: ${before ? `${label(S.current)}, scale ${S.model.scaleMult.toFixed(2)}` : "the plain base model"}.`));
    return err;
  }
}

function badMix() {
  const good = S.adapters.get("code_r4"), wide = S.adapters.get("code_r16");
  const layers = new Map(good.layers);
  const name = "block0.attn.qkv";
  layers.set(name, { ...good.layers.get(name), B: wide.layers.get(name).B, rankB: wide.layers.get(name).rankB });
  tryAttach(new Adapter("mixed", good.config, layers), "Rank 4 A with rank 16 B");
}

function badHeader() {
  const good = S.adapters.get("code_r4");
  tryAttach(new Adapter("liar", { ...good.config, rank: 8, alpha: 16 }, good.layers), "Header says rank 8");
}

async function onUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  try {
    const a = parseAdapter(await file.arrayBuffer(), file.name);
    a.label = file.name;
    a.bytes = file.size;
    attachAdapter(S.model, a);
    S.adapters.set(file.name, a);
    $("adapterSeg").append(h("button", { class: "small", "data-id": file.name, "aria-pressed": "true", onclick: () => choose(file.name) }, file.name));
    await choose(file.name);
  } catch (err) {
    S.model.adapter = S.current === "off" ? null : S.adapters.get(S.current);
    setStatus($("genStatus"), `${err.name}: ${err.message} The model is unchanged.`, "err");
  }
}

// ---- merge and unmerge --------------------------------------------------------------------------------------------
async function runMerge() {
  $("mergeGo").disabled = true;
  const id = S.current === "off" ? "code_r4" : S.current;
  const adapter = S.adapters.get(id);
  const scale = scaleNow() || 1;
  const saveMult = S.model.scaleMult;
  S.model.mergedAdapter = null;
  S.model.adapter = null;
  S.model.scaleMult = scale;
  const rows = [];
  const loss = () => meanLoss(S.model, S.parity.passages.code.ids);
  const snap = snapshotWeights(S.model);
  const hash0 = await weightsDigest(S.model);
  const add = async (step, lossV, extra = "") => {
    const d = await weightsDigest(S.model);
    rows.push([step, d, maxWeightDifference(S.model, snap), lossV, d === hash0]);
    void extra;
  };
  await add("1. base model", loss());
  S.model.adapter = adapter;
  await add("2. adapter on the side (not merged)", loss());
  S.model.adapter = null;
  mergeAdapter(S.model, adapter);
  await add("3. merged into the weights", loss());
  unmergeAdapter(S.model);
  S.model.adapter = null;
  await add("4. unmerged by subtracting", loss());
  for (const [k, v] of snap) S.model.t.get(k).set(v); // start the backup path from the original weights again
  mergeAdapter(S.model, adapter, { keepBackup: true });
  unmergeAdapter(S.model);
  S.model.adapter = null;
  await add("5. merged, then restored from a saved copy", loss());
  use(S.current);
  S.model.scaleMult = saveMult;
  const sub = rows[3], backup = rows[4];
  $("mergeTable").replaceChildren(
    h("thead", {}, h("tr", {}, h("th", {}, "step"), h("th", {}, "SHA-256 of base weights"), h("th", { class: "num" }, "max weight change"), h("th", { class: "num" }, "code loss"), h("th", {}, "same as original"))),
    h("tbody", {}, ...rows.map((r) => h("tr", {}, h("td", {}, r[0]), h("td", { class: "hash" }, `${r[1].slice(0, 16)}...`), h("td", { class: "num" }, r[2] === 0 ? "0" : r[2].toExponential(1)), h("td", { class: "num" }, r[3].toFixed(4)), h("td", { class: r[4] ? "" : "bad" }, r[4] ? "bit for bit" : "no")))));
  const merged = rows[2];
  $("mergeVerdict").replaceChildren(h("div", { class: backup[4] ? "okbox" : "errbox" },
    `${id} at scale ${scale.toFixed(2)}. Merged loss ${merged[3].toFixed(4)} against ${rows[1][3].toFixed(4)} with the adapter on the side. After subtracting, the largest weight change is ${sub[2].toExponential(1)} (${sub[4] ? "identical bit for bit" : "float32 rounding, not identical"}). After restoring the saved copy the weights are ${backup[4] ? "identical bit for bit" : "NOT identical"}.`));
  $("mergeGo").disabled = false;
}

// ---- results text and parity --------------------------------------------------------------------------------------
function showResults() {
  const r = S.results, st = r.setup;
  const M = Object.fromEntries(r.methods.map((m) => [m.name, m]));
  const lr = st.learning_rate_choice;
  $("resultsSub").textContent = `The base model is the 875k-parameter model from transformer-from-scratch, trained on Python documentation. The new domain is Python standard-library source code with the docstrings removed. Every run takes ${st.steps} steps of ${st.batch_size} sequences of ${st.context} tokens with the same tokenizer, batches and validation documents. Learning rates (full ${lr.full.chosen}, LoRA ${lr.lora.chosen}) were picked on a separate dev split, not on the validation split. Alpha is twice the rank, so the scale alpha/r is 2 at every rank.`;
  drawRankChart($("chRank"), r);
  drawParamsChart($("chParams"), r);
  drawForgettingChart($("chForget"), r);
  drawEfficiencyChart($("chEff"), r);
  drawScaleChart($("chScale"), r, scaleNow());
  drawResultsTable($("resTable"), r);
  const d = st.data;
  $("resNote").textContent = `Loss in nats per token, validation documents only: ${d.code.tokens_val.toLocaleString("en")} code tokens, ${d.old.tokens_val.toLocaleString("en")} documentation tokens, ${d.js.tokens_val.toLocaleString("en")} JavaScript tokens. Train times were measured with several runs sharing the machine, so compare step times (below) rather than wall time.`;
  const swap = r.adapter_swap, a = swap.adapters;
  $("swapText").textContent = `Two adapters, one base. Code adapter on code ${a.code_r4.code.toFixed(3)}, on JavaScript ${a.code_r4.js.toFixed(3)}; JavaScript adapter on JavaScript ${a.js_r4.js.toFixed(3)}, on code ${a.js_r4.code.toFixed(3)}; base on code ${r.base.code.toFixed(3)} and JavaScript ${r.base.js.toFixed(3)}. Attaching an adapter took ${swap.swap_ms.toFixed(2)} ms in Python, and after both swaps the base weights were ${swap.base_unchanged_after_swaps ? "byte-identical" : "NOT identical"}. Figure 3 repeats this live.`;
  const best = ["lora_r1", "lora_r4", "lora_r16"].map((n) => M[n]).sort((x, y) => x.val_loss.code - y.val_loss.code)[0];
  const st1 = r.step_time_seconds;
  const lines = [
    `On the new domain full fine-tuning reaches ${M.full.val_loss.code.toFixed(3)} and the best LoRA (r=${best.rank}) ${best.val_loss.code.toFixed(3)}, a gap of ${(best.val_loss.code - M.full.val_loss.code).toFixed(3)} nats ${best.val_loss.code > M.full.val_loss.code ? "in favour of full fine-tuning" : "in favour of LoRA"}. LoRA trains ${best.trainable_percent}% of the weights; full fine-tuning trains all of them.`,
    `Forgetting: the original-domain loss rises by ${(M.full.val_loss.old - r.base.old).toFixed(3)} after full fine-tuning and by ${(best.val_loss.old - r.base.old).toFixed(3)} after LoRA r=${best.rank}.`,
    `Speed: one training step takes ${st1.full} s for full fine-tuning and ${st1.lora_r4} s for LoRA r=4 on this model, measured back to back. LoRA still has to push gradients through every layer, so at this size the saving is modest.`,
    `Rank and data: going from rank 1 to 16 changes the new-domain loss from ${M.lora_r1.val_loss.code.toFixed(3)} to ${M.lora_r16.val_loss.code.toFixed(3)}. With 10% of the text, LoRA r=4 reaches ${r.data_efficiency[0].val_loss.code.toFixed(3)}, with 100% ${r.data_efficiency[2].val_loss.code.toFixed(3)}; all runs use the same number of steps, so the 10% run sees its text many times over.`,
    "One seed per setting, one tokenizer trained on documentation (it is poor at code), and a model under a million parameters. Differences of a few thousandths of a nat are within what another seed could change.",
  ];
  $("honest").replaceChildren(h("ul", { class: "limits" }, ...lines.map((t) => h("li", {}, t))));
}

function runParity() {
  const out = [];
  let worst = 0, n = 0;
  for (const c of S.parity.cases) {
    const ids = S.parity.prompts[c.prompt].ids;
    S.model.mergedAdapter = null;
    const { logits } = withAdapter(c.adapter, c.scale, () => S.model.forward(ids));
    const V = S.model.config.vocab_size;
    c.logits.forEach((row, t) => { worst = Math.max(worst, maxAbsDiff(logits.subarray(t * V, (t + 1) * V), row)); n += row.length; });
    const greedy = withAdapter(c.adapter, c.scale, () => generate(S.model, { promptIds: ids, maxNew: c.greedy.length, temperature: 0, topK: 0, seed: 1, policy: "stop" }));
    out.push(greedy.generated.map((g) => g.id).join() === c.greedy.join());
  }
  const ok = worst < 1e-3 && out.every(Boolean);
  $("parity").replaceChildren(h("span", { class: ok ? "pass" : "fail" }, ok ? "PASS" : "FAIL"), ` ${S.parity.cases.length} adapter cases, ${n.toLocaleString("en")} logits: largest difference from the Python float64 values ${worst.toExponential(1)}; greedy text identical in ${out.filter(Boolean).length} of ${out.length}.`);
}

async function main() {
  syncLabels();
  try {
    await load();
  } catch (err) {
    setStatus($("genStatus"), `Could not load the model files: ${err.message}`, "err");
    return;
  }
  buildControls();
  showFacts();
  showResults();
  runParity();
  const wantAdapter = params.get("adapter");
  if (params.has("scale")) $("scale").value = params.get("scale");
  if (params.has("prompt")) $("prompt").value = params.get("prompt");
  syncLabels();
  use(wantAdapter && (wantAdapter === "off" || S.adapters.has(wantAdapter)) ? wantAdapter : "off");
  markAdapterButtons();
  for (const id of ["go", "compare", "scoreGo", "breakScale", "badMix", "badHeader", "mergeGo"]) $(id).disabled = false;
  $("go").onclick = generateMain;
  $("compare").onclick = runCompare;
  $("scoreGo").onclick = runScore;
  $("breakScale").onclick = runScale2;
  $("badMix").onclick = badMix;
  $("badHeader").onclick = badHeader;
  $("mergeGo").onclick = runMerge;
  $("upload").onchange = onUpload;
  let timer = null;
  $("scale").oninput = () => {
    syncLabels();
    S.model.scaleMult = scaleNow();
    drawScaleChart($("chScale"), S.results, scaleNow());
    markAdapterButtons();
    clearTimeout(timer);
    if (S.current !== "off" && S.records.length) timer = setTimeout(generateMain, 350);
  };
  for (const id of ["temp", "maxnew"]) $(id).oninput = syncLabels;
  if (wants("generate") || wants("compare")) await generateMain();
  if (wants("compare")) await runCompare();
  if (wants("score")) await runScore();
  if (wants("scale2")) await runScale2();
  if (wants("mismatch")) { badMix(); }
  if (wants("merge")) await runMerge();
  document.documentElement.dataset.ready = "1";
}

main();
