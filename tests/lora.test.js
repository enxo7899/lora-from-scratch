import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { GPT, KVCache } from "../web/model.js";
import { parseWeights } from "../web/weights.js";
import { generate } from "../web/generate.js";
import { meanLoss, maxAbsDiff } from "../web/score.js";
import { mulberry32 } from "../web/sampling.js";
import { Adapter, AdapterFileError, AdapterShapeError, AdapterStateError, attachAdapter, maxWeightDifference, mergeAdapter, parseAdapter, snapshotWeights, unmergeAdapter, weightsDigest } from "../web/adapter.js";

const dir = fileURLToPath(new URL("../model/", import.meta.url));
const read = (p) => { const b = readFileSync(dir + p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const parity = JSON.parse(readFileSync(dir + "parity_lora.json", "utf8"));
const fresh = () => new GPT(parseWeights(read("weights.bin")));
const load = (name) => parseAdapter(read(`adapters/${name}.lora`), name);
const V = 512;

function logitsOf(model, ids) { return model.forward(ids).logits; }
function worstRow(logits, rows) { let w = 0; rows.forEach((r, t) => (w = Math.max(w, maxAbsDiff(logits.subarray(t * V, (t + 1) * V), r)))); return w; }
function setup(c) {
  const model = fresh();
  const a = load(c.adapter);
  attachAdapter(model, a);
  model.scaleMult = c.scale;
  return { model, a, ids: parity.prompts[c.prompt].ids };
}

for (const c of parity.cases) {
  const name = `${c.adapter} on a ${c.prompt} prompt at scale ${c.scale}`;
  test(`JS logits match the Python logits for ${name} to 1e-3`, () => {
    const { model, ids } = setup(c);
    assert.ok(worstRow(logitsOf(model, ids), c.logits) < 1e-3);
  });
  test(`greedy text with ${name} is the same token for token as Python`, () => {
    const { model, ids } = setup(c);
    const out = generate(model, { promptIds: ids, maxNew: c.greedy.length, temperature: 0, topK: 0, seed: 1, policy: "stop" });
    assert.deepEqual(out.generated.map((g) => g.id), c.greedy);
  });
  test(`held-out passage loss with ${name} matches Python to 1e-4`, () => {
    const { model } = setup(c);
    for (const [key, expected] of Object.entries(c.passage_loss)) assert.ok(Math.abs(meanLoss(model, parity.passages[key].ids) - expected) < 1e-4, key);
  });
}

test("the base model without an adapter matches the Python base logits", () => {
  const model = fresh();
  for (const [k, p] of Object.entries(parity.prompts)) assert.ok(worstRow(logitsOf(model, p.ids), parity.base[k].logits) < 1e-3, k);
});

test("scale zero gives exactly the base logits even with an adapter attached", () => {
  const model = fresh();
  const ids = parity.prompts.code.ids;
  const base = logitsOf(model, ids);
  attachAdapter(model, load("code_r4"));
  model.scaleMult = 0;
  assert.deepEqual(logitsOf(model, ids), base);
});

test("the cached one-token-at-a-time pass agrees with the full pass when an adapter is on", () => {
  const { model, ids } = setup(parity.cases[0]);
  const full = logitsOf(model, ids);
  const cache = new KVCache(model);
  ids.forEach((id, t) => assert.ok(maxAbsDiff(model.step(id, cache), full.subarray(t * V, (t + 1) * V)) < 1e-9));
});

test("adapter files parse to the rank and the layers they claim", () => {
  for (const [name, rank] of [["code_r1", 1], ["code_r4", 4], ["code_r16", 16], ["js_r4", 4]]) {
    const a = load(name);
    assert.equal(a.config.rank, rank);
    assert.equal(a.layers.size, 4 * a.config.targets.length);
    assert.equal(a.layers.get("block0.attn.qkv").B.length, rank * 384);
  }
});

test("a rank 4 matrix paired with a rank 16 matrix is refused with a message that names both ranks", () => {
  const good = load("code_r4"), wide = load("code_r16");
  const layers = new Map(good.layers);
  layers.set("block0.attn.qkv", { ...good.layers.get("block0.attn.qkv"), B: wide.layers.get("block0.attn.qkv").B, rankB: 16 });
  const model = fresh();
  assert.throws(() => attachAdapter(model, new Adapter("x", good.config, layers)), (e) => e instanceof AdapterShapeError && /A has rank 4 but B has rank 16/.test(e.message));
  assert.equal(model.adapter, null);
});

test("a header that lies about the rank is refused", () => {
  const good = load("code_r4");
  assert.throws(() => attachAdapter(fresh(), new Adapter("x", { ...good.config, rank: 8 }, good.layers)), /header says rank 8/);
});

test("an adapter for a wider model is refused before anything changes", () => {
  const good = load("code_r4");
  const layers = new Map([...good.layers].map(([k, d]) => [k, { ...d, nIn: d.nIn * 2 }]));
  assert.throws(() => attachAdapter(fresh(), new Adapter("x", good.config, layers)), (e) => e instanceof AdapterShapeError && /expects a 256/.test(e.message));
});

test("truncated and foreign files are rejected with clear errors", () => {
  const buf = read("adapters/code_r1.lora");
  assert.throws(() => parseAdapter(buf.slice(0, buf.byteLength - 100)), AdapterFileError);
  assert.throws(() => parseAdapter(new TextEncoder().encode("not an adapter at all").buffer), /not a LoRA/);
});

test("merging gives the same output as the live adapter", () => {
  const model = fresh();
  const a = load("code_r4");
  const ids = parity.prompts.code.ids;
  attachAdapter(model, a);
  const live = logitsOf(model, ids);
  mergeAdapter(model, a);
  assert.ok(maxAbsDiff(logitsOf(model, ids), live) < 1e-3);
});

test("merged weights reproduce the Python merged logits", () => {
  const c = parity.cases[0];
  const model = fresh();
  mergeAdapter(model, load(c.adapter));
  const out = logitsOf(model, parity.prompts[c.prompt].ids);
  const T = parity.prompts[c.prompt].ids.length;
  assert.ok(maxAbsDiff(out.subarray((T - 1) * V, T * V), c.merged_last_logits) < 1e-3);
});

test("merge then unmerge with a backup restores every weight bit for bit", async () => {
  const model = fresh();
  const before = await weightsDigest(model);
  const a = load("code_r16");
  mergeAdapter(model, a, { keepBackup: true });
  assert.notEqual(await weightsDigest(model), before);
  unmergeAdapter(model);
  assert.equal(await weightsDigest(model), before);
});

test("merge then unmerge by subtracting stays within one millionth of the original weights", () => {
  const model = fresh();
  const snap = snapshotWeights(model);
  const a = load("code_r4");
  mergeAdapter(model, a);
  assert.ok(maxWeightDifference(model, snap) > 1e-4, "merging changed the weights");
  unmergeAdapter(model);
  assert.ok(maxWeightDifference(model, snap) < 1e-6);
});

test("an adapter cannot be merged twice, unmerged when not merged, or attached while merged", () => {
  const model = fresh();
  const a = load("js_r4");
  assert.throws(() => unmergeAdapter(model), AdapterStateError);
  mergeAdapter(model, a);
  assert.throws(() => mergeAdapter(model, a), AdapterStateError);
  assert.throws(() => attachAdapter(model, a), AdapterStateError);
});

test("swapping adapters on one base switches behaviour and never changes the base weights", async () => {
  const model = fresh();
  const before = await weightsDigest(model);
  const ids = parity.prompts.code.ids;
  const [code, js] = [load("code_r4"), load("js_r4")];
  attachAdapter(model, code);
  const first = logitsOf(model, ids);
  attachAdapter(model, js);
  const second = logitsOf(model, ids);
  assert.ok(maxAbsDiff(first, second) > 0.05);
  attachAdapter(model, code);
  assert.deepEqual(logitsOf(model, ids), first);
  assert.equal(await weightsDigest(model), before);
});

test("random scales in [0, 2]: merged and live outputs agree and the loss is finite", () => {
  const rng = mulberry32(123);
  const a = load("code_r4");
  const ids = parity.passages.code.ids.slice(0, 24);
  for (let i = 0; i < 4; i++) {
    const scale = rng() * 2;
    const live = fresh();
    attachAdapter(live, a);
    live.scaleMult = scale;
    const merged = fresh();
    merged.scaleMult = scale;
    mergeAdapter(merged, a);
    assert.ok(maxAbsDiff(logitsOf(live, ids), logitsOf(merged, ids)) < 1e-3, `scale ${scale}`);
    assert.ok(Number.isFinite(meanLoss(live, ids)));
  }
});
