// LoRA adapters in the browser: reader for the file written by src/lora_scratch/adapter_io.py, shape checks that mirror
// lora.py, and merge / unmerge of an adapter into the base weights. The forward pass itself lives in model.js.

export class AdapterFileError extends Error {
  constructor(message) {
    super(message);
    this.name = "AdapterFileError";
  }
}

export class AdapterShapeError extends Error {
  constructor(message) {
    super(message);
    this.name = "AdapterShapeError";
  }
}

export class AdapterStateError extends Error {
  constructor(message) {
    super(message);
    this.name = "AdapterStateError";
  }
}

export const TARGETS = ["attn.qkv", "attn.proj", "mlp.fc", "mlp.proj"];

export class Adapter {
  /**
   * @param {string} name label shown in the page
   * @param {{rank: number, alpha: number, targets: string[]}} config
   * @param {Map<string, {A: Float32Array, B: Float32Array, nIn: number, nOut: number, r: number, rankA: number, rankB: number}>} layers
   * @param {object} meta free-form training notes stored in the file
   */
  constructor(name, config, layers, meta = {}) {
    this.name = name;
    this.config = config;
    this.layers = layers;
    this.meta = meta;
  }

  /** alpha / rank, before the live scale multiplier. */
  get scale() {
    return this.config.alpha / this.config.rank;
  }

  get parameters() {
    let n = 0;
    for (const l of this.layers.values()) n += l.A.length + l.B.length;
    return n;
  }
}

/** @param {ArrayBuffer} buffer @param {string} name @returns {Adapter} (shapes are not checked against a model yet) */
export function parseAdapter(buffer, name = "adapter") {
  const view = new DataView(buffer);
  if (buffer.byteLength < 8 || String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== "LORA") throw new AdapterFileError("not a LoRA adapter file");
  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const base = 8 + headerLength;
  const tensors = new Map();
  for (const t of header.tensors) {
    if (base + t.offset + t.length > buffer.byteLength) throw new AdapterFileError(`file truncated while reading ${t.name}`);
    tensors.set(t.name, { data: new Float32Array(buffer.slice(base + t.offset, base + t.offset + t.length)), shape: t.shape });
  }
  const layers = new Map();
  for (const key of tensors.keys()) {
    const layer = key.slice(0, key.lastIndexOf("."));
    if (layers.has(layer)) continue;
    const a = tensors.get(`${layer}.A`);
    const b = tensors.get(`${layer}.B`);
    if (!a || !b) throw new AdapterFileError(`${layer} is missing its A or B tensor`);
    layers.set(layer, { A: a.data, B: b.data, nIn: a.shape[0], nOut: b.shape[1], r: header.config.rank, rankA: a.shape[1], rankB: b.shape[0] });
  }
  return new Adapter(name, header.config, layers, header.meta ?? {});
}

/** Throw AdapterShapeError with a plain message unless every tensor fits the model and the rank stated in the header. */
export function validateAdapter(adapter, modelConfig) {
  const { dim, n_layers: L } = modelConfig;
  const widths = { "attn.qkv": [dim, 3 * dim], "attn.proj": [dim, dim], "mlp.fc": [dim, 4 * dim], "mlp.proj": [4 * dim, dim] };
  const expected = new Set();
  for (let l = 0; l < L; l++) for (const t of adapter.config.targets) expected.add(`block${l}.${t}`);
  for (const name of expected) {
    if (!adapter.layers.has(name)) throw new AdapterShapeError(`adapter has no weights for ${name}`);
  }
  for (const [name, d] of adapter.layers) {
    if (!expected.has(name)) throw new AdapterShapeError(`adapter has weights for ${name}, which this model does not have`);
    if (d.rankA !== d.rankB) throw new AdapterShapeError(`${name}: A has rank ${d.rankA} but B has rank ${d.rankB}; they must match`);
    if (d.rankA !== adapter.config.rank) throw new AdapterShapeError(`${name}: tensors have rank ${d.rankA} but the adapter header says rank ${adapter.config.rank}`);
    const [nIn, nOut] = widths[name.split(".").slice(1).join(".")];
    if (d.nIn !== nIn || d.nOut !== nOut) throw new AdapterShapeError(`${name}: adapter expects a ${d.nIn} -> ${d.nOut} layer, the model layer is ${nIn} -> ${nOut}`);
  }
}

/** Use the adapter for every later forward pass. Validates first; the model is untouched if validation fails. */
export function attachAdapter(model, adapter) {
  validateAdapter(adapter, model.config);
  if (model.mergedAdapter) throw new AdapterStateError("an adapter is merged into the weights; unmerge it first");
  model.adapter = adapter;
}

export function detachAdapter(model) {
  model.adapter = null;
}

/** Dense update scale * A B for one layer as a Float64Array (nIn x nOut). */
function product(d, scale) {
  const out = new Float64Array(d.nIn * d.nOut);
  const r = d.rankA;
  for (let i = 0; i < d.nIn; i++) {
    for (let k = 0; k < r; k++) {
      const a = scale * d.A[i * r + k];
      for (let j = 0; j < d.nOut; j++) out[i * d.nOut + j] += a * d.B[k * d.nOut + j];
    }
  }
  return out;
}

/**
 * Add scale * A B into the base weights and stop using the separate branch. With keepBackup the original weights are
 * copied first so unmergeAdapter can restore them bit for bit; without it unmerge subtracts the same product.
 */
export function mergeAdapter(model, adapter, { keepBackup = false } = {}) {
  validateAdapter(adapter, model.config);
  if (model.mergedAdapter) throw new AdapterStateError("an adapter is already merged");
  const scale = adapter.scale * model.scaleMult;
  const backups = keepBackup ? new Map() : null;
  for (const [name, d] of adapter.layers) {
    const w = model.get(`${name}.w`);
    if (backups) backups.set(name, w.slice());
    const delta = product(d, scale);
    for (let i = 0; i < w.length; i++) w[i] = w[i] + Math.fround(delta[i]);
  }
  model.adapter = null;
  model.mergedAdapter = { adapter, scale, backups };
}

export function unmergeAdapter(model) {
  const m = model.mergedAdapter;
  if (!m) throw new AdapterStateError("no adapter is merged");
  for (const [name, d] of m.adapter.layers) {
    const w = model.get(`${name}.w`);
    if (m.backups) w.set(m.backups.get(name));
    else {
      const delta = product(d, m.scale);
      for (let i = 0; i < w.length; i++) w[i] = w[i] - Math.fround(delta[i]);
    }
  }
  model.mergedAdapter = null;
  model.adapter = m.adapter;
}

/** SHA-256 (hex) over the bytes of every base tensor, in file order. Identical hashes mean identical weights, bit for bit. */
export async function weightsDigest(model) {
  const parts = [...model.t.values()];
  const all = new Uint8Array(parts.reduce((n, p) => n + p.byteLength, 0));
  let at = 0;
  for (const p of parts) {
    all.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), at);
    at += p.byteLength;
  }
  const digest = await crypto.subtle.digest("SHA-256", all);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Copy of every base tensor, to measure how far merge and unmerge moved the weights. */
export function snapshotWeights(model) {
  return new Map([...model.t].map(([k, v]) => [k, v.slice()]));
}

/** Largest absolute difference between the model's weights and a snapshot. */
export function maxWeightDifference(model, snapshot) {
  let worst = 0;
  for (const [k, v] of model.t) {
    const s = snapshot.get(k);
    for (let i = 0; i < v.length; i++) worst = Math.max(worst, Math.abs(v[i] - s[i]));
  }
  return worst;
}
