// The transformer forward pass in plain JavaScript: same maths as src/tiny_gpt (pre-LN blocks,
// causal multi-head attention, tanh-GELU MLP, tied output head). Weights are float32, activations float64.
// MIT licence, Copyright (c) 2026 EmSoft. Copied from https://github.com/enxo7899/transformer-from-scratch and extended
// with an optional LoRA branch on the four linear layers of every block (see adapter.js).

export class ContextOverflowError extends Error {
  constructor(message) {
    super(message);
    this.name = "ContextOverflowError";
  }
}

const GELU_C = Math.sqrt(2 / Math.PI);

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

/** out (rows x nOut) = x (rows x nIn) @ w (nIn x nOut) + b */
function linear(x, rows, nIn, w, b, nOut) {
  const out = new Float64Array(rows * nOut);
  for (let r = 0; r < rows; r++) {
    const o = r * nOut;
    for (let j = 0; j < nOut; j++) out[o + j] = b[j];
    for (let i = 0; i < nIn; i++) {
      const xi = x[r * nIn + i];
      const wi = i * nOut;
      for (let j = 0; j < nOut; j++) out[o + j] += xi * w[wi + j];
    }
  }
  return out;
}

function layerNorm(x, rows, dim, gamma, beta) {
  const out = new Float64Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    let mu = 0;
    for (let i = 0; i < dim; i++) mu += x[r * dim + i];
    mu /= dim;
    let v = 0;
    for (let i = 0; i < dim; i++) v += (x[r * dim + i] - mu) ** 2;
    const rstd = 1 / Math.sqrt(v / dim + 1e-5);
    for (let i = 0; i < dim; i++) out[r * dim + i] = (x[r * dim + i] - mu) * rstd * gamma[i] + beta[i];
  }
  return out;
}

/** In-place softmax over the first n entries of a. */
export function softmaxInPlace(a, n = a.length) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (a[i] > max) max = a[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    a[i] = Math.exp(a[i] - max);
    sum += a[i];
  }
  for (let i = 0; i < n; i++) a[i] /= sum;
  return a;
}

export class KVCache {
  /** @param {GPT} model @param {boolean} unbounded allow positions past the trained context (used only to demonstrate the failure) */
  constructor(model, unbounded = false) {
    const { n_layers: layers, dim } = model.config;
    this.length = 0;
    this.unbounded = unbounded;
    this.capacity = unbounded ? 2048 : model.config.context;
    this.keys = Array.from({ length: layers }, () => new Float64Array(this.capacity * dim));
    this.values = Array.from({ length: layers }, () => new Float64Array(this.capacity * dim));
  }
}

/** out += scale * (x A) B for one linear layer; A is nIn x r, B is r x nOut. */
function addLowRank(out, x, rows, nIn, nOut, { A, B, r }, scale) {
  const xa = new Float64Array(r);
  for (let row = 0; row < rows; row++) {
    xa.fill(0);
    for (let i = 0; i < nIn; i++) {
      const xi = x[row * nIn + i];
      for (let k = 0; k < r; k++) xa[k] += xi * A[i * r + k];
    }
    for (let k = 0; k < r; k++) {
      const c = scale * xa[k];
      for (let j = 0; j < nOut; j++) out[row * nOut + j] += c * B[k * nOut + j];
    }
  }
}

export class GPT {
  /** @param {{config: object, tensors: Map<string, Float32Array>}} weights from parseWeights */
  constructor({ config, tensors }) {
    this.config = config;
    this.t = tensors;
    /** @type {import("./adapter.js").Adapter | null} adapter used by every forward pass, or null for the plain base model */
    this.adapter = null;
    /** Multiplier on alpha / rank: 1 is the trained strength, 0 switches the adapter off. */
    this.scaleMult = 1;
    /** Set while an adapter is merged into the weights (see mergeAdapter). */
    this.mergedAdapter = null;
  }

  /** One linear layer of a block (base weights plus the live adapter branch, if any). `name` is like "block2.attn.qkv". */
  lin(x, rows, nIn, nOut, name) {
    const out = linear(x, rows, nIn, this.get(`${name}.w`), this.get(`${name}.b`), nOut);
    const branch = this.adapter?.layers.get(name);
    if (branch && this.scaleMult !== 0) addLowRank(out, x, rows, nIn, nOut, branch, this.adapter.scale * this.scaleMult);
    return out;
  }

  get(name) {
    const w = this.t.get(name);
    if (!w) throw new Error(`missing tensor ${name}`);
    return w;
  }

  /** Output logits for every position. With attention:true also returns attention[layer][head] as T*T row-major weights. */
  forward(ids, { attention = false } = {}) {
    const { context, n_layers: L, n_heads: H, dim: D, vocab_size: V } = this.config;
    const T = ids.length;
    if (T > context) throw new ContextOverflowError(`sequence of ${T} tokens exceeds the trained context of ${context}`);
    const hd = D / H;
    const wte = this.get("emb.wte");
    const wpe = this.get("emb.wpe");
    let x = new Float64Array(T * D);
    for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) x[t * D + i] = wte[ids[t] * D + i] + wpe[t * D + i];
    const attn = [];
    for (let l = 0; l < L; l++) {
      const p = (n) => this.get(`block${l}.${n}`);
      const h = layerNorm(x, T, D, p("ln1.gamma"), p("ln1.beta"));
      const qkv = this.lin(h, T, D, 3 * D, `block${l}.attn.qkv`);
      const merged = new Float64Array(T * D);
      const maps = [];
      for (let head = 0; head < H; head++) {
        const att = new Float64Array(T * T);
        const row = new Float64Array(T);
        for (let t = 0; t < T; t++) {
          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let i = 0; i < hd; i++) dot += qkv[t * 3 * D + head * hd + i] * qkv[s * 3 * D + D + head * hd + i];
            row[s] = dot / Math.sqrt(hd);
          }
          softmaxInPlace(row, t + 1);
          for (let s = 0; s <= t; s++) {
            att[t * T + s] = row[s];
            for (let i = 0; i < hd; i++) merged[t * D + head * hd + i] += row[s] * qkv[s * 3 * D + 2 * D + head * hd + i];
          }
        }
        maps.push(att);
      }
      attn.push(maps);
      const proj = this.lin(merged, T, D, D, `block${l}.attn.proj`);
      for (let i = 0; i < x.length; i++) x[i] += proj[i];
      const h2 = layerNorm(x, T, D, p("ln2.gamma"), p("ln2.beta"));
      const a = this.lin(h2, T, D, 4 * D, `block${l}.mlp.fc`);
      for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]);
      const m = this.lin(a, T, 4 * D, D, `block${l}.mlp.proj`);
      for (let i = 0; i < x.length; i++) x[i] += m[i];
    }
    const xf = layerNorm(x, T, D, this.get("ln_f.gamma"), this.get("ln_f.beta"));
    const logits = new Float64Array(T * V);
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < V; v++) {
        let s = 0;
        for (let i = 0; i < D; i++) s += xf[t * D + i] * wte[v * D + i];
        logits[t * V + v] = s;
      }
    }
    return attention ? { logits, attention: attn } : { logits };
  }

  /** Feed one token at position cache.length; returns logits for the next token. Pass out = [] to receive per-layer, per-head attention rows. */
  step(token, cache, out = null) {
    const { context, n_layers: L, n_heads: H, dim: D, vocab_size: V } = this.config;
    const pos = cache.length;
    if (pos >= context && !cache.unbounded) throw new ContextOverflowError(`position ${pos} is beyond the trained context of ${context}`);
    if (pos >= cache.capacity) throw new ContextOverflowError("cache is full");
    const hd = D / H;
    const wte = this.get("emb.wte");
    const wpe = this.get("emb.wpe");
    const posRow = Math.min(pos, context - 1); // only reachable in unbounded mode: reuse the last learned position
    let x = new Float64Array(D);
    for (let i = 0; i < D; i++) x[i] = wte[token * D + i] + wpe[posRow * D + i];
    for (let l = 0; l < L; l++) {
      const p = (n) => this.get(`block${l}.${n}`);
      const h = layerNorm(x, 1, D, p("ln1.gamma"), p("ln1.beta"));
      const qkv = this.lin(h, 1, D, 3 * D, `block${l}.attn.qkv`);
      cache.keys[l].set(qkv.subarray(D, 2 * D), pos * D);
      cache.values[l].set(qkv.subarray(2 * D, 3 * D), pos * D);
      const merged = new Float64Array(D);
      const rows = [];
      for (let head = 0; head < H; head++) {
        const row = new Float64Array(pos + 1);
        for (let s = 0; s <= pos; s++) {
          let dot = 0;
          for (let i = 0; i < hd; i++) dot += qkv[head * hd + i] * cache.keys[l][s * D + head * hd + i];
          row[s] = dot / Math.sqrt(hd);
        }
        softmaxInPlace(row);
        for (let s = 0; s <= pos; s++) for (let i = 0; i < hd; i++) merged[head * hd + i] += row[s] * cache.values[l][s * D + head * hd + i];
        rows.push(row);
      }
      if (out) out.push(rows);
      const proj = this.lin(merged, 1, D, D, `block${l}.attn.proj`);
      for (let i = 0; i < D; i++) x[i] += proj[i];
      const h2 = layerNorm(x, 1, D, p("ln2.gamma"), p("ln2.beta"));
      const a = this.lin(h2, 1, D, 4 * D, `block${l}.mlp.fc`);
      for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]);
      const m = this.lin(a, 1, 4 * D, D, `block${l}.mlp.proj`);
      for (let i = 0; i < D; i++) x[i] += m[i];
    }
    const xf = layerNorm(x, 1, D, this.get("ln_f.gamma"), this.get("ln_f.beta"));
    const logits = new Float64Array(V);
    for (let v = 0; v < V; v++) {
      let s = 0;
      for (let i = 0; i < D; i++) s += xf[i] * wte[v * D + i];
      logits[v] = s;
    }
    cache.length += 1;
    return logits;
  }
}
