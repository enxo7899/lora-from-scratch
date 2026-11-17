// Teacher-forced loss of a token sequence under the model as it currently stands (adapter, scale, merge state included).

/** Mean negative log-likelihood in nats per token of ids[1..] given the tokens before each. */
export function meanLoss(model, ids) {
  const V = model.config.vocab_size;
  const T = ids.length - 1;
  const { logits } = model.forward(ids.slice(0, T));
  let total = 0;
  for (let t = 0; t < T; t++) {
    const row = logits.subarray(t * V, (t + 1) * V);
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (row[v] > max) max = row[v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(row[v] - max);
    total += Math.log(sum) + max - row[ids[t + 1]];
  }
  return total / T;
}

/** Largest absolute difference between two equally long arrays. */
export function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
