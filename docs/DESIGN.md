# Design notes

## The pieces

```
 data/*.txt ─► split by document ─► base tokenizer (unchanged) ─► token ids
                                                                    │
 model/weights.bin (frozen base, from transformer-from-scratch)     │
        │                                                           ▼
        │   tiny_gpt.layers.Linear ── y = x W + b  (+ lora.forward(x))
        │                                 ▲
        │            lora_scratch.lora.LoRADelta: s * (x A) B, hand-written backward
        │                                 │  only A and B are given to AdamW
        ▼                                 ▼
  merge / unmerge  ◄──────────  model/adapters/*.lora  (a few hundred KB each)
        │                                 │
        └────────────┬────────────────────┘
                     ▼
   web/model.js (same maths, optional LoRA branch)  ─►  index.html figures
   node --test checks it against Python logits (model/parity_lora.json)
```

## Decisions and trade-offs

**An adapter is attached to a layer, not wrapped around it.** `Linear` has an optional `lora` slot. With it empty the layer is byte-for-byte the original code, so base-model tests, base weights and the weight file format are untouched. Swapping adapters is assigning that slot; detaching restores the exact base output (a test compares logits with `array_equal`).

**Freezing is a flag, not a copy.** `set_trainable(False)` makes `backward` still return input gradients but skip weight gradients (LayerNorm, embeddings, the tied head, the frozen `W`). The optimiser is built only from the adapter's parameters, so the base cannot move: a test hashes every base tensor before and after training steps.

**Hand-written backward for the adapter.** With `u = x A`, `y = s u B`: `dB = s uᵀ dy`, `dA = xᵀ (s dy Bᵀ)`, `dx = (s dy Bᵀ) Aᵀ`. Tests compare these with central finite differences in float64 for a single layer, for the whole model, and for random ranks, targets and scales. At initialisation `B = 0`, so the output is unchanged and `dA = 0` while `dB != 0`: both are tested.

**Scaling.** `s = alpha / rank` with `alpha = 2 * rank`, so `s = 2` at every rank and one learning rate fits all ranks. The page multiplies `s` by a live factor (0 to 2). Because the delta is linear in `s`, the slider blends smoothly from the base model to twice the trained strength.

**Merge and unmerge.** `merge` adds `s * A B` to `W` and drops the side branch. `unmerge` either subtracts the same product, which leaves float32 rounding of a few units in the last place, or restores a saved copy, which is exact. Both are offered, measured and shown on the page. Unmerging uses the scale that was merged, not the current slider.

**Shape errors before side effects.** `validate` runs before `attach` or `merge` in both Python and JavaScript: rank of A against B, rank against the header, widths against the model. A rejected file leaves the model as it was.

**File format.** `LORA`, a length-prefixed JSON header (rank, alpha, targets, base shape, training notes) and raw little-endian float32, the same style as the base weight file. No pickle.

**Fair comparison.** Same tokenizer, same batches (seeded), same number of steps and the same validation documents for every method. Learning rates were chosen per method on a separate dev split, not on validation. Training runs one BLAS thread per process so several runs can share a machine; wall times are therefore noisy, and the README reports a back-to-back step-time benchmark as well.

## What I would do next for production

- Several seeds per setting, with confidence intervals; with one seed, gaps of a few thousandths are not evidence.
- A tokenizer that fits the new domain (it is the main handicap on code), which would mean retraining embeddings and so leaves pure LoRA behind.
- Quantised base weights (the QLoRA idea) once the base is large enough for memory to matter.
- Evaluation on a task a client cares about, not only perplexity.
- A learning-rate and rank search that is cheaper than full runs, and early stopping on the dev split.
