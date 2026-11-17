"""Run every experiment and write docs/results.json.

  python scripts/experiments.py all                 everything, one after another (about 40 minutes on 4 cores)
  python scripts/experiments.py train NAME [NAME..] only the named runs (results are cached in runs/, so reruns resume)
  python scripts/experiments.py assemble            build docs/results.json from the cached runs plus the quick evaluations

Run names: lr_full_<lr>, lr_lora_<lr> (learning-rate choice on a dev split), then full, lora_r1, lora_r4, lora_r16,
lora_r4_attn, lora_r4_10pct, lora_r4_30pct and js_lora_r4. BLAS is limited to one thread so several runs can share a machine.
"""
from __future__ import annotations

import os

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_scratch.adapter_io import load_adapter, save_adapter  # noqa: E402
from lora_scratch.finetune import Domain, fixed_windows, load_domain, old_domain_windows, train_full, train_lora  # noqa: E402
from lora_scratch.lora import ATTENTION_TARGETS, LoRAAdapter, LoRAConfig, checksum  # noqa: E402
from tiny_gpt.bpe import BPETokenizer  # noqa: E402
from tiny_gpt.infer import generate  # noqa: E402
from tiny_gpt.train import evaluate, hardware  # noqa: E402
from tiny_gpt.weights import load_weights  # noqa: E402

RUNS, MODEL, DATA = ROOT / "runs", ROOT / "model", ROOT / "data"
STEPS, LR_STEPS, BATCH = 300, 100, 16
FULL_LRS, LORA_LRS = (1e-4, 3e-4, 1e-3, 3e-3), (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
MAIN = {
    "full": dict(kind="full", label="Full fine-tune"),
    "lora_r1": dict(kind="lora", rank=1, label="LoRA r=1"),
    "lora_r4": dict(kind="lora", rank=4, label="LoRA r=4"),
    "lora_r16": dict(kind="lora", rank=16, label="LoRA r=16"),
    "lora_r4_attn": dict(kind="lora", rank=4, targets=ATTENTION_TARGETS, label="LoRA r=4, attention only"),
    "lora_r4_10pct": dict(kind="lora", rank=4, fraction=0.1, label="LoRA r=4, 10% of data"),
    "lora_r4_30pct": dict(kind="lora", rank=4, fraction=0.3, label="LoRA r=4, 30% of data"),
    "js_lora_r4": dict(kind="lora", rank=4, domain="js", label="LoRA r=4 on JavaScript"),
}
SHIPPED = {"lora_r1": "code_r1", "lora_r4": "code_r4", "lora_r16": "code_r16", "js_lora_r4": "js_r4"}
SCALES = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]


class Env:
    """Tokenizer, base weights path, the three domains and their evaluation windows, loaded once."""

    def __init__(self) -> None:
        self.tok = BPETokenizer.load(MODEL / "tokenizer.json")
        self.context = load_weights(MODEL / "weights.bin").config.context
        self.domains = {n: load_domain(DATA / f"{n}.txt", n) for n in ("code", "js")}
        self.val = {n: fixed_windows(d.tokens(self.tok, "val"), self.context) for n, d in self.domains.items()}
        self.val["old"] = old_domain_windows(DATA / "old_domain_val.txt", self.tok, self.context)
        self.dev = fixed_windows(self.domains["code"].tokens(self.tok, "dev"), self.context)

    def base(self):
        return load_weights(MODEL / "weights.bin")

    def score(self, model, limit: int | None = None) -> dict[str, float]:
        return {k: round(evaluate(model, w, limit=limit), 4) for k, w in self.val.items()}


def cached(name: str) -> dict | None:
    path = RUNS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_lr(env: Env, name: str) -> None:
    kind, lr = name.split("_")[1], float(name.split("_")[2])
    model = env.base()
    tokens = env.domains["code"].tokens(env.tok, "train")
    if kind == "full":
        log = train_full(model, tokens, steps=LR_STEPS, lr=lr, batch=BATCH)
    else:
        log = train_lora(model, LoRAAdapter.create(model, LoRAConfig(rank=4), seed=0), tokens, steps=LR_STEPS, lr=lr, batch=BATCH)
    dev = evaluate(model, env.dev)
    (RUNS / f"{name}.json").write_text(json.dumps({"name": name, "kind": kind, "lr": lr, "steps": LR_STEPS, "dev_loss": round(dev, 4), "seconds": round(log.seconds, 1)}))
    say(f"{name}: dev loss {dev:.4f} in {log.seconds:.0f}s")


def chosen_lrs() -> dict:
    out = {}
    for kind, grid in (("full", FULL_LRS), ("lora", LORA_LRS)):
        runs = [cached(f"lr_{kind}_{lr:g}") for lr in grid]
        if not all(runs):
            raise SystemExit(f"learning-rate runs for {kind} are missing; run them first")
        out[kind] = {"grid": {f"{r['lr']:g}": r["dev_loss"] for r in runs}, "chosen": min(runs, key=lambda r: r["dev_loss"])["lr"]}
    return out


def run_main(env: Env, name: str) -> None:
    spec = MAIN[name]
    lr = chosen_lrs()[spec["kind"]]["chosen"]
    domain = env.domains[spec.get("domain", "code")]
    tokens = domain.tokens(env.tok, "train", spec.get("fraction", 1.0))
    model = env.base()
    probes = {"old_loss": env.val["old"], "new_loss": env.val[spec.get("domain", "code")]}
    kw = dict(steps=STEPS, lr=lr, batch=BATCH, probes=probes)
    if spec["kind"] == "full":
        log = train_full(model, tokens, **kw)
        adapter = None
    else:
        cfg = LoRAConfig(rank=spec["rank"], targets=spec.get("targets", LoRAConfig().targets))
        adapter = LoRAAdapter.create(model, cfg, seed=0, meta={"domain": spec.get("domain", "code"), "steps": STEPS, "lr": lr, "train_tokens": int(len(tokens))})
        log = train_lora(model, adapter, tokens, **kw)
    scores = env.score(model)
    shipped = SHIPPED.get(name)
    if adapter is not None:
        (MODEL / "adapters").mkdir(exist_ok=True)
        save_adapter(adapter, model, (MODEL / "adapters" / f"{shipped}.lora") if shipped else RUNS / f"{name}.lora")
    result = {"name": name, "label": spec["label"], "kind": spec["kind"], "rank": spec.get("rank"), "domain": spec.get("domain", "code"),
              "targets": list(spec.get("targets", LoRAConfig().targets)) if spec["kind"] == "lora" else None, "fraction": spec.get("fraction", 1.0),
              "lr": lr, "steps": STEPS, "batch": BATCH, "train_tokens": int(len(tokens)), "trainable_parameters": log.trainable_parameters,
              "seconds": round(log.seconds, 1), "final_train_loss": round(log.train_loss, 4), "val_loss": scores, "curve": log.curve}
    (RUNS / f"{name}.json").write_text(json.dumps(result))
    say(f"{name}: {scores} in {log.seconds:.0f}s, {log.trainable_parameters} trainable")


def step_time_benchmark(env: Env) -> dict:
    """Seconds per training step for each method, measured back to back on 5 batches after one warm-up batch."""
    if (RUNS / "step_time.json").exists():
        return json.loads((RUNS / "step_time.json").read_text())
    tokens = env.domains["code"].tokens(env.tok, "train")
    out = {}
    for label, kind, rank in (("full", "full", None), ("lora_r1", "lora", 1), ("lora_r4", "lora", 4), ("lora_r16", "lora", 16)):
        model = env.base()
        if kind == "full":
            train_full(model, tokens, steps=1, lr=1e-5, batch=BATCH)
            log = train_full(model, tokens, steps=5, lr=1e-5, batch=BATCH)
        else:
            adapter = LoRAAdapter.create(model, LoRAConfig(rank=rank), seed=0)
            train_lora(model, adapter, tokens, steps=1, lr=1e-5, batch=BATCH)
            log = train_lora(model, adapter, tokens, steps=5, lr=1e-5, batch=BATCH)
        out[label] = round(log.seconds / 5, 3)
    (RUNS / "step_time.json").write_text(json.dumps(out))
    return out


def quick_evaluations(env: Env) -> dict:
    """Base model, adapter swapping, scale sweep, merge round trip and sample text. Needs the shipped adapters."""
    base = env.base()
    base_hash = checksum(base)
    result = {"base": env.score(base)}
    adapters = {n: load_adapter(MODEL / "adapters" / f"{n}.lora") for n in ("code_r4", "js_r4")}
    swap = {"adapters": {}, "swap_ms": None}
    timings = []
    for name, adapter in adapters.items():
        t = time.perf_counter()
        adapter.attach(base)
        timings.append((time.perf_counter() - t) * 1000)
        swap["adapters"][name] = env.score(base)
    adapters["js_r4"].detach(base)
    swap["swap_ms"] = round(float(np.mean(timings)), 3)
    swap["base_unchanged_after_swaps"] = checksum(base) == base_hash
    swap["base_after_detach"] = env.score(base)
    result["swap"] = swap

    adapter = adapters["code_r4"]
    sweep = []
    for s in SCALES:
        adapter.set_scale(s)
        adapter.attach(base)
        sweep.append({"scale": s, **{k: round(evaluate(base, env.val[k], limit=128), 4) for k in ("old", "code", "js")}})
    adapter.set_scale(1.0)
    result["scale_sweep"] = {"adapter": "code_r4", "windows": 128, "points": sweep}

    adapter.attach(base)
    live = evaluate(base, env.val["code"], limit=128)
    weights_before = {n: p.copy() for n, p, _ in base.named_parameters()}
    adapter.merge(base)
    merged = evaluate(base, env.val["code"], limit=128)
    adapter.unmerge(base)
    drift = max(float(np.abs(weights_before[n] - p).max()) for n, p, _ in base.named_parameters())
    hash_subtract = checksum(base) == base_hash
    adapter.detach(base)
    fresh = env.base()  # the subtract path above may have left rounding; test the backup path on pristine weights
    pristine = checksum(fresh)
    adapter.merge(fresh, keep_backup=True)
    adapter.unmerge(fresh)
    adapter.detach(fresh)
    result["merge"] = {"live_loss": round(live, 5), "merged_loss": round(merged, 5), "max_abs_weight_drift_after_unmerge": drift,
                       "bit_for_bit_after_subtract": hash_subtract, "bit_for_bit_with_backup": checksum(fresh) == pristine}
    base = env.base()  # samples below use untouched weights

    prompt = "def read_file(path):\n    with open(path) as f:\n        "
    ids = env.tok.encode(prompt)
    samples = {}
    for label, ad in (("base", None), ("code_r4", adapters["code_r4"]), ("js_r4", adapters["js_r4"])):
        if ad:
            ad.attach(base)
        samples[label] = env.tok.decode(generate(base, ids, 60, temperature=0.8, top_k=40, seed=7))
        if ad:
            ad.detach(base)
    result["samples"] = {"prompt": prompt, "temperature": 0.8, "top_k": 40, "seed": 7, "text": samples}
    return result


def assemble(env: Env) -> None:
    lrs = chosen_lrs()
    runs = {n: cached(n) for n in MAIN}
    missing = [n for n, r in runs.items() if r is None]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    quick = quick_evaluations(env)
    base_params = env.base().n_parameters()
    methods = []
    for n in ("full", "lora_r1", "lora_r4", "lora_r16", "lora_r4_attn"):
        r = runs[n]
        methods.append({k: r[k] for k in ("name", "label", "kind", "rank", "targets", "lr", "trainable_parameters", "seconds", "final_train_loss", "val_loss")}
                       | {"trainable_percent": round(100 * r["trainable_parameters"] / base_params, 2), "curve": r["curve"]})
    efficiency = [{"fraction": f, "train_tokens": runs[n]["train_tokens"], "val_loss": runs[n]["val_loss"], "final_train_loss": runs[n]["final_train_loss"]}
                  for f, n in ((0.1, "lora_r4_10pct"), (0.3, "lora_r4_30pct"), (1.0, "lora_r4"))]
    js = runs["js_lora_r4"]
    results = {
        "setup": {"base_parameters": base_params, "steps": STEPS, "batch_size": BATCH, "context": env.context, "tokens_per_run": STEPS * BATCH * env.context,
                  "weight_decay": 0.0, "warmup_steps": 20, "grad_clip": 1.0, "alpha_rule": "alpha = 2 * rank", "seed": 0,
                  "learning_rate_choice": {"steps": LR_STEPS, "metric": "loss on the code dev split (held out from train and val)", **lrs},
                  "hardware": hardware() | {"blas_threads_per_run": 1, "python": platform.python_version()},
                  "data": {n: {"documents_train": len(d.train_docs), "documents_dev": len(d.dev_docs), "documents_val": len(d.val_docs),
                               "tokens_train": int(len(d.tokens(env.tok, "train"))), "tokens_val": int(env.val[n].y.size)} for n, d in env.domains.items()}
                          | {"old": {"tokens_val": int(env.val["old"].y.size)}}},
        "base": quick["base"],
        "methods": methods,
        "data_efficiency": efficiency,
        "adapter_swap": quick["swap"] | {"js_training": {k: js[k] for k in ("trainable_parameters", "seconds", "val_loss")}},
        "scale_sweep": quick["scale_sweep"],
        "merge": quick["merge"],
        "step_time_seconds": step_time_benchmark(env),
        "total_training_seconds": round(sum(r["seconds"] for r in runs.values()) + sum(cached(f"lr_{k}_{lr:g}")["seconds"] for k, g in (("full", FULL_LRS), ("lora", LORA_LRS)) for lr in g), 1),
        "samples": quick["samples"],
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    say("wrote docs/results.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["all", "train", "assemble"])
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    env = Env()
    lr_names = [f"lr_full_{lr:g}" for lr in FULL_LRS] + [f"lr_lora_{lr:g}" for lr in LORA_LRS]
    names = lr_names + list(MAIN) if args.command == "all" else args.names
    if args.command in ("all", "train"):
        for n in names:
            if n in lr_names:
                if cached(n) is None:
                    run_lr(env, n)
            elif n in MAIN:
                if cached(n) is None:
                    run_main(env, n)
            else:
                raise SystemExit(f"unknown run {n}")
    if args.command in ("all", "assemble"):
        assemble(env)


if __name__ == "__main__":
    main()
