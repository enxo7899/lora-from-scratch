"""Domains, evaluation and the fine-tuning loop shared by the experiments and the tests."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from tiny_gpt.bpe import BPETokenizer
from tiny_gpt.data import Windows, fixed_windows, load_documents, random_batch, split_documents, tokenize_documents
from tiny_gpt.layers import Array
from tiny_gpt.model import GPT
from tiny_gpt.optim import AdamW, clip_grad_norm, cosine_lr
from tiny_gpt.train import evaluate


@dataclass
class Domain:
    """One body of text, split by whole document into train, dev (learning-rate choice) and val (reported)."""

    name: str
    train_docs: list[str]
    dev_docs: list[str]
    val_docs: list[str]

    def tokens(self, tok: BPETokenizer, part: str, fraction: float = 1.0) -> np.ndarray:
        docs = {"train": self.train_docs, "dev": self.dev_docs, "val": self.val_docs}[part]
        if part == "train" and fraction < 1.0:
            docs = docs[: max(1, round(len(docs) * fraction))]
        return tokenize_documents(tok, docs)


def load_domain(path: str | Path, name: str, seed: int = 0, val_fraction: float = 0.10, dev_fraction: float = 0.06) -> Domain:
    docs = load_documents(path)
    rest, val = split_documents(docs, val_fraction, seed)
    train, dev = split_documents(rest, dev_fraction, seed + 1)
    return Domain(name, train, dev, val)


def old_domain_windows(path: str | Path, tok: BPETokenizer, context: int) -> Windows:
    """Validation windows of the base model's own corpus (Python documentation prose)."""
    return fixed_windows(tokenize_documents(tok, load_documents(path)), context)


@dataclass
class TrainLog:
    steps: int
    seconds: float
    trainable_parameters: int
    train_loss: float
    curve: list[dict]


def finetune(
    model: GPT,
    params: Iterable[tuple[str, Array, Array]],
    zero_grad: Callable[[], None],
    train_tokens: np.ndarray,
    *,
    steps: int,
    lr: float,
    batch: int = 16,
    warmup: int = 20,
    weight_decay: float = 0.0,
    clip: float = 1.0,
    seed: int = 0,
    probes: dict[str, Windows] | None = None,
    probe_every: int = 50,
    probe_windows: int = 16,
) -> TrainLog:
    """Train only `params` with AdamW. Everything else keeps its values because it is never given to the optimiser.

    `probes` maps a name to evaluation windows scored every `probe_every` steps (a cheap learning curve).
    """
    items = list(params)
    opt = AdamW(items, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    context = model.config.context
    curve, window = [], []
    start = time.time()
    for step in range(steps):
        opt.lr = cosine_lr(step, steps, lr, warmup)
        x, y = random_batch(train_tokens, batch, context, rng)
        zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        clip_grad_norm([g for _, _, g in items], clip)
        opt.step()
        window.append(loss)
        if probes and ((step + 1) % probe_every == 0 or step + 1 == steps):
            point = {"step": step + 1, "train_loss": round(float(np.mean(window)), 4), "seconds": round(time.time() - start, 1)}
            for name, w in probes.items():
                point[name] = round(evaluate(model, w, limit=probe_windows), 4)
            curve.append(point)
            window = []
    seconds = time.time() - start
    final = curve[-1]["train_loss"] if curve else (float(np.mean(window)) if window else float("nan"))
    return TrainLog(steps, seconds, sum(p.size for _, p, _ in items), final, curve)


def train_full(model: GPT, tokens: np.ndarray, **kw) -> TrainLog:
    model.set_trainable(True)
    return finetune(model, model.named_parameters(), model.zero_grad, tokens, **kw)


def train_lora(model: GPT, adapter, tokens: np.ndarray, **kw) -> TrainLog:
    """Freeze the base model, attach the adapter and train only A and B."""
    model.set_trainable(False)
    adapter.attach(model)
    try:
        return finetune(model, adapter.named_parameters(), adapter.zero_grad, tokens, **kw)
    finally:
        model.set_trainable(True)
