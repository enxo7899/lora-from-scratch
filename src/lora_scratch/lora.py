"""Low-rank adapters (LoRA) with a hand-written backward pass.

A frozen linear layer computes y = x W + b. An adapter adds a low-rank correction:

    y = x W + b + s * (x A) B        A: (n_in, r), B: (r, n_out), s = alpha / r * scale_mult

B starts at zero, so a fresh adapter changes nothing. Only A and B are trained.
`merge` folds s * A B into W so inference needs no extra work; `unmerge` takes it out again.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from tiny_gpt.layers import Array
from tiny_gpt.model import GPT

#: Linear layers inside every transformer block that an adapter may target.
TARGETS = ("attn.qkv", "attn.proj", "mlp.fc", "mlp.proj")
ATTENTION_TARGETS = ("attn.qkv", "attn.proj")


class AdapterShapeError(ValueError):
    """Raised when adapter tensors disagree with each other or with the base model."""


class AdapterStateError(RuntimeError):
    """Raised for an impossible merge/unmerge order, for example merging twice."""


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float | None = None  # defaults to 2 * rank, so the scale alpha / rank is 2 at every rank
    targets: tuple[str, ...] = TARGETS

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        unknown = set(self.targets) - set(TARGETS)
        if unknown or not self.targets:
            raise ValueError(f"targets must be a non-empty subset of {TARGETS}, got {self.targets}")
        if self.alpha is None:
            object.__setattr__(self, "alpha", 2.0 * self.rank)

    @property
    def scale(self) -> float:
        return float(self.alpha) / self.rank

    def to_dict(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha, "targets": list(self.targets)}


class LoRADelta:
    """The correction for one linear layer, with cached activations and gradients for backward."""

    def __init__(self, n_in: int, n_out: int, config: LoRAConfig, rng: np.random.Generator | None = None, dtype=np.float32):
        r = config.rank
        std = 1.0 / np.sqrt(n_in)
        self.A: Array = (rng.standard_normal((n_in, r)) * std if rng is not None else np.zeros((n_in, r))).astype(dtype)
        self.B: Array = np.zeros((r, n_out), dtype=dtype)
        self.gA, self.gB = np.zeros_like(self.A), np.zeros_like(self.B)
        self.config = config
        self.scale_mult = 1.0
        self._x: Array | None = None
        self._xa: Array | None = None

    @property
    def scale(self) -> float:
        return self.config.scale * self.scale_mult

    def delta(self, x: Array) -> Array:
        return self.scale * ((x @ self.A) @ self.B)

    def forward(self, x: Array) -> Array:
        self._x, self._xa = x, x @ self.A
        return self.scale * (self._xa @ self.B)

    def backward(self, dy: Array) -> Array:
        """Accumulate dA and dB; return the gradient this branch sends to the layer input."""
        s = self.scale
        x2 = self._x.reshape(-1, self._x.shape[-1])
        xa2 = self._xa.reshape(-1, self._xa.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        dxa = s * (dy2 @ self.B.T)
        self.gB += s * (xa2.T @ dy2)
        self.gA += x2.T @ dxa
        return (dxa @ self.A.T).reshape(self._x.shape)

    def product(self) -> Array:
        """The dense update s * A B that merging adds to the base weight."""
        return self.scale * (self.A @ self.B)


@dataclass
class LoRAAdapter:
    """All deltas of one adapter, keyed by layer name such as 'block2.attn.qkv'."""

    config: LoRAConfig
    deltas: dict[str, LoRADelta]
    meta: dict = field(default_factory=dict)
    _backups: dict[str, Array] | None = field(default=None, repr=False)
    _merged_scale: float | None = field(default=None, repr=False)

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def create(cls, model: GPT, config: LoRAConfig, seed: int = 0, meta: dict | None = None) -> "LoRAAdapter":
        rng = np.random.default_rng(seed)
        dtype = model.children["emb"].wte.dtype
        deltas = {}
        for name, lin in target_layers(model, config.targets).items():
            deltas[name] = LoRADelta(lin.w.shape[0], lin.w.shape[1], config, rng, dtype)
        return cls(config, deltas, meta or {})

    # ---- parameters ---------------------------------------------------------------------
    def named_parameters(self) -> Iterator[tuple[str, Array, Array]]:
        for name, d in self.deltas.items():
            yield f"{name}.A", d.A, d.gA
            yield f"{name}.B", d.B, d.gB

    def n_parameters(self) -> int:
        return sum(p.size for _, p, _ in self.named_parameters())

    def zero_grad(self) -> None:
        for _, _, g in self.named_parameters():
            g.fill(0.0)

    def set_scale(self, scale_mult: float) -> None:
        """Multiply every delta by scale_mult on top of alpha / rank (1.0 is the trained strength, 0 switches it off)."""
        for d in self.deltas.values():
            d.scale_mult = float(scale_mult)

    @property
    def merged(self) -> bool:
        return self._merged_scale is not None

    # ---- checking and attaching ----------------------------------------------------------------
    def validate(self, model: GPT) -> None:
        """Raise AdapterShapeError unless every tensor fits the model and the stated rank."""
        r = self.config.rank
        layers = target_layers(model, self.config.targets)
        if set(layers) != set(self.deltas):
            missing = sorted(set(layers) ^ set(self.deltas))[:3]
            raise AdapterShapeError(f"adapter and model disagree about target layers, for example {missing}")
        for name, d in self.deltas.items():
            n_in, n_out = layers[name].w.shape
            if d.A.ndim != 2 or d.B.ndim != 2:
                raise AdapterShapeError(f"{name}: A and B must be matrices")
            if d.A.shape[1] != d.B.shape[0]:
                raise AdapterShapeError(f"{name}: A has rank {d.A.shape[1]} but B has rank {d.B.shape[0]}; they must match")
            if d.A.shape[1] != r:
                raise AdapterShapeError(f"{name}: tensors have rank {d.A.shape[1]} but the adapter header says rank {r}")
            if d.A.shape[0] != n_in or d.B.shape[1] != n_out:
                raise AdapterShapeError(f"{name}: adapter expects a {d.A.shape[0]} -> {d.B.shape[1]} layer, the model layer is {n_in} -> {n_out}")

    def attach(self, model: GPT) -> None:
        self.validate(model)
        if self.merged:
            raise AdapterStateError("adapter is merged into the weights; unmerge it before attaching")
        for name, lin in target_layers(model, self.config.targets).items():
            lin.lora = self.deltas[name]

    def detach(self, model: GPT) -> None:
        for lin in target_layers(model, self.config.targets).values():
            lin.lora = None

    # ---- merging -------------------------------------------------------------------------------
    def merge(self, model: GPT, keep_backup: bool = False) -> None:
        """Add s * A B to every target weight and stop using the separate branch.

        With keep_backup the original weights are copied first, so unmerge() restores them bit for bit.
        Without it unmerge() subtracts the same product, which is exact in real numbers but can leave
        float32 rounding differences of a few units in the last place.
        """
        if self.merged:
            raise AdapterStateError("adapter is already merged")
        self.validate(model)
        layers = target_layers(model, self.config.targets)
        self._backups = {n: lin.w.copy() for n, lin in layers.items()} if keep_backup else None
        for name, lin in layers.items():
            lin.w += self.deltas[name].product().astype(lin.w.dtype)
            lin.lora = None
        self._merged_scale = next(iter(self.deltas.values())).scale_mult

    def unmerge(self, model: GPT) -> None:
        if not self.merged:
            raise AdapterStateError("adapter is not merged")
        layers = target_layers(model, self.config.targets)
        for name, lin in layers.items():
            d = self.deltas[name]
            if self._backups is not None:
                lin.w[...] = self._backups[name]
            else:
                current = d.scale_mult
                d.scale_mult = self._merged_scale
                lin.w -= d.product().astype(lin.w.dtype)
                d.scale_mult = current
        self._backups, self._merged_scale = None, None
        self.attach(model)


def target_layers(model: GPT, targets: tuple[str, ...]) -> dict:
    """Map 'block{i}.{target}' to the Linear layer it names."""
    out = {}
    for i, block in enumerate(model.blocks):
        for target in targets:
            part, leaf = target.split(".")
            out[f"block{i}.{target}"] = block.children["attn" if part == "attn" else "mlp"].children[leaf]
    return out


def checksum(model: GPT) -> str:
    """SHA-256 over the raw bytes of every parameter, to show weights are identical bit for bit."""
    import hashlib

    h = hashlib.sha256()
    for name, p, _ in model.named_parameters():
        h.update(name.encode())
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()
