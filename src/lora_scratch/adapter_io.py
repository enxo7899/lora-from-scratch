"""Adapter file format shared with the browser.

b"LORA", uint32 little-endian header length, UTF-8 JSON header (config, base shape, tensor table, free-form
meta), zero padding to 4 bytes, then every tensor as little-endian float32 in header order.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from tiny_gpt.model import GPT

from .lora import LoRAAdapter, LoRAConfig, LoRADelta

MAGIC = b"LORA"


class AdapterFileError(ValueError):
    """Raised when an adapter file is malformed."""


def save_adapter(adapter: LoRAAdapter, model: GPT, path: str | Path) -> int:
    tensors, blobs, offset = [], [], 0
    for name, p, _ in adapter.named_parameters():
        data = np.ascontiguousarray(p, dtype="<f4").tobytes()
        tensors.append({"name": name, "shape": list(p.shape), "offset": offset, "length": len(data)})
        blobs.append(data)
        offset += len(data)
    header = json.dumps({"format": 1, "dtype": "float32", "config": adapter.config.to_dict(), "base": model.config.to_dict(),
                         "meta": adapter.meta, "tensors": tensors}, separators=(",", ":")).encode("utf-8")
    header += b" " * (-(len(header) + 8) % 4)
    payload = MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs)
    Path(path).write_bytes(payload)
    return len(payload)


def load_adapter(path: str | Path, dtype=np.float32) -> LoRAAdapter:
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise AdapterFileError("not a LoRA adapter file")
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen].decode("utf-8"))
    base = 8 + hlen
    config = LoRAConfig(rank=header["config"]["rank"], alpha=header["config"]["alpha"], targets=tuple(header["config"]["targets"]))
    arrays = {}
    for t in header["tensors"]:
        chunk = raw[base + t["offset"]: base + t["offset"] + t["length"]]
        if len(chunk) != t["length"]:
            raise AdapterFileError(f"file truncated while reading {t['name']}")
        arrays[t["name"]] = np.frombuffer(chunk, dtype="<f4").reshape(t["shape"]).astype(dtype)
    deltas = {}
    for name in sorted({n.rsplit(".", 1)[0] for n in arrays}):
        if f"{name}.A" not in arrays or f"{name}.B" not in arrays:
            raise AdapterFileError(f"{name} is missing its A or B tensor")
        a, b = arrays[f"{name}.A"], arrays[f"{name}.B"]
        d = LoRADelta(a.shape[0], b.shape[1], LoRAConfig(rank=max(1, a.shape[1]), alpha=config.alpha, targets=config.targets), None, dtype)
        d.A, d.B, d.gA, d.gB = a, b, np.zeros_like(a), np.zeros_like(b)
        d.config = config
        deltas[name] = d
    return LoRAAdapter(config, deltas, header.get("meta", {}))


def attach_from_file(model: GPT, path: str | Path) -> LoRAAdapter:
    """Load an adapter and attach it, raising AdapterShapeError with a plain message if it does not fit."""
    adapter = load_adapter(path, dtype=model.children["emb"].wte.dtype)
    adapter.attach(model)
    return adapter

