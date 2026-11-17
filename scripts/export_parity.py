"""Write model/parity_lora.json: what the Python model outputs with each shipped adapter, for the JavaScript to match.

  python scripts/export_parity.py

Logits are computed in float64 from the float32 files on disk, exactly the numbers the browser reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_scratch.adapter_io import load_adapter  # noqa: E402
from lora_scratch.finetune import load_domain  # noqa: E402
from tiny_gpt.bpe import BPETokenizer  # noqa: E402
from tiny_gpt.data import load_documents  # noqa: E402
from tiny_gpt.infer import generate  # noqa: E402
from tiny_gpt.weights import load_weights  # noqa: E402

PROMPTS = {
    "code": "def read_file(path):\n    with open(path) as f:\n        return f.read()",
    "js": "function readFile(path) {\n    const text = fs.readFileSync(path, \"utf8\");",
}
CASES = [("code_r4", "code", 1.0), ("code_r4", "code", 1.5), ("code_r16", "code", 1.0), ("js_r4", "js", 1.0), ("js_r4", "code", 0.5)]
PASSAGE_TOKENS = 128


def passage(tok: BPETokenizer, docs: list[str]) -> dict:
    for doc in docs:
        ids = tok.encode(doc)
        if len(ids) >= PASSAGE_TOKENS:
            return {"ids": ids[:PASSAGE_TOKENS], "text": tok.decode(ids[:PASSAGE_TOKENS])}
    raise ValueError("no document is long enough")


def nll(model, ids: list[int]) -> float:
    _, loss = model.forward(np.asarray(ids[:-1])[None], np.asarray(ids[1:])[None])
    return float(loss)


def main() -> None:
    tok = BPETokenizer.load(ROOT / "model" / "tokenizer.json")
    model = load_weights(ROOT / "model" / "weights.bin", dtype=np.float64)
    adapters = {n: load_adapter(ROOT / "model" / "adapters" / f"{n}.lora", dtype=np.float64) for n in {c[0] for c in CASES}}
    passages = {
        "docs": passage(tok, load_documents(ROOT / "data" / "old_domain_val.txt")),
        "code": passage(tok, load_domain(ROOT / "data" / "code.txt", "code").val_docs),
        "js": passage(tok, load_domain(ROOT / "data" / "js.txt", "js").val_docs),
    }
    out = {"prompts": {k: {"text": v, "ids": tok.encode(v)} for k, v in PROMPTS.items()}, "passages": passages, "cases": [], "base": {}}
    for k, p in out["prompts"].items():
        logits, _ = model.forward(np.asarray(p["ids"])[None])
        out["base"][k] = {"logits": [[round(float(v), 5) for v in row] for row in logits[0]],
                          "passage_loss": {n: round(nll(model, q["ids"]), 6) for n, q in passages.items()} if k == "code" else None}
    for name, prompt, scale in CASES:
        ad = adapters[name]
        ad.set_scale(scale)
        ad.attach(model)
        ids = out["prompts"][prompt]["ids"]
        logits, _ = model.forward(np.asarray(ids)[None])
        greedy = generate(model, ids, 24, temperature=0)
        losses = {n: round(nll(model, q["ids"]), 6) for n, q in passages.items()}
        ad.merge(model)
        merged, _ = model.forward(np.asarray(ids)[None])
        ad.unmerge(model)
        ad.detach(model)
        ad.set_scale(1.0)
        out["cases"].append({"adapter": name, "prompt": prompt, "scale": scale, "logits": [[round(float(v), 5) for v in row] for row in logits[0]],
                             "merged_last_logits": [round(float(v), 5) for v in merged[0, -1]], "greedy": greedy, "passage_loss": losses})
    path = ROOT / "model" / "parity_lora.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
