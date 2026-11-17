"""Build the new-domain corpora and the old-domain validation set. Nothing is downloaded.

  code.txt   Python standard-library source code with the docstrings removed (PSF-2.0).
  js.txt     JavaScript source of four MIT-licensed command-line packages found in the global npm folder.
  old_domain_val.txt  the validation documents of the base model's own corpus (Python documentation prose).

Documents are separated by a line holding only <<<DOC>>>. Output is ASCII. Run `python3 data/build_domains.py --help`.
"""
from __future__ import annotations

import argparse
import ast
import codecs
import subprocess
import sysconfig
from pathlib import Path

import numpy as np

DOC_SEPARATOR = "<<<DOC>>>"
MAX_CHARS = 6000
MIN_CHARS = 1500
SKIP_PARTS = {"test", "tests", "idlelib", "lib2to3", "site-packages", "dist-packages", "turtledemo", "__pycache__", "ensurepip", "pydoc_data"}
JS_PACKAGES = ("eslint", "nodemon", "http-server", "serve")
# Documents that name commercial AI vendors are left out; the names are stored rot13-encoded.
EXCLUDED = tuple(codecs.decode(w, "rot13") for w in ("pynhqr", "naguebcvp", "bcranv", "trzvav"))


def to_ascii(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)


def strip_docstrings(source: str) -> str:
    """Remove module, class and function docstrings by deleting their lines."""
    tree = ast.parse(source)
    drop: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                drop.update(range(first.lineno, first.end_lineno + 1))
    lines = [ln for i, ln in enumerate(source.split("\n"), 1) if i not in drop]
    return "\n".join(lines)


def chunk(text: str) -> str:
    """Whole lines up to MAX_CHARS, with runs of blank lines collapsed."""
    out, size, blank = [], 0, 0
    for ln in text.split("\n"):
        blank = blank + 1 if not ln.strip() else 0
        if blank > 1:
            continue
        if size + len(ln) + 1 > MAX_CHARS:
            break
        out.append(ln.rstrip())
        size += len(ln) + 1
    return "\n".join(out).strip("\n")


def acceptable(text: str) -> bool:
    low = text.lower()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    repetitive = not lines or len(set(lines)) / len(lines) < 0.7  # data tables, generated files
    return len(text) >= MIN_CHARS and not repetitive and not any(w in low for w in EXCLUDED)


def python_documents() -> list[str]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    docs = []
    for path in sorted(stdlib.rglob("*.py")):
        if SKIP_PARTS & set(path.relative_to(stdlib).parts):
            continue
        try:
            text = chunk(to_ascii(strip_docstrings(path.read_text(encoding="utf-8"))))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        if acceptable(text):
            docs.append(text)
    return docs


def javascript_documents() -> list[str]:
    root = Path(subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, check=True).stdout.strip())
    docs = []
    for pkg in JS_PACKAGES:
        for path in sorted((root / pkg).rglob("*.js")):
            parts = set(path.relative_to(root / pkg).parts)
            if "node_modules" in parts or "test" in parts or "tests" in parts:
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
            if not raw or len(raw) / (raw.count("\n") + 1) > 160:  # minified or bundled
                continue
            text = chunk(to_ascii(raw))
            if acceptable(text):
                docs.append(text)
    return docs


def pick(docs: list[str], budget: int, seed: int) -> list[str]:
    order = np.random.default_rng(seed).permutation(len(docs))
    kept, total = [], 0
    for i in order:
        if total + len(docs[i]) > budget:
            continue
        kept.append(docs[i])
        total += len(docs[i])
    return kept


def write(path: Path, docs: list[str]) -> None:
    path.write_text(f"\n{DOC_SEPARATOR}\n".join(docs) + "\n", encoding="ascii")
    print(f"{path.name}: {len(docs)} documents, {sum(map(len, docs))} characters")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    ap.add_argument("--budget", type=int, default=1_500_000, help="characters kept per new domain")
    ap.add_argument("--base-corpus", help="path to the base model's corpus.txt, to extract old_domain_val.txt")
    args = ap.parse_args()
    out = Path(args.out)
    write(out / "code.txt", pick(python_documents(), args.budget, seed=1))
    write(out / "js.txt", pick(javascript_documents(), args.budget, seed=2))
    if args.base_corpus:
        from tiny_gpt.data import load_documents, split_documents

        _, val = split_documents(load_documents(args.base_corpus), seed=1337)
        write(out / "old_domain_val.txt", val)


if __name__ == "__main__":
    main()
