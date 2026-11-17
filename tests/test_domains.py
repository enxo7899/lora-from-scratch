"""The new-domain corpora: ASCII, split by document, nested subsets for the data-efficiency runs."""
from pathlib import Path

import numpy as np

from lora_scratch.finetune import load_domain
from tiny_gpt.bpe import BPETokenizer
from tiny_gpt.data import load_documents

ROOT = Path(__file__).resolve().parents[1]


def test_domain_files_are_plain_ascii_documents():
    for name in ("code", "js", "old_domain_val"):
        text = (ROOT / "data" / f"{name}.txt").read_text(encoding="ascii")
        assert text.count("<<<DOC>>>") >= 40


def test_train_dev_and_val_share_no_document():
    for name in ("code", "js"):
        d = load_domain(ROOT / "data" / f"{name}.txt", name)
        parts = [set(d.train_docs), set(d.dev_docs), set(d.val_docs)]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert len(d.train_docs) + len(d.dev_docs) + len(d.val_docs) == len(load_documents(ROOT / "data" / f"{name}.txt"))


def test_the_split_is_the_same_every_time_and_subsets_are_nested():
    tok = BPETokenizer.load(ROOT / "model" / "tokenizer.json")
    a, b = load_domain(ROOT / "data" / "code.txt", "code"), load_domain(ROOT / "data" / "code.txt", "code")
    assert a.val_docs == b.val_docs
    ten, thirty, full = (a.tokens(tok, "train", f) for f in (0.1, 0.3, 1.0))
    assert len(ten) < len(thirty) < len(full)
    assert np.array_equal(ten, thirty[: len(ten)])


def test_new_domains_are_unlike_the_base_corpus_in_the_tokenizer_they_share():
    tok = BPETokenizer.load(ROOT / "model" / "tokenizer.json")
    old = load_documents(ROOT / "data" / "old_domain_val.txt")
    code = load_documents(ROOT / "data" / "code.txt")
    bytes_per_token = lambda docs: sum(len(d) for d in docs) / sum(len(tok.encode(d)) for d in docs)
    assert bytes_per_token(code) < bytes_per_token(old) - 0.2  # code is harder for a tokenizer trained on prose
