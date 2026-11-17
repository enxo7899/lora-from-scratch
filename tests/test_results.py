"""docs/results.json is consistent with the files it describes."""
import json
from pathlib import Path

from lora_scratch.adapter_io import load_adapter

ROOT = Path(__file__).resolve().parents[1]
def _no_constants(name):
    raise ValueError(f"results.json contains {name}, which browsers cannot parse")


R = json.loads((ROOT / "docs" / "results.json").read_text(), parse_constant=_no_constants)


def test_untuned_old_domain_loss_matches_the_base_repo_to_two_thousandths():
    base = json.loads((ROOT / "model" / "base_results.json").read_text())["results"]["transformer"]["val_loss"]
    assert abs(R["base"]["old"] - base) < 0.002


def test_reported_trainable_counts_match_the_shipped_adapter_files():
    methods = {m["name"]: m for m in R["methods"]}
    for name, file in (("lora_r1", "code_r1"), ("lora_r4", "code_r4"), ("lora_r16", "code_r16")):
        assert methods[name]["trainable_parameters"] == load_adapter(ROOT / "model" / "adapters" / f"{file}.lora").n_parameters()
    assert methods["full"]["trainable_parameters"] == R["setup"]["base_parameters"]


def test_every_run_used_the_same_steps_and_batches():
    assert {m["name"] for m in R["methods"]} >= {"full", "lora_r1", "lora_r4", "lora_r16"}
    assert R["setup"]["tokens_per_run"] == R["setup"]["steps"] * R["setup"]["batch_size"] * R["setup"]["context"]


def test_swapping_left_the_base_weights_unchanged_and_merge_restores_exactly_from_backup():
    assert R["adapter_swap"]["base_unchanged_after_swaps"] is True
    assert R["adapter_swap"]["base_after_detach"] == R["base"]
    assert R["merge"]["bit_for_bit_with_backup"] is True
    assert R["merge"]["max_abs_weight_drift_after_unmerge"] < 1e-6
