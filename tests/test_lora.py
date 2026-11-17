"""LoRA: gradients, merge and unmerge, freezing, shape errors and the adapter file."""
import numpy as np
import pytest

from lora_scratch.adapter_io import AdapterFileError, load_adapter, save_adapter
from lora_scratch.lora import AdapterShapeError, AdapterStateError, LoRAAdapter, LoRAConfig, checksum, target_layers
from tiny_gpt.layers import Linear
from tiny_gpt.model import GPT, GPTConfig
from tiny_gpt.optim import AdamW

from test_gradients import numeric_grad, rel_err, tiny_model

IDS = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
TARGETS = np.array([[2, 3, 4, 5, 6, 7], [5, 4, 3, 2, 1, 0]])


def trained_adapter(model: GPT, rank: int = 3, seed: int = 11, **kw) -> LoRAAdapter:
    """An adapter whose B is no longer zero, so every path through it matters."""
    adapter = LoRAAdapter.create(model, LoRAConfig(rank=rank, **kw), seed=seed)
    rng = np.random.default_rng(seed)
    for d in adapter.deltas.values():
        d.B[...] = rng.standard_normal(d.B.shape) * 0.3
    return adapter


def logits(model: GPT) -> np.ndarray:
    return model.forward(IDS)[0].copy()


# ---- gradients -------------------------------------------------------------------------------------
def test_lora_layer_gradient_matches_finite_differences_for_input_A_and_B():
    rng = np.random.default_rng(1)
    lin = Linear(5, 4, rng, std=0.5, dtype=np.float64)
    lin.set_trainable(False)
    from lora_scratch.lora import LoRADelta

    d = LoRADelta(5, 4, LoRAConfig(rank=2, alpha=3.0), rng, np.float64)
    d.B[...] = rng.standard_normal(d.B.shape)
    lin.lora = d
    x = rng.standard_normal((2, 3, 5))
    r = rng.standard_normal((2, 3, 4))
    loss = lambda: float((lin.forward(x) * r).sum())
    lin.forward(x)
    dx = lin.backward(r)
    assert rel_err(dx, numeric_grad(loss, x)) < 1e-6
    assert rel_err(d.gA, numeric_grad(loss, d.A)) < 1e-6
    assert rel_err(d.gB, numeric_grad(loss, d.B)) < 1e-6


def test_whole_model_gradient_of_every_adapter_matrix_matches_finite_differences():
    model = tiny_model()
    adapter = trained_adapter(model)
    adapter.attach(model)
    model.set_trainable(False)
    adapter.zero_grad()
    model.forward(IDS, TARGETS)
    model.backward()
    for name, p, g in adapter.named_parameters():
        num = numeric_grad(lambda: model.forward(IDS, TARGETS)[1], p)
        assert rel_err(g, num) < 1e-5, name


@pytest.mark.parametrize("seed", range(4))
def test_random_ranks_targets_and_scales_pass_the_gradient_check(seed):
    rng = np.random.default_rng(seed)
    model = tiny_model(seed)
    targets = tuple(rng.choice(["attn.qkv", "attn.proj", "mlp.fc", "mlp.proj"], size=int(rng.integers(1, 5)), replace=False))
    adapter = trained_adapter(model, rank=int(rng.integers(1, 5)), seed=seed, targets=targets)
    adapter.set_scale(float(rng.uniform(0, 2)))
    adapter.attach(model)
    adapter.zero_grad()
    model.forward(IDS, TARGETS)
    model.backward()
    for name, p, g in adapter.named_parameters():
        assert rel_err(g, numeric_grad(lambda: model.forward(IDS, TARGETS)[1], p)) < 1e-5, name


def test_a_fresh_adapter_has_zero_gradient_for_A_and_nonzero_for_B():
    model = tiny_model()
    adapter = LoRAAdapter.create(model, LoRAConfig(rank=2), seed=0)
    adapter.attach(model)
    model.set_trainable(False)
    adapter.zero_grad()
    model.forward(IDS, TARGETS)
    model.backward()
    assert all(not d.gA.any() for d in adapter.deltas.values())
    assert all(d.gB.any() for d in adapter.deltas.values())


# ---- freezing and updates --------------------------------------------------------------------------
def test_zero_initialised_B_leaves_the_model_output_exactly_unchanged():
    model = tiny_model()
    before = logits(model)
    adapter = LoRAAdapter.create(model, LoRAConfig(rank=4), seed=3)
    adapter.attach(model)
    assert np.array_equal(logits(model), before)


def test_only_adapter_parameters_change_during_an_optimiser_step():
    model = tiny_model()
    adapter = trained_adapter(model)
    adapter.attach(model)
    model.set_trainable(False)
    base_hash, adapter_before = checksum(model), [p.copy() for _, p, _ in adapter.named_parameters()]
    opt = AdamW(adapter.named_parameters(), lr=0.05, weight_decay=0.0)
    for _ in range(3):
        adapter.zero_grad()
        model.forward(IDS, TARGETS)
        model.backward()
        opt.step()
    assert checksum(model) == base_hash
    assert all(not np.array_equal(b, p) for b, (_, p, _) in zip(adapter_before, adapter.named_parameters()))


def test_frozen_base_accumulates_no_gradients():
    model = tiny_model()
    adapter = trained_adapter(model)
    adapter.attach(model)
    model.set_trainable(False)
    model.zero_grad()
    model.forward(IDS, TARGETS)
    model.backward()
    assert all(not g.any() for _, _, g in model.named_parameters())


def test_lora_training_lowers_the_loss_on_a_small_batch():
    model = tiny_model()
    adapter = LoRAAdapter.create(model, LoRAConfig(rank=2), seed=1)
    adapter.attach(model)
    model.set_trainable(False)
    opt = AdamW(adapter.named_parameters(), lr=0.05, weight_decay=0.0)
    first = model.forward(IDS, TARGETS)[1]
    for _ in range(40):
        adapter.zero_grad()
        model.forward(IDS, TARGETS)
        model.backward()
        opt.step()
    assert model.forward(IDS, TARGETS)[1] < first - 0.3


# ---- scale, merge, unmerge -------------------------------------------------------------------------
def test_scale_zero_equals_the_base_and_the_delta_is_linear_in_scale():
    model = tiny_model()
    base = logits(model)
    adapter = trained_adapter(model)
    adapter.attach(model)
    adapter.set_scale(0.0)
    assert np.array_equal(logits(model), base)
    d = next(iter(adapter.deltas.values()))
    x = np.random.default_rng(0).standard_normal((2, 8))
    adapter.set_scale(1.0)
    one = d.delta(x)
    adapter.set_scale(2.0)
    assert np.allclose(d.delta(x), 2 * one)


def test_merged_weights_give_the_same_output_as_the_live_adapter():
    model = tiny_model()
    adapter = trained_adapter(model)
    adapter.attach(model)
    live = logits(model)
    adapter.merge(model)
    assert np.allclose(logits(model), live, atol=1e-10)


def test_merge_then_unmerge_restores_float64_weights_to_rounding_error():
    model = tiny_model()
    original = {n: p.copy() for n, p, _ in model.named_parameters()}
    adapter = trained_adapter(model)
    adapter.merge(model)
    assert any(not np.array_equal(original[n], p) for n, p, _ in model.named_parameters())
    adapter.unmerge(model)
    assert max(float(np.abs(original[n] - p).max()) for n, p, _ in model.named_parameters()) < 1e-14


def test_merge_with_backup_then_unmerge_is_bit_for_bit_in_float32():
    model = GPT(GPTConfig(vocab_size=11, context=6, n_layers=2, n_heads=2, dim=8), seed=2, dtype=np.float32)
    before = checksum(model)
    adapter = trained_adapter(model)
    adapter.merge(model, keep_backup=True)
    assert checksum(model) != before
    adapter.unmerge(model)
    assert checksum(model) == before


def test_float32_unmerge_without_backup_stays_within_one_millionth():
    model = GPT(GPTConfig(vocab_size=11, context=6, n_layers=2, n_heads=2, dim=8), seed=2, dtype=np.float32)
    original = {n: p.copy() for n, p, _ in model.named_parameters()}
    adapter = trained_adapter(model)
    adapter.merge(model)
    adapter.unmerge(model)
    assert max(float(np.abs(original[n] - p).max()) for n, p, _ in model.named_parameters()) < 1e-6


def test_unmerge_removes_the_scale_that_was_merged_even_if_the_slider_moved():
    model = tiny_model()
    original = {n: p.copy() for n, p, _ in model.named_parameters()}
    adapter = trained_adapter(model)
    adapter.set_scale(1.5)
    adapter.merge(model)
    adapter.set_scale(0.3)
    adapter.unmerge(model)
    assert max(float(np.abs(original[n] - p).max()) for n, p, _ in model.named_parameters()) < 1e-14


def test_merging_twice_or_unmerging_an_unmerged_adapter_is_an_error():
    model = tiny_model()
    adapter = trained_adapter(model)
    with pytest.raises(AdapterStateError):
        adapter.unmerge(model)
    adapter.merge(model)
    with pytest.raises(AdapterStateError):
        adapter.merge(model)
    with pytest.raises(AdapterStateError):
        adapter.attach(model)


# ---- swapping, targets, counts ---------------------------------------------------------------------
def test_swapping_adapters_switches_behaviour_and_detaching_restores_the_base_exactly():
    model = tiny_model()
    base = logits(model)
    first, second = trained_adapter(model, seed=1), trained_adapter(model, seed=2)
    first.attach(model)
    out_first = logits(model)
    second.attach(model)
    out_second = logits(model)
    assert not np.allclose(out_first, out_second) and not np.allclose(out_first, base)
    first.attach(model)
    assert np.array_equal(logits(model), out_first)
    first.detach(model)
    assert np.array_equal(logits(model), base)


def test_attention_only_adapter_leaves_the_mlp_layers_untouched():
    model = tiny_model()
    adapter = trained_adapter(model, targets=("attn.qkv", "attn.proj"))
    adapter.attach(model)
    assert all(b.children["mlp"].children[n].lora is None for b in model.blocks for n in ("fc", "proj"))
    assert all(b.children["attn"].children[n].lora is not None for b in model.blocks for n in ("qkv", "proj"))


def test_parameter_count_is_rank_times_input_plus_output_width_per_layer():
    model = tiny_model()
    adapter = LoRAAdapter.create(model, LoRAConfig(rank=3))
    expected = sum(3 * (lin.w.shape[0] + lin.w.shape[1]) for lin in target_layers(model, adapter.config.targets).values())
    assert adapter.n_parameters() == expected


def test_alpha_defaults_to_twice_the_rank_and_bad_settings_are_rejected():
    assert LoRAConfig(rank=8).scale == 2.0
    with pytest.raises(ValueError):
        LoRAConfig(rank=0)
    with pytest.raises(ValueError):
        LoRAConfig(rank=2, targets=("embedding",))


# ---- errors and files ------------------------------------------------------------------------------
def test_mismatched_rank_between_A_and_B_raises_a_clear_shape_error():
    model = tiny_model()
    adapter = trained_adapter(model, rank=3)
    name, d = next(iter(adapter.deltas.items()))
    d.B = np.zeros((5, d.B.shape[1]))
    with pytest.raises(AdapterShapeError, match=r"A has rank 3 but B has rank 5"):
        adapter.attach(model)


def test_adapter_for_a_wider_model_is_refused_without_changing_anything():
    small = tiny_model()
    wide = GPT(GPTConfig(vocab_size=11, context=6, n_layers=2, n_heads=2, dim=16), seed=1, dtype=np.float64)
    adapter = trained_adapter(wide)
    before = logits(small)
    with pytest.raises(AdapterShapeError, match="expects a 16"):
        adapter.attach(small)
    assert np.array_equal(logits(small), before)


def test_adapter_file_round_trip_gives_identical_logits(tmp_path):
    model = tiny_model()
    adapter = trained_adapter(model, rank=2)
    adapter.meta = {"domain": "demo"}
    adapter.attach(model)
    live = logits(model)
    adapter.detach(model)
    path = tmp_path / "a.lora"
    size = save_adapter(adapter, model, path)
    loaded = load_adapter(path, dtype=np.float64)
    loaded.attach(model)
    assert path.stat().st_size == size and loaded.meta == {"domain": "demo"}
    assert np.allclose(logits(model), live, atol=1e-6)  # float32 on disk


def test_damaged_adapter_files_are_rejected(tmp_path):
    model = tiny_model()
    path = tmp_path / "a.lora"
    save_adapter(trained_adapter(model), model, path)
    raw = path.read_bytes()
    (tmp_path / "short.lora").write_bytes(raw[:-40])
    (tmp_path / "wrong.lora").write_bytes(b"NOPE" + raw[4:])
    with pytest.raises(AdapterFileError, match="truncated"):
        load_adapter(tmp_path / "short.lora")
    with pytest.raises(AdapterFileError, match="not a LoRA"):
        load_adapter(tmp_path / "wrong.lora")


def test_rank_written_in_the_file_header_must_match_the_tensors(tmp_path):
    import json
    import struct

    model = tiny_model()
    path = tmp_path / "a.lora"
    save_adapter(trained_adapter(model, rank=3), model, path)
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen])
    header["config"]["rank"] = 5
    new = json.dumps(header, separators=(",", ":")).encode()
    new += b" " * (-(len(new) + 8) % 4)
    path.write_bytes(b"LORA" + struct.pack("<I", len(new)) + new + raw[8 + hlen:])
    with pytest.raises(AdapterShapeError, match="header says rank 5"):
        load_adapter(path, dtype=np.float64).attach(model)
