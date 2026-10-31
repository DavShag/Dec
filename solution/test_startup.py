"""The startup patches (startup.py) change when values are computed, never the values."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_bf16_to_fp16_on_the_gpu_is_the_cpu_conversion():
    gen = torch.Generator().manual_seed(0)
    values = torch.cat([
        torch.randn(1 << 20, generator=gen) * scale for scale in (1e-8, 1e-5, 1e-2, 1.0, 1e3, 1e5)
    ] + [torch.tensor([0.0, -0.0, float("inf"), -float("inf"), 65504.0, 65520.0, 6e-8, 3e-8])])
    weight = values.to(torch.bfloat16)  # a checkpoint tensor as transformers reads it, on the CPU
    on_cpu = weight.to(device="cuda", dtype=torch.float16)  # converted on the CPU, then copied
    on_gpu = weight.to(device="cuda").to(torch.float16)  # copied, then converted on the GPU
    assert torch.equal(on_cpu.view(torch.int16), on_gpu.view(torch.int16))


def test_in_memory_map_skips_the_function_hash(monkeypatch):
    from datasets import Dataset, arrow_dataset

    from solution.startup import patch_dataset_map

    patch_dataset_map()
    vocabulary = {f"token{i}": i for i in range(200_000)}  # what made the hash slow: a tokenizer

    def format_row(row):
        return {"ids": [vocabulary[word] for word in row["text"].split()]}

    def no_hashing(*args, **kwargs):
        raise AssertionError("the mapped function was hashed")

    monkeypatch.setattr(arrow_dataset, "update_fingerprint", no_hashing)  # the name map() calls
    rows = [{"text": " ".join(f"token{(7 * i + j) % 200_000}" for j in range(i % 9 + 1))}
            for i in range(64)]
    mapped = Dataset.from_list(rows).map(format_row, remove_columns=["text"])
    assert mapped["ids"] == [format_row(row)["ids"] for row in rows]


def test_trl_gets_the_package_mapping_a_scan_returns():
    import importlib.metadata

    from solution.startup import _known_package_mapping

    original = importlib.metadata.packages_distributions
    scanned = original()
    with _known_package_mapping():
        assert importlib.metadata.packages_distributions is not original
        handed = importlib.metadata.packages_distributions()
        handed["solution-test"] = []  # a copy: trl cannot change transformers' mapping
        assert importlib.metadata.packages_distributions().keys() == scanned.keys()
    assert {name: sorted(d) for name, d in handed.items() if name != "solution-test"} == \
        {name: sorted(d) for name, d in scanned.items()}
    assert importlib.metadata.packages_distributions is original


def test_library_imports_pause_the_collector_and_freeze_what_they_created():
    import gc

    from solution.startup import library_imports

    assert gc.isenabled()
    frozen_before = gc.get_freeze_count()
    with library_imports():
        assert not gc.isenabled()
        made = [[] for _ in range(1000)]  # long-lived objects created while importing
    assert gc.isenabled()
    assert gc.get_freeze_count() >= frozen_before + len(made)
    gc.unfreeze()  # leave the test process as it was
