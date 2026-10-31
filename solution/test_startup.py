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


def test_compile_waits_for_the_first_call(monkeypatch):
    from solution.startup import _compile_on_first_call

    compiles = []

    def fake_compile(model=None, **options):  # stands in for the real torch.compile
        compiles.append((model, options))
        return lambda *args, **kwargs: ("compiled", model(*args, **kwargs))

    monkeypatch.setattr(torch, "compile", fake_compile)
    with _compile_on_first_call():
        @torch.compile(dynamic=False)
        def double(x):
            return 2 * x

        plain = torch.compile(lambda x: x + 1)
        module = torch.nn.Linear(2, 2)
        torch.compile(module)  # a module compiles as before, at once
        assert [c[0] for c in compiles] == [module]
    assert torch.compile is fake_compile
    assert double(3) == ("compiled", 6) and double(4) == ("compiled", 8)
    assert plain(1) == ("compiled", 2)
    assert [(c[0].__name__, c[1]) for c in compiles[1:]] == [("double", {"dynamic": False}),
                                                            ("<lambda>", {})]


def test_a_module_standing_in_imports_the_real_one_when_used(tmp_path, monkeypatch):
    import sys

    from solution.startup import _ImportedOnUse

    package = tmp_path / "standin_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "metrics.py").write_text("def roc_curve(y, s):\n    return ('roc', y, s)\n\n"
                                        "def other():\n    return 'other'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules["standin_pkg.metrics"] = _ImportedOnUse("standin_pkg.metrics",
                                                        deferred=("roc_curve",))
    try:
        from standin_pkg.metrics import roc_curve  # what transformers runs at its import

        assert "standin_pkg" not in sys.modules  # nothing imported yet
        assert roc_curve(1, 2) == ("roc", 1, 2)  # the call imports the real module
        import standin_pkg.metrics as real

        assert not isinstance(real, _ImportedOnUse) and real.other() == "other"
        assert roc_curve(3, 4) == ("roc", 3, 4)
    finally:
        for name in ("standin_pkg.metrics", "standin_pkg"):
            sys.modules.pop(name, None)


def test_wandb_standing_in_answers_run_until_imported(tmp_path, monkeypatch):
    import importlib.util
    import sys

    from solution.startup import _ImportedOnUse

    package = tmp_path / "standin_wb"
    package.mkdir()
    (package / "__init__.py").write_text("run = 'a real run'\n\ndef init():\n    return 'init'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    stand_in = _ImportedOnUse("standin_wb", unimported={"run": None})
    sys.modules["standin_wb"] = stand_in
    try:
        assert importlib.util.find_spec("standin_wb") is not None  # what is_wandb_available runs
        import standin_wb

        assert standin_wb.run is None and "standin_wb" in sys.modules
        assert sys.modules["standin_wb"] is stand_in  # still not imported
        assert standin_wb.init() == "init"  # any other use imports the real module
        assert sys.modules["standin_wb"] is not stand_in
        assert standin_wb.run == "a real run"  # and from then on answers from it
    finally:
        sys.modules.pop("standin_wb", None)
