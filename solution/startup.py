"""The start and end of `soup train`: costs moved or removed, none of the values the run produces changed.

The measured time is the whole command, and ~20 s of it are not training (README). Avoidable in
loading the model and the data:
- `datasets` fingerprints each `Dataset.map` by pickling the mapped function with dill. Soup's
  formatting function closes over the tokenizer, so the map pickles its 151k-entry vocabulary:
  ~88% of the map's time (cProfile), the tokenization itself is ~10%. For a dataset held in memory
  (Soup builds it with `Dataset.from_list`) the fingerprint only names a cache file that is never
  written, so we pass a random one, which is what `datasets` does itself for a function it cannot
  hash.
- transformers loads each checkpoint tensor with one `.to(device, dtype)`; for a CPU tensor PyTorch
  converts the dtype (bf16 -> fp16 here) on the CPU before the copy. We copy first and convert on
  the GPU: the same IEEE round-to-nearest conversion, so the same fp16 values and the same NF4
  weights.
- bitsandbytes builds its 16-entry NF4 table anew, with a host-to-device copy, for each of the 196
  linears it quantizes; we build it once per device.

Avoidable in `apply()`'s imports of Soup's libraries (`library_imports`), and at the exit:
- trl's import calls `importlib.metadata.packages_distributions()`, which reads the metadata of
  every installed distribution (hundreds on Kaggle's image). transformers' import, in the
  harness, made the same call moments earlier in the same environment and keeps the result
  (read-only); while trl imports, the call returns a copy of it.
- Python's cyclic garbage collector: the libraries import ~500k objects (modules, classes,
  functions) that live until the process exits. While they import, every full collection walks
  all of them, and the interpreter's shutdown walks them again in several passes (~0.8 s of the
  exit on our machine). They import with the collector paused and are then moved to its permanent
  generation (`gc.freeze`), so later passes skip them; at exit, one ordinary collection finalizes
  the run's garbage before shutdown, then the rest is frozen too. Reference counting frees objects
  as before; only cycles among the frozen, long-lived objects are never collected, and every
  output of the run is written and closed by then (Soup writes each rewind line under its own
  `open` and commits its database after each write).
"""

import atexit
import contextlib
import functools
import gc
import inspect

import torch


def _is_cuda(device):
    try:
        return torch.device(device).type == "cuda"
    except (RuntimeError, TypeError):
        return False


def patch_dataset_map():
    from datasets import Dataset
    from datasets.fingerprint import generate_random_fingerprint

    original = Dataset.map
    if getattr(original, "_soupk", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def map(self, *args, **kwargs):  # noqa: A001 - the method it replaces
        if not self.cache_files and \
                "new_fingerprint" not in signature.bind_partial(self, *args, **kwargs).arguments:
            kwargs["new_fingerprint"] = generate_random_fingerprint()  # nothing is cached for it
        return original(self, *args, **kwargs)

    map._soupk = True
    Dataset.map = map


def patch_checkpoint_loading():
    from transformers import core_model_loading

    original = core_model_loading._materialize_copy
    if getattr(original, "_soupk", False):
        return

    def _materialize_copy(tensor, device=None, dtype=None):
        if dtype in (torch.float16, torch.bfloat16) and device is not None and _is_cuda(device):
            loaded = original(tensor, device=device)  # the stored dtype, copied as it is
            return loaded.to(dtype=dtype)  # converted on the GPU
        return original(tensor, device, dtype)

    _materialize_copy._soupk = True
    core_model_loading._materialize_copy = _materialize_copy


def patch_nf4_table():
    import bitsandbytes.functional as bnb_functional

    original = bnb_functional.get_4bit_type
    if getattr(original, "_soupk", False):
        return
    tables = {}

    def get_4bit_type(typename, device=None, blocksize=64):
        key = (typename, None if device is None else str(torch.device(device)), blocksize)
        if key not in tables:  # read-only: a QuantState keeps it as its code
            tables[key] = original(typename, device=device, blocksize=blocksize)
        return tables[key]

    get_4bit_type._soupk = True
    bnb_functional.get_4bit_type = get_4bit_type


def _collect_then_freeze():
    gc.collect()  # the run's garbage, finalized now as shutdown's first pass would
    gc.freeze()   # shutdown's passes then skip every object still alive


@contextlib.contextmanager
def _known_package_mapping():
    """packages_distributions() returns a copy of the mapping transformers computed at import."""
    import importlib.metadata

    try:
        from transformers.utils import import_utils

        known = import_utils.PACKAGE_DISTRIBUTION_MAPPING
    except (ImportError, AttributeError):
        known = None
    if not isinstance(known, dict):
        yield
        return
    original = importlib.metadata.packages_distributions

    def packages_distributions():
        return {name: list(dists) for name, dists in known.items()}

    importlib.metadata.packages_distributions = packages_distributions
    try:
        yield
    finally:
        importlib.metadata.packages_distributions = original


@contextlib.contextmanager
def library_imports():
    """`apply()`'s imports: the collector paused and trl handed the known package mapping; what
    they created is then frozen, and the exit collects once, then freezes the rest."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        with _known_package_mapping():
            yield
    finally:
        gc.freeze()
        if enabled:
            gc.enable()
    if not getattr(_collect_then_freeze, "_registered", False):
        _collect_then_freeze._registered = True
        atexit.register(_collect_then_freeze)
