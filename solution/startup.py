"""The start of `soup train`: three costs moved or removed, none of the values the run produces changed.

The measured time is the whole command, and ~20 s of it are not training (README). Three of those
costs are avoidable:
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
"""

import functools
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
