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
- bitsandbytes wraps its CPU fallback ops in `torch.compile` at import, which imports most of
  Inductor for functions a CUDA run never calls; while it imports, `torch.compile` defers each
  compile to the function's first call.
- transformers imports scikit-learn and SciPy (`_UNUSED_HERE`) for assisted generation, GLUE
  metrics and object-detection losses; those modules stand in as ones that import themselves
  when one of the names transformers took from them is called.
- trl and transformers import wandb when it is installed; a run without W&B reporting only reads
  `wandb.run`, None until `wandb.init()`. wandb stands in as a module with that one answer until
  something else of it is used, which imports it.
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
import importlib
import importlib.machinery
import importlib.util
import inspect
import sys
import types

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

def patch_peft_kbit():
    try:
        import peft.utils.other
    except ImportError:
        return
    original = peft.utils.other.prepare_model_for_kbit_training
    if getattr(original, "_soupk", False):
        return

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        embed = getattr(model, "get_input_embeddings", lambda: None)()
        if embed is not None and hasattr(embed, "to"):
            orig_to = embed.to
            embed.to = lambda *a, **kw: embed
            try:
                res = original(model, *args, **kwargs)
            finally:
                embed.to = orig_to
            return res
        return original(model, *args, **kwargs)

    prepare_model_for_kbit_training._soupk = True
    import peft
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    peft.utils.other.prepare_model_for_kbit_training = prepare_model_for_kbit_training



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
def _compile_on_first_call():
    """torch.compile(fn) returns fn compiled when it is first called, not where it is decorated."""
    original = torch.compile

    def compile(model=None, **options):  # noqa: A001 - stands in for torch.compile
        if model is None:  # @torch.compile(...)
            return lambda fn: compile(fn, **options)
        if isinstance(model, torch.nn.Module) or not callable(model):
            return original(model, **options)
        compiled = None

        @functools.wraps(model)
        def first_call_compiles(*args, **kwargs):
            nonlocal compiled
            if compiled is None:
                compiled = original(model, **options)
            return compiled(*args, **kwargs)

        return first_call_compiles

    torch.compile = compile
    try:
        yield
    finally:
        torch.compile = original


def _import_bitsandbytes():
    """bitsandbytes' import wraps its CPU fallback ops in torch.compile, which imports most of
    Inductor right away (~1 s on Kaggle's CPU) for functions a CUDA run never calls: import it
    with torch.compile deferring each compile to the function's first call (same function, same
    fallback to the plain one if compiling fails, which bitsandbytes' own wrapper handles)."""
    if "bitsandbytes" in sys.modules:
        return
    with _compile_on_first_call():
        import bitsandbytes  # noqa: F401


class _ImportedOnUse(types.ModuleType):
    """Stands in `sys.modules` for a module until it is used, then imports the real module (which
    takes its place) and answers from it. `deferred` are the names a `from <module> import ...`
    binds before any use: they become functions that import the module when called. `unimported`
    are attributes with a known value for as long as the real module has not been imported."""

    def __init__(self, name, deferred=(), unimported=None):
        super().__init__(name)
        self._unimported = dict(unimported or {})
        # find_spec(name) returns a loaded module's spec; one that exists, so it finds a module
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for attr in deferred:
            setattr(self, attr, self._deferred(attr))

    def _real(self):
        if sys.modules.get(self.__name__) is self:
            del sys.modules[self.__name__]
        return importlib.import_module(self.__name__)

    def __getattr__(self, attr):
        if attr.startswith("__"):  # the import system's probes (__path__, ...): a plain module
            raise AttributeError(attr)
        if attr in self._unimported and sys.modules.get(self.__name__) in (None, self):
            return self._unimported[attr]
        return getattr(self._real(), attr)

    def _deferred(self, attr):
        def call(*args, **kwargs):
            return getattr(self._real(), attr)(*args, **kwargs)

        call.__name__ = call.__qualname__ = attr
        return call


# Modules transformers imports at its own import when their package is installed (Kaggle's image
# has scikit-learn and SciPy), with the names it binds there, for features a training run does
# not use: assisted generation (roc_curve), the GLUE metrics of transformers.data (f1_score,
# matthews_corrcoef, pearsonr, spearmanr), the object-detection losses (linear_sum_assignment).
# ~1.5 s of imports on Kaggle's CPU.
_UNUSED_HERE = {
    "sklearn.metrics": ("roc_curve", "f1_score", "matthews_corrcoef"),
    "scipy.stats": ("pearsonr", "spearmanr"),
    "scipy.optimize": ("linear_sum_assignment",),
}


@contextlib.contextmanager
def _unused_modules_on_use():
    """While apply() imports, each module of _UNUSED_HERE stands in as one whose names import the
    real module when called (any other use imports it at once)."""
    standing = {}
    for name, deferred in _UNUSED_HERE.items():
        if name not in sys.modules and importlib.util.find_spec(name.split(".")[0]) is not None:
            standing[name] = sys.modules[name] = _ImportedOnUse(name, deferred=deferred)
    try:
        yield
    finally:
        for name, stand_in in standing.items():
            if sys.modules.get(name) is stand_in:
                del sys.modules[name]  # its names keep working: each imports on its first call


def _wandb_on_use():
    """trl's trainers import wandb when it is installed (Kaggle's image has it: ~0.35 s), and
    transformers' is_wandb_available() imports it to check: for a run that does not report to
    W&B (Soup's report_to is "none") only `wandb.run` is ever read, in TRL's model card at a
    checkpoint, and it is None until wandb.init(), which needs the real module. wandb stands in as
    a module whose `run` is None until the real module is imported; any other use imports it."""
    if "wandb" not in sys.modules and importlib.util.find_spec("wandb") is not None:
        sys.modules["wandb"] = _ImportedOnUse("wandb", unimported={"run": None})


@contextlib.contextmanager
def library_imports():
    """`apply()`'s imports: the collector paused, trl handed the known package mapping,
    bitsandbytes' compiles, scikit-learn, SciPy and wandb deferred; what they created is then
    frozen, and the exit collects once, then freezes the rest."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        _import_bitsandbytes()
        _wandb_on_use()
        with _known_package_mapping(), _unused_modules_on_use():
            yield
    finally:
        gc.freeze()
        if enabled:
            gc.enable()
    if not getattr(_collect_then_freeze, "_registered", False):
        _collect_then_freeze._registered = True
        atexit.register(_collect_then_freeze)
