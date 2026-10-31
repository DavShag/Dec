"""Compile our kernels on a background thread while Soup loads the model.

The harness times the whole `soup train` command and starts every run with an empty Triton cache,
so each kernel variant would otherwise compile on its first launch inside the first training step
(several seconds on Kaggle's CPUs). `JITFunction.warmup` compiles without launching, from dtypes
alone, so this thread needs no GPU memory and runs the same way in every run. A variant the thread
has not reached yet simply compiles on first launch, as it would without it.

Compiling is not all of it: Triton builds each kernel's launcher, a small C extension (one run of
the C compiler per kernel signature), only at the kernel's first launch. On the T4 that was ~2.5 s
of the first step, after the warm-up had compiled everything. So two threads compile and two more
build each launcher into the Triton cache as soon as its kernel is compiled; the first launch then
only loads it. They start at the end of `apply()`, once Soup's libraries are imported (started
before those imports, they slowed them down more than they gained). Kaggle, one session, step 1:
3.4 s with launchers built in the step, 1.2 s with one compiling and one building thread from the
model's loading on, 0.74-0.78 s now.

The arguments mirror the real launches (same dtypes, same integer divisibility, same constexprs and
launch options), or the compiled variant would not be the one training asks for.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import triton

GRID = (1,)
COMPILE_THREADS, LAUNCHER_THREADS = 2, 2  # Kaggle's T4 machines have 4 vCPUs
_started = False


def _model_dims():
    """(hidden, vocab, heads, kv_heads, head_dim) of the model `soup train --config` will load."""
    args = sys.argv
    if "--config" not in args or args.index("--config") + 1 >= len(args):
        return None
    import yaml
    from transformers import AutoConfig

    with open(args[args.index("--config") + 1], encoding="utf-8") as handle:
        base = yaml.safe_load(handle).get("base")
    config = AutoConfig.from_pretrained(base)
    if getattr(config, "model_type", None) != "qwen2":
        return None
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return (config.hidden_size, config.vocab_size, config.num_attention_heads,
            config.num_key_value_heads, head_dim)


def _compile_all(dims, warm):
    """Every variant a training step launches: warm(kernel, *args, **kwargs) gets the
    arguments of its JITFunction.warmup."""
    from . import integrate, lce, nf4, norm, rope, swiglu

    hidden, vocab, nq, nkv, head_dim = dims
    f16, f32, i32, i64, u8 = torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8
    block_h = triton.next_power_of_2(hidden)
    heads = triton.next_power_of_2(nq)
    # forward, in the order a step first needs them
    warm(integrate.embed_fp32_kernel, i64, f16, f32, hidden, BLOCK_H=block_h, num_warps=4,
         grid=GRID)
    warm(norm.add_rmsnorm_fwd_kernel,
         f32, f16, f32, f32, f16, f32, f32, torch.bool, torch.bool, i64, u8, f16, f16, hidden,
         1e-6, 1.05, 1.05, 1, 1, 0, 1, BLOCK_H=block_h, num_warps=8, grid=GRID)
    slot = (u8, u8, f32, f32, f32, f16, 0.5, 1024)
    warm(nf4.nf4_dequant_kernel, *slot, *slot, *slot, 16, 16, BLOCK=nf4.BLOCK_BYTES, num_warps=4,
         enable_fp_fusion=False, grid=GRID)
    warm(rope.qkv_rope_fwd_kernel, f16, f16, f16, f32, f32, i64, f16, f16, f16, 64, 0, NQ=nq,
         NKV=nkv, D=head_dim, HEADS=heads, HAS_LQ=True, HAS_LV=True, num_warps=4, grid=GRID)
    warm(swiglu.example_swiglu_fwd_kernel, f16, f16, f16, 8960 * 16, BLOCK=swiglu.BLOCK,
         grid=GRID)
    warm(lce.lce_row_kernel, f16, vocab, i64, f32, f32, i32, vocab, lce.GRAD_SCALE, 1,
         BLOCK_V=lce.BLOCK_V, num_warps=8, grid=GRID)
    # backward
    warm(rope.qkv_rope_bwd_kernel, f16, f16, f16, f32, f32, i64, f16, 64, 0, nq * head_dim,
         head_dim, nq * head_dim, head_dim, nq * head_dim, head_dim, NQ=nq, NKV=nkv, D=head_dim,
         HEADS=heads, KV_HEADS=triton.next_power_of_2(nkv), num_warps=4, grid=GRID)
    warm(swiglu.example_swiglu_bwd_kernel, f16, f16, f16, f16, f16, 8960 * 16,
         BLOCK=swiglu.BLOCK, grid=GRID)
    warm(norm.add_rmsnorm_bwd_kernel,
         f32, f16, f32, f32, f16, f32, f32, f16, f16, u8, f16, f16, f32, f16, hidden, 1.05, 1.05,
         1, 1, 0, 1, 1, 1, 1, BLOCK_H=block_h, num_warps=8, grid=GRID)


def _build_launcher(kernel):
    """Build the C launcher of a compiled kernel into the Triton cache.

    Triton creates a kernel's launcher, a small C extension compiled with the system's C compiler
    and cached by its source, only at the kernel's first launch: one compiler run per kernel
    signature, inside the first training step. Building it here leaves that launch only the
    cached module to load (the launch still does that itself: this thread never touches the
    compiled kernel the launch uses)."""
    from triton.runtime import driver

    if kernel is not None and hasattr(kernel, "src"):
        driver.active.launcher_cls(kernel.src, kernel.metadata)


def _run(dims):
    try:
        # some threads compile while others build each compiled kernel's launcher: all of them
        # mostly wait on the compilers' subprocesses (ptxas, cc)
        with ThreadPoolExecutor(LAUNCHER_THREADS, thread_name_prefix="soupk-triton-launcher") \
                as launchers, \
                ThreadPoolExecutor(COMPILE_THREADS, thread_name_prefix="soupk-triton-compile") \
                as compilers:
            def compile_variant(kernel, *args, **kwargs):
                launchers.submit(_build_launcher, kernel.warmup(*args, **kwargs))

            _compile_all(dims, lambda kernel, *args, **kwargs: compilers.submit(
                compile_variant, kernel, *args, **kwargs))
    except Exception:  # noqa: BLE001 - a warm-up failure only means compiling on first launch
        pass


def start():
    """Start compiling in the background, once (no-op outside `soup train` or without CUDA).

    The model's dimensions are read here, in the calling thread, so the warm-up threads import
    nothing but Triton's compiler while the main thread goes on."""
    global _started
    if _started or not torch.cuda.is_available() or "--config" not in sys.argv:
        return None
    _started = True
    try:
        dims = _model_dims()
    except Exception:  # noqa: BLE001 - no warm-up: the kernels compile on first launch
        return None
    if dims is None:
        return None
    thread = threading.Thread(target=_run, args=(dims,), name="soupk-triton-warmup", daemon=True)
    thread.start()
    return thread
