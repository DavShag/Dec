"""Compile our kernels on a background thread while Soup loads the model.

The harness times the whole `soup train` command and starts every run with an empty Triton cache,
so each kernel variant would otherwise compile on its first launch inside the first training step
(several seconds on Kaggle's CPUs). `JITFunction.warmup` compiles without launching, from dtypes
alone, so this thread needs no GPU memory and runs the same way in every run. A variant the thread
has not reached yet simply compiles on first launch, as it would without it.

The arguments mirror the real launches (same dtypes, same integer divisibility, same constexprs and
launch options), or the compiled variant would not be the one training asks for.
"""

import sys
import threading

import torch
import triton

GRID = (1,)


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


def _compile_all(dims):
    from . import integrate, lce, nf4, norm, rope, swiglu

    hidden, vocab, nq, nkv, head_dim = dims
    f16, f32, i32, i64, u8 = torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8
    block_h = triton.next_power_of_2(hidden)
    heads = triton.next_power_of_2(nq)
    # forward, in the order a step first needs them
    integrate.embed_fp32_kernel.warmup(i64, f16, f32, hidden, BLOCK_H=block_h, num_warps=4,
                                       grid=GRID)
    for has_delta, write32 in ((False, True), (True, True), (True, False)):
        norm.add_rmsnorm_fwd_kernel.warmup(
            f32, f16 if has_delta else f32, f32, f32, f16, f32 if write32 else f16, f32, hidden,
            1e-6, HAS_DELTA=has_delta, WRITE32=write32, BLOCK_H=block_h, num_warps=8, grid=GRID)
    nf4.nf4_dequant_kernel.warmup(u8, u8, f32, f32, f32, f16, 0.5, 1024, BLOCK=nf4.BLOCK_BYTES,
                                  num_warps=4, enable_fp_fusion=False, grid=GRID)
    for unpad, rep in ((True, nq // nkv), (False, 1)):
        rope.qkv_rope_fwd_kernel.warmup(
            f16, f16, f16, f32, f32, f16, f16, f16, i32 if unpad else f16, 64, 0, 128, 128, 128,
            128, 128, 128, NQ=nq, NKV=nkv, REP=rep, D=head_dim, HEADS=heads, HAS_LQ=True,
            HAS_LV=True, UNPAD=unpad, num_warps=4, grid=GRID)
    swiglu.example_swiglu_fwd_kernel.warmup(f16, f16, f16, 8960 * 16, BLOCK=swiglu.BLOCK,
                                            grid=GRID)
    for with_grad in (True, False):
        lce.lce_row_kernel.warmup(f16, vocab, i64, f32, f32, i32, vocab, lce.GRAD_SCALE,
                                  WITH_GRAD=with_grad, BLOCK_V=lce.BLOCK_V, num_warps=8, grid=GRID)
    # backward
    for unpad, rep in ((True, nq // nkv), (False, 1)):
        rope.qkv_rope_bwd_kernel.warmup(
            f16, f16, f16, f32, f32, f16, i64 if unpad else f16, 64, 0, 128, 128, 128, 128, 128,
            128, 128, 128, 128, NQ=nq, NKV=nkv, REP=rep, D=head_dim, HEADS=heads,
            KV_HEADS=triton.next_power_of_2(nkv), UNPAD=unpad, num_warps=4, grid=GRID)
    swiglu.example_swiglu_bwd_kernel.warmup(f16, f16, f16, f16, f16, 8960 * 16,
                                            BLOCK=swiglu.BLOCK, grid=GRID)
    for gy32, gsum in ((True, True), (False, True), (False, False)):
        norm.add_rmsnorm_bwd_kernel.warmup(
            f32, f32, f32, f16, f32, f32, f32, f16, hidden, HAS_GY16=True, HAS_GY32=gy32,
            HAS_GSUM=gsum, WRITE_DDELTA=True, BLOCK_H=block_h, num_warps=8, grid=GRID)


def _run():
    try:
        dims = _model_dims()
        if dims is not None:
            _compile_all(dims)
    except Exception:  # noqa: BLE001 - a warm-up failure only means compiling on first launch
        pass


def start():
    """Start compiling in the background (no-op outside `soup train` or without CUDA)."""
    if not torch.cuda.is_available() or "--config" not in sys.argv:
        return None
    thread = threading.Thread(target=_run, name="soupk-triton-warmup", daemon=True)
    thread.start()
    return thread
