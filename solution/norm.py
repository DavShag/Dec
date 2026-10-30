"""RMSNorm with the residual add fused in, fp32 statistics, fp16 output for the next GEMM.

The base run's Qwen2RMSNorm on this model (fp32 residual stream, fp32 weight) is pow, mean, add,
rsqrt, mul, mul in fp32, after which every 4-bit linear casts the fp32 result to fp16 again. One
row per program here: the residual add (when given), the statistics and both multiplies stay in
fp32 registers and the row is written once as fp16, plus fp32 only where LoRA dropout reads it.
Backward recomputes x_hat from the saved fp32 input and the per-row 1/rms (Liger Kernel's RMSNorm
derivation, Hsu et al. 2024): dx = rstd * (dy*w - x_hat * mean(dy*w * x_hat)).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_fwd_kernel(x_ptr, delta_ptr, w_ptr, sum_ptr, y16_ptr, y32_ptr, rstd_ptr, H, eps,
                           HAS_DELTA: tl.constexpr, WRITE32: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    inside = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    if HAS_DELTA:
        x += tl.load(delta_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
        tl.store(sum_ptr + row * H + cols, x, mask=inside)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    y = tl.load(w_ptr + cols, mask=inside, other=0.0) * (x * rstd)
    tl.store(y16_ptr + row * H + cols, y.to(tl.float16), mask=inside)
    if WRITE32:
        tl.store(y32_ptr + row * H + cols, y, mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def add_rmsnorm_bwd_kernel(x_ptr, w_ptr, rstd_ptr, gy16_ptr, gy32_ptr, gsum_ptr, dx_ptr,
                           ddelta_ptr, H, HAS_GY16: tl.constexpr, HAS_GY32: tl.constexpr,
                           HAS_GSUM: tl.constexpr, WRITE_DDELTA: tl.constexpr,
                           BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    inside = cols < H
    x_hat = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0) * tl.load(rstd_ptr + row)
    grad = tl.zeros([BLOCK_H], dtype=tl.float32)
    if HAS_GY16:
        grad += tl.load(gy16_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    if HAS_GY32:
        grad += tl.load(gy32_ptr + row * H + cols, mask=inside, other=0.0)
    grad = grad * tl.load(w_ptr + cols, mask=inside, other=0.0)
    dx = (grad - x_hat * (tl.sum(grad * x_hat, axis=0) / H)) * tl.load(rstd_ptr + row)
    if HAS_GSUM:
        dx += tl.load(gsum_ptr + row * H + cols, mask=inside, other=0.0)
    tl.store(dx_ptr + row * H + cols, dx, mask=inside)
    if WRITE_DDELTA:
        tl.store(ddelta_ptr + row * H + cols, dx.to(tl.float16), mask=inside)


class AddRMSNorm(torch.autograd.Function):
    """(x + delta, y16, y32) with y = weight * rmsnorm(x + delta); delta may be None (then the sum
    output is None and x itself is normalized). The weight is frozen."""

    @staticmethod
    def forward(ctx, x, delta, weight, eps, want32):
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        x = x.contiguous()
        total = torch.empty_like(x, dtype=torch.float32) if delta is not None else None
        y16 = torch.empty(x.shape, dtype=torch.float16, device=x.device)
        y32 = torch.empty(x.shape, dtype=torch.float32, device=x.device) if want32 else None
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        delta_c = delta.contiguous() if delta is not None else x
        add_rmsnorm_fwd_kernel[(rows,)](x, delta_c, weight, total if total is not None else x, y16,
                                        y32 if want32 else y16, rstd, hidden, eps,
                                        HAS_DELTA=delta is not None, WRITE32=want32,
                                        BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
        ctx.save_for_backward(total if total is not None else x, weight, rstd)
        ctx.has_delta, ctx.delta_dtype = delta is not None, getattr(delta, "dtype", None)
        ctx.x_dtype = x.dtype
        return total, y16, y32

    @staticmethod
    def backward(ctx, g_sum, g_y16, g_y32):
        saved, weight, rstd = ctx.saved_tensors
        hidden = saved.shape[-1]
        rows = saved.numel() // hidden
        dx = torch.empty(saved.shape, dtype=torch.float32, device=saved.device)
        write_dd = ctx.has_delta and ctx.delta_dtype == torch.float16
        ddelta = torch.empty(saved.shape, dtype=torch.float16, device=saved.device) if write_dd else dx
        g16 = g_y16.contiguous() if g_y16 is not None else dx
        g32 = g_y32.contiguous() if g_y32 is not None else dx
        gs = g_sum.contiguous() if g_sum is not None else dx
        add_rmsnorm_bwd_kernel[(rows,)](saved, weight, rstd, g16, g32, gs, dx, ddelta, hidden,
                                        HAS_GY16=g_y16 is not None, HAS_GY32=g_y32 is not None,
                                        HAS_GSUM=g_sum is not None, WRITE_DDELTA=write_dd,
                                        BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
        grad_x = dx if ctx.x_dtype == torch.float32 else dx.to(ctx.x_dtype)
        if not ctx.has_delta:
            return grad_x, None, None, None, None
        grad_delta = ddelta if write_dd else dx.to(ctx.delta_dtype)
        return grad_x, grad_delta, None, None, None


def add_rmsnorm(x, delta, weight, eps, want32=False):
    return AddRMSNorm.apply(x, delta, weight, eps, want32)
