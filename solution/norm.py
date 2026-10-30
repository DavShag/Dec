"""RMSNorm with the residual add fused in, fp32 statistics, fp16 output for the next GEMM, and the
LoRA dropout of the q/v adapters applied in the same pass.

The base run's Qwen2RMSNorm on this model (fp32 residual stream, fp32 weight) is pow, mean, add,
rsqrt, mul, mul in fp32, after which every 4-bit linear casts the fp32 result to fp16 again and
PEFT runs dropout, a mask multiply and another cast for each LoRA adapter. One row per program here:
the residual add (when given), the statistics and both multiplies stay in fp32 registers; the row is
written once as fp16 for the GEMM and, for the adapters, once more per adapter as
fp16(y * keep * 1/(1-p)), the exact value the dropout kernel and the autocast cast produce. The keep
masks come from PyTorch's own dropout kernel, drawn in the base run's order (decoder.py).
Backward recomputes x_hat from the saved fp32 input and the per-row 1/rms (Liger Kernel's RMSNorm
derivation, Hsu et al. 2024): dx = rstd * (dy*w - x_hat * mean(dy*w * x_hat)), with dy the sum of
the GEMM's gradient and both adapters' masked gradients.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_fwd_kernel(x_ptr, delta_ptr, w_ptr, sum_ptr, y16_ptr, y32_ptr, rstd_ptr, kq_ptr,
                           kv_ptr, pidx_ptr, xq_ptr, xv_ptr, H, eps, scale_q, scale_v,
                           HAS_DELTA: tl.constexpr, WRITE32: tl.constexpr, LORA: tl.constexpr,
                           BLOCK_H: tl.constexpr):
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
    if LORA:  # keep masks are [B*T, H] over the padded batch; pidx maps this row into it
        keep_row = tl.load(pidx_ptr + row).to(tl.int64) * H + cols
        keep_q = tl.load(kq_ptr + keep_row, mask=inside, other=0)
        tl.store(xq_ptr + row * H + cols, tl.where(keep_q, y * scale_q, 0.0).to(tl.float16),
                 mask=inside)
        keep_v = tl.load(kv_ptr + keep_row, mask=inside, other=0)
        tl.store(xv_ptr + row * H + cols, tl.where(keep_v, y * scale_v, 0.0).to(tl.float16),
                 mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def add_rmsnorm_bwd_kernel(x_ptr, w_ptr, rstd_ptr, gy16_ptr, gy32_ptr, gsum_ptr, gq_ptr, gv_ptr,
                           kq_ptr, kv_ptr, pidx_ptr, dx_ptr, ddelta_ptr, H, scale_q, scale_v,
                           HAS_GY16: tl.constexpr, HAS_GY32: tl.constexpr,
                           HAS_GSUM: tl.constexpr, LORA: tl.constexpr,
                           WRITE_DDELTA: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    inside = cols < H
    x_hat = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0) * tl.load(rstd_ptr + row)
    grad = tl.zeros([BLOCK_H], dtype=tl.float32)
    if HAS_GY16:
        grad += tl.load(gy16_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    if HAS_GY32:
        grad += tl.load(gy32_ptr + row * H + cols, mask=inside, other=0.0)
    if LORA:
        keep_row = tl.load(pidx_ptr + row).to(tl.int64) * H + cols
        g_q = tl.load(gq_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
        grad += tl.where(tl.load(kq_ptr + keep_row, mask=inside, other=0), g_q * scale_q, 0.0)
        g_v = tl.load(gv_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
        grad += tl.where(tl.load(kv_ptr + keep_row, mask=inside, other=0), g_v * scale_v, 0.0)
    grad = grad * tl.load(w_ptr + cols, mask=inside, other=0.0)
    dx = (grad - x_hat * (tl.sum(grad * x_hat, axis=0) / H)) * tl.load(rstd_ptr + row)
    if HAS_GSUM:
        dx += tl.load(gsum_ptr + row * H + cols, mask=inside, other=0.0)
    tl.store(dx_ptr + row * H + cols, dx, mask=inside)
    if WRITE_DDELTA:
        tl.store(ddelta_ptr + row * H + cols, dx.to(tl.float16), mask=inside)


class AddRMSNorm(torch.autograd.Function):
    """(x + delta, y16, y32, xq16, xv16) with y = weight * rmsnorm(x + delta).

    delta may be None (then x itself is normalized and the sum output is None); y32 only with
    want32; xq16 / xv16 only with lora = (keep_q, keep_v, pidx, scale_q, scale_v), the dropped-out
    fp16 inputs of the q and v adapters. The weight is frozen."""

    @staticmethod
    def forward(ctx, x, delta, weight, eps, want32, lora):
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        x = x.contiguous()
        new = lambda dtype: torch.empty(x.shape, dtype=dtype, device=x.device)  # noqa: E731
        total = new(torch.float32) if delta is not None else None
        y16 = new(torch.float16)
        y32 = new(torch.float32) if want32 else None
        xq, xv = (new(torch.float16), new(torch.float16)) if lora is not None else (None, None)
        keep_q, keep_v, pidx, scale_q, scale_v = lora if lora is not None else (x, x, x, 1.0, 1.0)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        add_rmsnorm_fwd_kernel[(rows,)](
            x, delta.contiguous() if delta is not None else x, weight,
            total if total is not None else x, y16, y32 if want32 else y16, rstd, keep_q, keep_v,
            pidx, xq if xq is not None else y16, xv if xv is not None else y16, hidden, eps,
            scale_q, scale_v, HAS_DELTA=delta is not None, WRITE32=want32,
            LORA=lora is not None, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
        saved = [total if total is not None else x, weight, rstd]
        if lora is not None:
            saved += [keep_q, keep_v, pidx]
            ctx.scales = (scale_q, scale_v)
        ctx.save_for_backward(*saved)
        ctx.has_delta, ctx.delta_dtype = delta is not None, getattr(delta, "dtype", None)
        ctx.x_dtype, ctx.lora = x.dtype, lora is not None
        return total, y16, y32, xq, xv

    @staticmethod
    def backward(ctx, g_sum, g_y16, g_y32, g_xq, g_xv):
        saved, weight, rstd, *keeps = ctx.saved_tensors
        hidden = saved.shape[-1]
        rows = saved.numel() // hidden
        dx = torch.empty(saved.shape, dtype=torch.float32, device=saved.device)
        write_dd = ctx.has_delta and ctx.delta_dtype == torch.float16
        ddelta = torch.empty(saved.shape, dtype=torch.float16, device=saved.device) if write_dd else dx
        use_lora = ctx.lora and (g_xq is not None or g_xv is not None)
        if use_lora:
            zeros = None
            if g_xq is None or g_xv is None:
                zeros = torch.zeros(saved.shape, dtype=torch.float16, device=saved.device)
            g_xq = g_xq.contiguous() if g_xq is not None else zeros
            g_xv = g_xv.contiguous() if g_xv is not None else zeros
            keep_q, keep_v, pidx = keeps
            scale_q, scale_v = ctx.scales
        else:
            g_xq = g_xv = keep_q = keep_v = pidx = dx
            scale_q = scale_v = 1.0
        add_rmsnorm_bwd_kernel[(rows,)](
            saved, weight, rstd, g_y16.contiguous() if g_y16 is not None else dx,
            g_y32.contiguous() if g_y32 is not None else dx,
            g_sum.contiguous() if g_sum is not None else dx, g_xq, g_xv, keep_q, keep_v, pidx, dx,
            ddelta, hidden, scale_q, scale_v, HAS_GY16=g_y16 is not None,
            HAS_GY32=g_y32 is not None, HAS_GSUM=g_sum is not None, LORA=use_lora,
            WRITE_DDELTA=write_dd, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
        grad_x = dx if ctx.x_dtype == torch.float32 else dx.to(ctx.x_dtype)
        if not ctx.has_delta:
            return grad_x, None, None, None, None, None
        grad_delta = ddelta if write_dd else dx.to(ctx.delta_dtype)
        return grad_x, grad_delta, None, None, None, None


def add_rmsnorm(x, delta, weight, eps, want32=False, lora=None):
    """See AddRMSNorm; returns (sum, y16, y32, xq16, xv16), None for what was not asked for."""
    return AddRMSNorm.apply(x, delta, weight, eps, want32, lora)
