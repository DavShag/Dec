"""RMSNorm with the residual add fused in, fp32 statistics, fp16 output for the next GEMM, and the
LoRA dropout of the q/v adapters applied in the same pass.

The base run's Qwen2RMSNorm on this model (fp32 residual stream, fp32 weight) is pow, mean, add,
rsqrt, mul, mul in fp32, after which every 4-bit linear casts the fp32 result to fp16 again and
PEFT runs dropout, a mask multiply and another cast for each LoRA adapter. One row per program here:
the residual add (when given), the statistics and both multiplies stay in fp32 registers; the row is
written once as fp16 for the GEMM and, for the adapters, once more per adapter as
fp16(y * keep * 1/(1-p)), the exact value the dropout kernel and the autocast cast produce. The keep
masks come from PyTorch's own dropout kernel, drawn in the base run's order (decoder.py), as bool
tensors over the whole padded batch; the kernel also writes this row's two masks packed 8 per byte,
and only those are kept for backward (H/4 bytes per token instead of 2H over every padded position).
Backward recomputes x_hat from the saved fp32 input and the per-row 1/rms (Liger Kernel's RMSNorm
derivation, Hsu et al. 2024): dx = rstd * (dy*w - x_hat * mean(dy*w * x_hat)), with dy the sum of
the GEMM's gradient and both adapters' masked gradients.
"""

import torch
import triton
import triton.language as tl


# The flags are runtime arguments (not constexpr, not specialized) so that every call compiles to
# the same kernel: compilation happens inside the timed run, once per variant.
@triton.jit(do_not_specialize=["has_delta", "write_sum", "write32", "lora"])
def add_rmsnorm_fwd_kernel(x_ptr, delta_ptr, w_ptr, sum_ptr, y16_ptr, y32_ptr, rstd_ptr, kq_ptr,
                           kv_ptr, pidx_ptr, bits_ptr, xq_ptr, xv_ptr, H, eps, scale_q, scale_v,
                           has_delta, write_sum, write32, lora, BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    inside = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    if has_delta:
        x += tl.load(delta_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    if write_sum:
        tl.store(sum_ptr + row * H + cols, x, mask=inside)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    y = tl.load(w_ptr + cols, mask=inside, other=0.0) * (x * rstd)
    tl.store(y16_ptr + row * H + cols, y.to(tl.float16), mask=inside)
    if write32:
        tl.store(y32_ptr + row * H + cols, y, mask=inside)
    if lora:  # keep masks are [B*T, H] over the padded batch; pidx maps this row into it
        keep_row = tl.load(pidx_ptr + row).to(tl.int64) * H + cols
        keep_q = tl.load(kq_ptr + keep_row, mask=inside, other=0)
        tl.store(xq_ptr + row * H + cols, tl.where(keep_q, y * scale_q, 0.0).to(tl.float16),
                 mask=inside)
        keep_v = tl.load(kv_ptr + keep_row, mask=inside, other=0)
        tl.store(xv_ptr + row * H + cols, tl.where(keep_v, y * scale_v, 0.0).to(tl.float16),
                 mask=inside)
        # both masks of this row for backward, 8 per byte (bit c % 8 of byte c // 8): q, then v
        bit = 1 << (cols % 8)
        packed_q = tl.sum(tl.reshape(keep_q.to(tl.int32) * bit, [BLOCK_H // 8, 8]), axis=1)
        packed_v = tl.sum(tl.reshape(keep_v.to(tl.int32) * bit, [BLOCK_H // 8, 8]), axis=1)
        nbytes = H // 8
        byte = tl.arange(0, BLOCK_H // 8)
        bits_row = bits_ptr + row * (2 * nbytes)
        tl.store(bits_row + byte, packed_q.to(tl.uint8), mask=byte < nbytes)
        tl.store(bits_row + nbytes + byte, packed_v.to(tl.uint8), mask=byte < nbytes)
    tl.store(rstd_ptr + row, rstd)


@triton.jit(do_not_specialize=["add_delta", "has_gy16", "has_gy32", "has_gsum", "lora", "regen",
                               "write_ddelta"])
def add_rmsnorm_bwd_kernel(x_ptr, delta_ptr, w_ptr, rstd_ptr, gy16_ptr, gy32_ptr, gsum_ptr, gq_ptr,
                           gv_ptr, bits_ptr, xq_ptr, xv_ptr, dx_ptr, ddelta_ptr, H, scale_q,
                           scale_v, add_delta, has_gy16, has_gy32, has_gsum, lora, regen,
                           write_ddelta, BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    inside = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0)
    if add_delta:  # the forward's sum, the same fp32 add of the same values
        x += tl.load(delta_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    x_hat = x * tl.load(rstd_ptr + row)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0)
    grad = tl.zeros([BLOCK_H], dtype=tl.float32)
    if has_gy16:
        grad += tl.load(gy16_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    if has_gy32:
        grad += tl.load(gy32_ptr + row * H + cols, mask=inside, other=0.0)
    if lora:  # the row's keep masks, as the forward kernel packed them
        nbytes = H // 8
        bits_row = bits_ptr + row * (2 * nbytes)
        shift = cols % 8
        keep_q = (tl.load(bits_row + cols // 8, mask=inside, other=0).to(tl.int32) >> shift) & 1
        keep_v = (tl.load(bits_row + nbytes + cols // 8, mask=inside, other=0).to(tl.int32)
                  >> shift) & 1
        g_q = tl.load(gq_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
        grad += tl.where(keep_q != 0, g_q * scale_q, 0.0)
        g_v = tl.load(gv_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
        grad += tl.where(keep_v != 0, g_v * scale_v, 0.0)
        if regen:  # the adapters' fp16 inputs again, the forward's values: y = w * (x * rstd)
            y = w * x_hat
            tl.store(xq_ptr + row * H + cols, tl.where(keep_q != 0, y * scale_q, 0.0).to(tl.float16),
                     mask=inside)
            tl.store(xv_ptr + row * H + cols, tl.where(keep_v != 0, y * scale_v, 0.0).to(tl.float16),
                     mask=inside)
    grad = grad * w
    dx = (grad - x_hat * (tl.sum(grad * x_hat, axis=0) / H)) * tl.load(rstd_ptr + row)
    if has_gsum:
        dx += tl.load(gsum_ptr + row * H + cols, mask=inside, other=0.0)
    tl.store(dx_ptr + row * H + cols, dx, mask=inside)
    if write_ddelta:
        tl.store(ddelta_ptr + row * H + cols, dx.to(tl.float16), mask=inside)


_PLACEHOLDERS = {}


def _placeholder(dtype, device):
    """A one-element tensor standing in for an unused pointer argument of a given dtype."""
    key = (dtype, str(device))
    if key not in _PLACEHOLDERS:
        _PLACEHOLDERS[key] = torch.zeros(1, dtype=dtype, device=device)
    return _PLACEHOLDERS[key]


class AddRMSNorm(torch.autograd.Function):
    """(x + delta, y16, y32, xq16, xv16) with y = weight * rmsnorm(x + delta).

    delta may be None (then x itself is normalized and the sum output is None); y32 only with
    want32; xq16 / xv16 only with lora = (keep_q, keep_v, pidx, scale_q, scale_v), the dropped-out
    fp16 inputs of the q and v adapters (keep masks [B*T, H] bool over the padded batch, pidx the
    padded position of each row; H a multiple of 8). The weight is frozen.

    With the adapters' A weights (a_q, a_v, [r, H]) the last two outputs are instead xq @ a_q.T and
    xv @ a_v.T, the GEMMs PEFT's lora_A runs under fp16 autocast, and xq / xv are not kept: backward
    writes them again from the saved input, 1/rms and packed masks (the same values) for the
    gradients of a_q and a_v. That is 2H fp16 values per token a layer does not keep.

    keep_sum=False (with an fp16 delta, for an x that backward keeps anyway: the layer's residual
    input, which the first norm saves): x and delta are kept instead of their fp32 sum, which
    backward forms again with the same fp32 add. 2H bytes per token less, nothing recomputed but
    an add."""

    @staticmethod
    def forward(ctx, x, delta, weight, eps, want32, lora, a_q=None, a_v=None, keep_sum=True):
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        x = x.contiguous()
        delta = delta.contiguous() if delta is not None else None
        new = lambda dtype: torch.empty(x.shape, dtype=dtype, device=x.device)  # noqa: E731
        total = new(torch.float32) if delta is not None else None
        y16 = new(torch.float16)
        y32 = new(torch.float32) if want32 else None
        xq, xv = (new(torch.float16), new(torch.float16)) if lora is not None else (None, None)
        hold = lambda dtype: _placeholder(dtype, x.device)  # noqa: E731
        keep_q, keep_v, pidx, scale_q, scale_v = lora if lora is not None else (
            hold(torch.bool), hold(torch.bool), hold(torch.int64), 1.0, 1.0)
        bits = (torch.empty((rows, 2 * (hidden // 8)), dtype=torch.uint8, device=x.device)
                if lora is not None else hold(torch.uint8))
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        add_rmsnorm_fwd_kernel[(rows,)](
            x, delta if delta is not None else hold(torch.float16), weight,
            total if total is not None else hold(torch.float32), y16,
            y32 if want32 else hold(torch.float32), rstd, keep_q, keep_v, pidx, bits,
            xq if xq is not None else hold(torch.float16),
            xv if xv is not None else hold(torch.float16), hidden, eps, scale_q, scale_v,
            int(delta is not None), int(delta is not None), int(want32), int(lora is not None),
            BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
        ctx.sum_parts = delta is not None and not keep_sum and delta.dtype == torch.float16
        saved = [total if total is not None and not ctx.sum_parts else x, weight, rstd]
        ctx.adapters = lora is not None and a_q is not None
        if lora is not None:
            saved.append(bits)  # the padded masks themselves are not kept
            ctx.scales = (scale_q, scale_v)
        if ctx.adapters:
            saved += [a_q, a_v]
            # F.linear(x, a) under autocast: x @ fp16(a).T, one fp16 GEMM each
            xq = torch.mm(xq.view(rows, hidden), a_q.to(torch.float16).t()).view(*x.shape[:-1], -1)
            xv = torch.mm(xv.view(rows, hidden), a_v.to(torch.float16).t()).view(*x.shape[:-1], -1)
        if ctx.sum_parts:
            saved.append(delta)
        ctx.save_for_backward(*saved)
        ctx.has_delta, ctx.delta_dtype = delta is not None, getattr(delta, "dtype", None)
        ctx.x_dtype, ctx.lora = x.dtype, lora is not None
        return total, y16, y32, xq, xv

    @staticmethod
    def backward(ctx, g_sum, g_y16, g_y32, g_xq, g_xv):
        saved, weight, rstd, *rest = ctx.saved_tensors
        delta = rest.pop() if ctx.sum_parts else None
        hidden = saved.shape[-1]
        rows = saved.numel() // hidden
        dx = torch.empty(saved.shape, dtype=torch.float32, device=saved.device)
        write_dd = ctx.has_delta and ctx.delta_dtype == torch.float16
        ddelta = torch.empty(saved.shape, dtype=torch.float16, device=saved.device) if write_dd else None
        hold = lambda dtype: _placeholder(dtype, saved.device)  # noqa: E731
        use_lora = ctx.lora and (g_xq is not None or g_xv is not None)
        g_zq = g_zv = None
        regen = use_lora and ctx.adapters
        if regen:  # gradients of the adapters' fp16 inputs: as autograd forms them for x @ a.T
            a_q, a_v = rest[1], rest[2]
            g_zq, g_zv = g_xq, g_xv
            g_xq = torch.mm(g_zq.reshape(rows, -1), a_q.to(torch.float16)) if g_zq is not None else None
            g_xv = torch.mm(g_zv.reshape(rows, -1), a_v.to(torch.float16)) if g_zv is not None else None
        if use_lora:
            zeros = None
            if g_xq is None or g_xv is None:
                zeros = torch.zeros(saved.shape, dtype=torch.float16, device=saved.device)
            g_xq = g_xq.contiguous() if g_xq is not None else zeros
            g_xv = g_xv.contiguous() if g_xv is not None else zeros
            bits = rest[0]
            scale_q, scale_v = ctx.scales
        else:
            g_xq = g_xv = hold(torch.float16)
            bits, scale_q, scale_v = hold(torch.uint8), 1.0, 1.0
        xq = torch.empty(saved.shape, dtype=torch.float16, device=saved.device) if regen else None
        xv = torch.empty(saved.shape, dtype=torch.float16, device=saved.device) if regen else None
        add_rmsnorm_bwd_kernel[(rows,)](
            saved, delta if delta is not None else hold(torch.float16), weight, rstd,
            g_y16.to(torch.float16).contiguous() if g_y16 is not None else hold(torch.float16),
            g_y32.contiguous() if g_y32 is not None else hold(torch.float32),
            g_sum.contiguous() if g_sum is not None else hold(torch.float32), g_xq, g_xv, bits,
            xq if regen else hold(torch.float16), xv if regen else hold(torch.float16), dx,
            ddelta if write_dd else hold(torch.float16), hidden, scale_q, scale_v,
            int(delta is not None), int(g_y16 is not None), int(g_y32 is not None),
            int(g_sum is not None),
            int(use_lora), int(regen), int(write_dd), BLOCK_H=triton.next_power_of_2(hidden),
            num_warps=8)
        grad_a_q = grad_a_v = None
        if regen:  # autograd's weight gradient of x @ a.T (column-major mat2): g.T @ x, to a's dtype
            if g_zq is not None and ctx.needs_input_grad[6]:
                grad_a_q = torch.mm(g_zq.reshape(rows, -1).t(), xq.view(rows, hidden)).to(a_q.dtype)
            if g_zv is not None and ctx.needs_input_grad[7]:
                grad_a_v = torch.mm(g_zv.reshape(rows, -1).t(), xv.view(rows, hidden)).to(a_v.dtype)
        grad_x = dx if ctx.x_dtype == torch.float32 else dx.to(ctx.x_dtype)
        grad_delta = None
        if ctx.has_delta:
            grad_delta = ddelta if write_dd else dx.to(ctx.delta_dtype)
        return grad_x, grad_delta, None, None, None, None, grad_a_q, grad_a_v, None


def add_rmsnorm(x, delta, weight, eps, want32=False, lora=None, adapters=None, keep_sum=True):
    """See AddRMSNorm; returns (sum, y16, y32, xq16, xv16), None for what was not asked for, or
    with adapters = (a_q, a_v) the adapters' A outputs in place of xq16 / xv16."""
    a_q, a_v = adapters if adapters is not None else (None, None)
    return AddRMSNorm.apply(x, delta, weight, eps, want32, lora, a_q, a_v, keep_sum)


def rmsnorm16(x, weight, eps, delta=None):
    """fp16(weight * rmsnorm(x + delta)) without autograd: the y16 that AddRMSNorm computed from the
    same fp32 sum (the same kernel and values), for a backward that rebuilds it."""
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    hold = lambda dtype: _placeholder(dtype, x.device)  # noqa: E731
    y16 = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    add_rmsnorm_fwd_kernel[(rows,)](
        x, delta if delta is not None else hold(torch.float16), weight, hold(torch.float32), y16,
        hold(torch.float32), rstd, hold(torch.bool), hold(torch.bool), hold(torch.int64),
        hold(torch.uint8), hold(torch.float16), hold(torch.float16), hidden, eps, 1.0, 1.0,
        int(delta is not None), 0, 0, 0, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
    return y16
