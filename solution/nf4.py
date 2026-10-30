"""Frozen NF4 linear layers: our dequantization kernel, then one cuBLAS fp16 GEMM.

bitsandbytes 0.50 on a T4 handles a training-sized 4-bit linear as dequantize + F.linear, then
casts the fp16 output back to the input dtype (fp32 here, since the norms and the residual stream
are fp32). Our version:
- dequantizes NF4 with double quantization in one Triton pass: each byte holds two 4-bit codes (high
  nibble first); a weight is code[nibble] * (code2[absmax_q] * absmax2 + offset), with one absmax
  per 64 weights and one absmax2 per 256 absmax values (QLoRA, Dettmers et al. 2023);
- stacks projections that share an input (q|k|v, gate|up) so they share the dequantize buffer, and
  q|k|v run as one GEMM with the bias in its epilogue;
- keeps the outputs in fp16: those values are fp16 GEMM results either way, the base run only
  stored them as fp32 copies. Nothing is saved for backward but the packed weights, which backward
  dequantizes again.
"""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 1024


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, absmax2_ptr, code2_ptr, code_ptr, out_ptr, offset,
                       n_bytes, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    byte = pid * BLOCK + tl.arange(0, BLOCK)
    inside = byte < n_bytes
    packed = tl.load(packed_ptr + byte, mask=inside, other=0).to(tl.int32)
    block = byte // 32  # 64 weights per absmax, two weights per byte
    level = tl.load(absmax_ptr + block, mask=inside, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + level) * tl.load(absmax2_ptr + block // 256, mask=inside,
                                                 other=0.0) + offset
    high = tl.load(code_ptr + (packed >> 4)) * scale
    low = tl.load(code_ptr + (packed & 15)) * scale
    pairs = tl.reshape(tl.join(high, low), [2 * BLOCK])
    pos = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + pos, pairs.to(tl.float16), mask=pos < 2 * n_bytes)


class NF4Weight:
    """What the kernel needs from one bitsandbytes Params4bit (nf4, double quantization)."""

    def __init__(self, param):
        state = param.quant_state
        self.packed = param.data.reshape(-1)
        self.absmax = state.absmax
        self.absmax2 = state.state2.absmax
        self.code2 = state.state2.code
        self.code = state.code
        self.offset = float(state.offset)
        self.rows, self.cols = state.shape
        self.n_bytes = self.packed.numel()

    def dequantize_into(self, out):
        """Write this weight as fp16 [rows, cols] into `out` (contiguous, same shape)."""
        grid = (triton.cdiv(self.n_bytes, BLOCK_BYTES),)
        nf4_dequant_kernel[grid](self.packed, self.absmax, self.absmax2, self.code2, self.code, out,
                                 self.offset, self.n_bytes, BLOCK=BLOCK_BYTES, num_warps=4,
                                 enable_fp_fusion=False)  # bitsandbytes rounds mul, then add
        return out


def nf4_supported(module):
    """A bitsandbytes Linear4bit with the layout nf4_dequant_kernel reads."""
    weight = getattr(module, "weight", None)
    state = getattr(weight, "quant_state", None)
    if state is None or weight.device.type != "cuda":
        return False
    return (state.quant_type == "nf4" and state.blocksize == 64 and state.nested
            and state.state2.blocksize == 256 and len(state.shape) == 2
            and weight.dtype == torch.uint8 and weight.data.is_contiguous()
            and weight.data.numel() * 2 == state.shape[0] * state.shape[1])


def dequantize_stack(weights):
    """The weights stacked by rows, fp16 [sum(rows), cols]."""
    out = torch.empty((sum(w.rows for w in weights), weights[0].cols), dtype=torch.float16,
                      device=weights[0].packed.device)
    row = 0
    for w in weights:
        w.dequantize_into(out[row:row + w.rows])
        row += w.rows
    return out


class NF4Linear(torch.autograd.Function):
    """y_i = x @ W_i.T (+ bias for the single-output case) for frozen NF4 weights W_i sharing x.

    One output: one GEMM. Several outputs (gate|up): one dequantize buffer, a GEMM per output
    into its own contiguous tensor, and backward accumulates the input gradient over them."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, x, weights, bias):
        flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
        dense = dequantize_stack(weights)
        ctx.weights, ctx.in_shape, ctx.in_dtype = weights, x.shape, x.dtype
        if len(weights) == 1 or bias is not None:
            out = torch.addmm(bias, flat, dense.t()) if bias is not None else flat @ dense.t()
            return out.view(*x.shape[:-1], out.shape[-1])
        outs, row = [], 0
        for w in weights:
            outs.append((flat @ dense[row:row + w.rows].t()).view(*x.shape[:-1], w.rows))
            row += w.rows
        return tuple(outs)

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, *grads):
        dense = dequantize_stack(ctx.weights)
        if len(grads) == 1:
            grad = grads[0].reshape(-1, dense.shape[0]).to(torch.float16)
            grad_x = grad @ dense
        else:
            grad_x, row = None, 0
            for w, g in zip(ctx.weights, grads):
                part = dense[row:row + w.rows]
                row += w.rows
                if g is None:
                    continue
                g = g.reshape(-1, w.rows).to(torch.float16)
                grad_x = g @ part if grad_x is None else torch.addmm(grad_x, g, part)
        return grad_x.view(ctx.in_shape).to(ctx.in_dtype), None, None


def nf4_linear(x, weights, bias=None):
    return NF4Linear.apply(x, tuple(weights), bias)
