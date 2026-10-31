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


@triton.jit(do_not_specialize=["blocks0", "blocks1"])  # n_i stay specialized (multiples
# of 16): the masks then vectorize; every real weight has the same divisibility, so one variant
def nf4_dequant_kernel(p0, a0, b0, c0, k0, o0, off0, n0, p1, a1, b1, c1, k1, o1, off1, n1,
                       p2, a2, b2, c2, k2, o2, off2, n2, blocks0, blocks1, BLOCK: tl.constexpr):
    """Up to three weights (q|k|v, gate|up, or one) in one launch: the first blocks0 programs
    dequantize weight 0, the next blocks1 weight 1, the rest weight 2; each program turns BLOCK
    packed bytes into 2 * BLOCK fp16 weights. One launch per group keeps the per-launch CPU cost
    (Python launcher, harness timing) off the critical path."""
    pid = tl.program_id(0)
    first, second = pid < blocks0, pid < blocks0 + blocks1
    packed_ptr = tl.where(first, p0, tl.where(second, p1, p2))
    absmax_ptr = tl.where(first, a0, tl.where(second, a1, a2))
    absmax2_ptr = tl.where(first, b0, tl.where(second, b1, b2))
    code2_ptr = tl.where(first, c0, tl.where(second, c1, c2))
    code_ptr = tl.where(first, k0, tl.where(second, k1, k2))
    out_ptr = tl.where(first, o0, tl.where(second, o1, o2))
    offset = tl.where(first, off0, tl.where(second, off1, off2))
    n_bytes = tl.where(first, n0, tl.where(second, n1, n2))
    blk = tl.where(first, pid, tl.where(second, pid - blocks0, pid - blocks0 - blocks1))

    byte = blk * BLOCK + tl.arange(0, BLOCK)
    inside = byte < n_bytes
    packed = tl.load(packed_ptr + byte, mask=inside, other=0).to(tl.int32)
    block = byte // 32  # 64 weights per absmax, two weights per byte
    level = tl.load(absmax_ptr + block, mask=inside, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + level) * tl.load(absmax2_ptr + block // 256, mask=inside,
                                                 other=0.0) + offset
    high = tl.load(code_ptr + (packed >> 4)) * scale
    low = tl.load(code_ptr + (packed & 15)) * scale
    pairs = tl.reshape(tl.join(high, low), [2 * BLOCK])
    pos = blk * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
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

    def args(self, out):
        return (self.packed, self.absmax, self.absmax2, self.code2, self.code, out, self.offset,
                self.n_bytes)


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


_SCRATCH = {}


def scratch(numel, device):
    """fp16 [numel], a view of one buffer per device for transients of which only one is alive at
    a time: the dense weights of a dequantize launch, which the GEMMs right after it read, and the
    fused head's chunk of logits (lce.py). Their uses follow one another in one CUDA stream, so
    reusing the memory is safe, and the allocator no longer cuts holes of their sizes among the
    activations (on the T4 it reserved ~150 MiB more than it had allocated at the peak).
    Allocated at its largest size when the model is prepared (integrate.prepare_model), before
    any activation."""
    buffer = _SCRATCH.get(device)
    if buffer is None or buffer.numel() < numel:
        _SCRATCH[device] = buffer = torch.empty(numel, dtype=torch.float16, device=device)
    return buffer[:numel]


def dequantize_stack(weights, into_scratch=False):
    """The weights stacked by rows, fp16 [sum(rows), cols]: one launch per three weights. With
    into_scratch the result is a view of the shared scratch buffer, valid until its next use."""
    rows, cols, device = sum(w.rows for w in weights), weights[0].cols, weights[0].packed.device
    out = (scratch(rows * cols, device).view(rows, cols) if into_scratch else
           torch.empty((rows, cols), dtype=torch.float16, device=device))
    row = 0
    for first in range(0, len(weights), 3):
        group, slots = weights[first:first + 3], []
        for w in group:
            slots.append(w.args(out[row:row + w.rows]))
            row += w.rows
        blocks = [triton.cdiv(slot[-1], BLOCK_BYTES) for slot in slots]
        while len(slots) < 3:  # unused slots: never reached, they get no programs
            slots.append(slots[0][:-1] + (0,))
            blocks.append(0)
        nf4_dequant_kernel[(sum(blocks),)](*slots[0], *slots[1], *slots[2], blocks[0], blocks[1],
                                           BLOCK=BLOCK_BYTES, num_warps=4,
                                           enable_fp_fusion=False)  # bnb rounds mul, then add
    return out


class NF4Linear(torch.autograd.Function):
    """y_i = x @ W_i.T (+ bias for the single-output case) for frozen NF4 weights W_i sharing x.

    One output: one GEMM. Several outputs (gate|up): one dequantize launch into one buffer, then a
    GEMM per output, on its rows of the buffer, into its own contiguous tensor; backward
    accumulates the input gradient over them. One launch instead of one per weight costs a second
    27.5 MB at once and saves a launch per layer each way, and every launch of our kernels costs
    the CPU more than the GPU here (Triton's launcher plus the harness's timing events)."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, x, weights, bias):
        flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
        ctx.weights, ctx.in_shape, ctx.in_dtype = weights, x.shape, x.dtype
        dense = dequantize_stack(weights, into_scratch=True)  # read by the GEMMs below only
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
        dense = dequantize_stack(ctx.weights, into_scratch=True)
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
