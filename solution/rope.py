"""The q/k/v epilogue of the attention block in one pass each way.

After the fused q|k|v GEMM the base run does, per layer: three fp16 -> fp32 output casts, a clone
and an add per LoRA projection, RoPE as q*cos + rotate_half(q)*sin (slices, a cat, a neg, two muls,
an add, for q and for k), repeat_kv's expanded copies, and three fp32 -> fp16 casts for SDPA. Here
one program per real token reads its q|k|v row and the LoRA outputs, adds them in fp32, rotates q
and k in fp32 at the token's position (rotate_half as a load of the other half with a sign), and
writes fp16 q, k, v token-major, [n, heads, head_dim], k and v already repeated to every query head
(SDPA's memory-efficient kernel has no grouped-query mode on sm75). Backward is the transpose:
inverse rotation, the sum over the repeated k/v heads, and one fp16 gradient row for the q|k|v
GEMM, whose q and v parts are also the LoRA outputs' gradients.
RoPE as in Su et al. 2021 (RoFormer), in the rotate_half form of transformers' Qwen2.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])  # T varies per batch: one compile
def qkv_rope_fwd_kernel(qkv_ptr, lq_ptr, lv_ptr, cos_ptr, sin_ptr, pos_ptr, q_ptr, k_ptr, v_ptr,
                        T, cos_sb, NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
                        HEADS: tl.constexpr, HAS_LQ: tl.constexpr, HAS_LV: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + n).to(tl.int64)  # the token's place in the padded [B, T] batch
    b, t = pos // T, pos % T
    d = tl.arange(0, D)[None, :]
    swap = (d + D // 2) % D
    sign = tl.where(d < D // 2, -1.0, 1.0)
    cos = tl.load(cos_ptr + b * cos_sb + t * D + d)
    sin = tl.load(sin_ptr + b * cos_sb + t * D + d)
    h = tl.arange(0, HEADS)[:, None]
    heads = h < NQ
    row = qkv_ptr + n * ((NQ + 2 * NKV) * D)
    out = n * (NQ * D) + h * D + d

    q = tl.load(row + h * D + d, mask=heads, other=0.0).to(tl.float32)
    q_sw = tl.load(row + h * D + swap, mask=heads, other=0.0).to(tl.float32)
    if HAS_LQ:
        lrow = lq_ptr + n * (NQ * D)
        q += tl.load(lrow + h * D + d, mask=heads, other=0.0).to(tl.float32)
        q_sw += tl.load(lrow + h * D + swap, mask=heads, other=0.0).to(tl.float32)
    tl.store(q_ptr + out, (q * cos + (sign * q_sw) * sin).to(tl.float16), mask=heads)

    src = h // (NQ // NKV)  # query head h reads kv head h // group
    k = tl.load(row + (NQ + src) * D + d, mask=heads, other=0.0).to(tl.float32)
    k_sw = tl.load(row + (NQ + src) * D + swap, mask=heads, other=0.0).to(tl.float32)
    tl.store(k_ptr + out, (k * cos + (sign * k_sw) * sin).to(tl.float16), mask=heads)
    v = tl.load(row + (NQ + NKV + src) * D + d, mask=heads, other=0.0).to(tl.float32)
    if HAS_LV:
        v += tl.load(lv_ptr + n * (NKV * D) + src * D + d, mask=heads, other=0.0).to(tl.float32)
    tl.store(v_ptr + out, v.to(tl.float16), mask=heads)


@triton.jit(do_not_specialize=["T"])
def qkv_rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, pos_ptr, dqkv_ptr, T, cos_sb,
                        dq_sr, dq_sh, dk_sr, dk_sh, dv_sr, dv_sh, NQ: tl.constexpr,
                        NKV: tl.constexpr, D: tl.constexpr, HEADS: tl.constexpr,
                        KV_HEADS: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + n).to(tl.int64)
    b, t = pos // T, pos % T
    d = tl.arange(0, D)[None, :]
    swap = (d + D // 2) % D
    sign = tl.where(d < D // 2, -1.0, 1.0)
    cos = tl.load(cos_ptr + b * cos_sb + t * D + d)
    sin_sw = tl.load(sin_ptr + b * cos_sb + t * D + swap)
    out = dqkv_ptr + n * ((NQ + 2 * NKV) * D)

    h = tl.arange(0, HEADS)[:, None]
    heads = h < NQ
    g = tl.load(dq_ptr + n * dq_sr + h * dq_sh + d, mask=heads, other=0.0).to(tl.float32)
    g_sw = tl.load(dq_ptr + n * dq_sr + h * dq_sh + swap, mask=heads, other=0.0).to(tl.float32)
    tl.store(out + h * D + d, (g * cos - (sign * g_sw) * sin_sw).to(tl.float16), mask=heads)

    j = tl.arange(0, KV_HEADS)[:, None]
    kv_heads = j < NKV
    gk = tl.zeros([KV_HEADS, D], dtype=tl.float32)
    gk_sw = tl.zeros([KV_HEADS, D], dtype=tl.float32)
    gv = tl.zeros([KV_HEADS, D], dtype=tl.float32)
    for r in tl.static_range(NQ // NKV):  # the repeated heads of kv head j
        head = j * (NQ // NKV) + r
        gk += tl.load(dk_ptr + n * dk_sr + head * dk_sh + d, mask=kv_heads, other=0.0).to(tl.float32)
        gk_sw += tl.load(dk_ptr + n * dk_sr + head * dk_sh + swap, mask=kv_heads,
                         other=0.0).to(tl.float32)
        gv += tl.load(dv_ptr + n * dv_sr + head * dv_sh + d, mask=kv_heads, other=0.0).to(tl.float32)
    tl.store(out + (NQ + j) * D + d, (gk * cos - (sign * gk_sw) * sin_sw).to(tl.float16),
             mask=kv_heads)
    tl.store(out + (NQ + NKV + j) * D + d, gv.to(tl.float16), mask=kv_heads)


def _rows(g):
    return g if g.stride(-1) == 1 else g.contiguous()


class QKVRope(torch.autograd.Function):
    """qkv [n, (NQ + 2 NKV) D] fp16 rows of the real tokens (+ LoRA outputs for q and v) ->
    q, k, v [n, NQ, D] fp16, k/v repeated to the NQ query heads."""

    @staticmethod
    def forward(ctx, qkv, lq, lv, cos, sin, pos, seq, nq, nkv, head_dim):
        n = qkv.shape[0]
        qkv, cos, sin = qkv.contiguous(), cos.contiguous(), sin.contiguous()
        cos_sb = 0 if cos.shape[0] == 1 else seq * head_dim
        q, k, v = (torch.empty((n, nq, head_dim), dtype=torch.float16, device=qkv.device)
                   for _ in range(3))
        heads = triton.next_power_of_2(nq)
        if n:
            qkv_rope_fwd_kernel[(n,)](
                qkv, lq.contiguous() if lq is not None else qkv,
                lv.contiguous() if lv is not None else qkv, cos, sin, pos, q, k, v, seq, cos_sb,
                NQ=nq, NKV=nkv, D=head_dim, HEADS=heads, HAS_LQ=lq is not None,
                HAS_LV=lv is not None, num_warps=4)
        ctx.save_for_backward(cos, sin, pos)
        ctx.meta = (n, seq, nq, nkv, head_dim, cos_sb, heads)
        ctx.has_lora = (lq is not None, lv is not None)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin, pos = ctx.saved_tensors
        n, seq, nq, nkv, head_dim, cos_sb, heads = ctx.meta
        dqkv = torch.empty((n, (nq + 2 * nkv) * head_dim), dtype=torch.float16, device=cos.device)
        dq, dk, dv = (_rows(g.to(torch.float16)) for g in (dq, dk, dv))
        if n:
            qkv_rope_bwd_kernel[(n,)](
                dq, dk, dv, cos, sin, pos, dqkv, seq, cos_sb, dq.stride(0), dq.stride(1),
                dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1), NQ=nq, NKV=nkv,
                D=head_dim, HEADS=heads, KV_HEADS=triton.next_power_of_2(nkv), num_warps=4)
        dlq = dqkv[:, :nq * head_dim] if ctx.has_lora[0] else None
        dlv = dqkv[:, (nq + nkv) * head_dim:] if ctx.has_lora[1] else None
        return dqkv, dlq, dlv, None, None, None, None, None, None, None


def qkv_rope(qkv, lq, lv, cos, sin, pos, seq, nq, nkv, head_dim):
    """pos: each row's flattened position in the padded [B, seq] batch (for RoPE's angle)."""
    return QKVRope.apply(qkv, lq, lv, cos, sin, pos, seq, nq, nkv, head_dim)
