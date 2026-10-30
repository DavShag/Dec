"""The q/k/v epilogue of the attention block in one pass each way.

After the fused q|k|v GEMM the base run does, per layer: three fp16 -> fp32 output casts, a clone
and an add per LoRA projection, RoPE as q*cos + rotate_half(q)*sin (slices, a cat, a neg, two muls,
an add, for q and for k), repeat_kv's expanded copies, and three fp32 -> fp16 casts for SDPA. Here
one program per token reads its q|k|v row and the LoRA outputs, adds them in fp32, rotates q and k
in fp32 (rotate_half as a load of the other half with a sign), and writes the fp16 tensors SDPA
receives in the base run, with the same memory layouts (q token-major, k/v expanded to all query
heads). Backward is the transpose: inverse rotation, the sum over the repeated k/v heads, one fp16
gradient row for the q|k|v GEMM, whose q and v parts are also the LoRA outputs' gradients.
RoPE as in Su et al. 2021 (RoFormer), in the rotate_half form of transformers' Qwen2.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qkv_rope_fwd_kernel(qkv_ptr, lq_ptr, lv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, comp_ptr,
                        T, cos_sb, q_sb, q_sh, q_st, kv_sb, kv_sh, kv_st,
                        NQ: tl.constexpr, NKV: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                        HEADS: tl.constexpr, HAS_LQ: tl.constexpr, HAS_LV: tl.constexpr,
                        UNPAD: tl.constexpr):
    """One program per padded position; with UNPAD the inputs are the compact rows of the real
    tokens (comp maps a padded position to its row, -1 for padding, which gets zeros)."""
    pos = tl.program_id(0).to(tl.int64)
    b, t = pos // T, pos % T
    if UNPAD:
        n = tl.load(comp_ptr + pos).to(tl.int64)
        real = n >= 0
        n = tl.maximum(n, 0)
    else:
        n = pos
        real = pos >= 0
    d = tl.arange(0, D)[None, :]
    swap = (d + D // 2) % D
    sign = tl.where(d < D // 2, -1.0, 1.0)
    cos = tl.load(cos_ptr + b * cos_sb + t * D + d)
    sin = tl.load(sin_ptr + b * cos_sb + t * D + d)
    h = tl.arange(0, HEADS)[:, None]
    row = qkv_ptr + n * ((NQ + 2 * NKV) * D)

    in_q = (h < NQ) & real
    q = tl.load(row + h * D + d, mask=in_q, other=0.0).to(tl.float32)
    q_sw = tl.load(row + h * D + swap, mask=in_q, other=0.0).to(tl.float32)
    if HAS_LQ:
        lrow = lq_ptr + n * (NQ * D)
        q += tl.load(lrow + h * D + d, mask=in_q, other=0.0).to(tl.float32)
        q_sw += tl.load(lrow + h * D + swap, mask=in_q, other=0.0).to(tl.float32)
    q_out = q * cos + (sign * q_sw) * sin
    tl.store(q_ptr + b * q_sb + h * q_sh + t * q_st + d, q_out.to(tl.float16), mask=h < NQ)

    out_kv = h < NKV * REP
    in_kv = out_kv & real
    src = h // REP
    k = tl.load(row + (NQ + src) * D + d, mask=in_kv, other=0.0).to(tl.float32)
    k_sw = tl.load(row + (NQ + src) * D + swap, mask=in_kv, other=0.0).to(tl.float32)
    k_out = k * cos + (sign * k_sw) * sin
    tl.store(k_ptr + b * kv_sb + h * kv_sh + t * kv_st + d, k_out.to(tl.float16), mask=out_kv)
    v = tl.load(row + (NQ + NKV + src) * D + d, mask=in_kv, other=0.0).to(tl.float32)
    if HAS_LV:
        v += tl.load(lv_ptr + n * (NKV * D) + src * D + d, mask=in_kv, other=0.0).to(tl.float32)
    tl.store(v_ptr + b * kv_sb + h * kv_sh + t * kv_st + d, v.to(tl.float16), mask=out_kv)


@triton.jit
def qkv_rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, dqkv_ptr, pidx_ptr, T, cos_sb,
                        dq_sb, dq_sh, dq_st, dk_sb, dk_sh, dk_st, dv_sb, dv_sh, dv_st,
                        NQ: tl.constexpr, NKV: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                        HEADS: tl.constexpr, KV_HEADS: tl.constexpr, UNPAD: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    if UNPAD:
        pos = tl.load(pidx_ptr + n).to(tl.int64)
    else:
        pos = n
    b, t = pos // T, pos % T
    d = tl.arange(0, D)[None, :]
    swap = (d + D // 2) % D
    sign = tl.where(d < D // 2, -1.0, 1.0)
    cos = tl.load(cos_ptr + b * cos_sb + t * D + d)
    sin_sw = tl.load(sin_ptr + b * cos_sb + t * D + swap)
    out = dqkv_ptr + n * ((NQ + 2 * NKV) * D)

    h = tl.arange(0, HEADS)[:, None]
    in_q = h < NQ
    g = tl.load(dq_ptr + b * dq_sb + h * dq_sh + t * dq_st + d, mask=in_q, other=0.0).to(tl.float32)
    g_sw = tl.load(dq_ptr + b * dq_sb + h * dq_sh + t * dq_st + swap, mask=in_q,
                   other=0.0).to(tl.float32)
    tl.store(out + h * D + d, (g * cos - (sign * g_sw) * sin_sw).to(tl.float16), mask=in_q)

    j = tl.arange(0, KV_HEADS)[:, None]
    in_kv = j < NKV
    gk = tl.zeros([KV_HEADS, D], dtype=tl.float32)
    gk_sw = tl.zeros([KV_HEADS, D], dtype=tl.float32)
    gv = tl.zeros([KV_HEADS, D], dtype=tl.float32)
    for r in tl.static_range(REP):
        head = j * REP + r
        base_k = dk_ptr + b * dk_sb + head * dk_sh + t * dk_st
        gk += tl.load(base_k + d, mask=in_kv, other=0.0).to(tl.float32)
        gk_sw += tl.load(base_k + swap, mask=in_kv, other=0.0).to(tl.float32)
        gv += tl.load(dv_ptr + b * dv_sb + head * dv_sh + t * dv_st + d, mask=in_kv,
                      other=0.0).to(tl.float32)
    tl.store(out + (NQ + j) * D + d, (gk * cos - (sign * gk_sw) * sin_sw).to(tl.float16),
             mask=in_kv)
    tl.store(out + (NQ + NKV + j) * D + d, gv.to(tl.float16), mask=in_kv)


class QKVRope(torch.autograd.Function):
    """qkv [B, T, (NQ + 2 NKV) D] fp16 (+ LoRA outputs for q and v) -> q, k, v for SDPA:
    q [B, NQ, T, D] in token-major memory, k/v [B, NKV*REP, T, D] contiguous, all fp16. With
    `unpad` the inputs are [n, ...] rows of the real tokens and the outputs are still padded (zeros
    at the padding positions, which the attention mask hides)."""

    @staticmethod
    def forward(ctx, qkv, lq, lv, cos, sin, nq, nkv, rep, head_dim, unpad):
        if unpad is not None:  # qkv [n, W] compact rows; unpad = (batch, seq, pidx, comp)
            batch, seq, pidx, comp = unpad
        else:
            batch, seq, _ = qkv.shape
            pidx = comp = None
        qkv = qkv.contiguous()
        cos, sin = cos.contiguous(), sin.contiguous()
        cos_sb = 0 if cos.shape[0] == 1 else seq * head_dim
        q = torch.empty((batch, seq, nq, head_dim), dtype=torch.float16,
                        device=qkv.device).transpose(1, 2)
        k = torch.empty((batch, nkv * rep, seq, head_dim), dtype=torch.float16, device=qkv.device)
        v = torch.empty_like(k)
        lq_c = lq.contiguous() if lq is not None else qkv
        lv_c = lv.contiguous() if lv is not None else qkv
        heads = triton.next_power_of_2(max(nq, nkv * rep))
        qkv_rope_fwd_kernel[(batch * seq,)](
            qkv, lq_c, lv_c, cos, sin, q, k, v, comp if comp is not None else qkv, seq, cos_sb,
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), NQ=nq,
            NKV=nkv, REP=rep, D=head_dim, HEADS=heads, HAS_LQ=lq is not None,
            HAS_LV=lv is not None, UNPAD=comp is not None, num_warps=4)
        ctx.save_for_backward(cos, sin, *(() if pidx is None else (pidx,)))
        ctx.rows = qkv.shape[:-1]
        ctx.meta = (batch, seq, nq, nkv, rep, head_dim, cos_sb, heads)
        ctx.has_lora = (lq is not None, lv is not None)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin, *pidx = ctx.saved_tensors
        batch, seq, nq, nkv, rep, head_dim, cos_sb, heads = ctx.meta
        dqkv = torch.empty((*ctx.rows, (nq + 2 * nkv) * head_dim), dtype=torch.float16,
                           device=cos.device)
        dq, dk, dv = (g.to(torch.float16) for g in (dq, dk, dv))
        rows = dqkv.numel() // dqkv.shape[-1]
        qkv_rope_bwd_kernel[(rows,)](
            dq, dk, dv, cos, sin, dqkv, pidx[0] if pidx else dqkv, seq, cos_sb, dq.stride(0),
            dq.stride(1), dq.stride(2), dk.stride(0), dk.stride(1), dk.stride(2), dv.stride(0),
            dv.stride(1), dv.stride(2), NQ=nq, NKV=nkv, REP=rep, D=head_dim, HEADS=heads,
            KV_HEADS=triton.next_power_of_2(nkv), UNPAD=bool(pidx), num_warps=4)
        dlq = dqkv[..., :nq * head_dim] if ctx.has_lora[0] else None
        dlv = dqkv[..., (nq + nkv) * head_dim:] if ctx.has_lora[1] else None
        return dqkv, dlq, dlv, None, None, None, None, None, None, None


def qkv_rope(qkv, lq, lv, cos, sin, nq, nkv, rep, head_dim, unpad=None):
    """unpad = (batch, seq, pidx, comp) when qkv/lq/lv hold only the real tokens' rows: pidx is
    each row's padded position, comp the row of each padded position (-1 for padding)."""
    return QKVRope.apply(qkv, lq, lv, cos, sin, nq, nkv, rep, head_dim, unpad)
