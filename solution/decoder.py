"""The Qwen2 decoder stack of a training forward, built from our kernels, on the real tokens only.

Per layer, the base run is HF's Qwen2DecoderLayer with bitsandbytes 4-bit linears and PEFT LoRA on
q_proj / v_proj, on the padded batch, everything around the GEMMs in separate fp32 kernels. Here,
on the [n, hidden] rows of the n real tokens:

    keep_q, keep_v = dropout masks                        PyTorch's own kernel, base run's RNG order
    s, h16, zq, zv = add_rmsnorm(residual, pending, keeps, A)  norm.py: add + RMSNorm + dropout,
                                                              then the adapters' A GEMMs
    qkv            = nf4_linear(h16, [Wq; Wk; Wv], bias)      nf4.py: one dequant launch, one GEMM
    lq, lv         = LoRA_B(zq / zv) * scaling                PEFT's modules, under autocast
    q, k, v        = qkv_rope(qkv, lq, lv, cos, sin)          rope.py: LoRA add, RoPE, GQA repeat
    attn           = causal SDPA of every row, one call       the padded, masked SDPA's result;
                                                              backward keeps k/v un-repeated
    s2, m16        = add_rmsnorm(s, nf4_linear(attn, Wo))
    gate, up       = nf4_linear(m16, [Wg; Wu])                (RecomputedMLP: not kept, repeated
                                                              in backward from s2)
    pending        = nf4_linear(swiglu(gate, up), Wd)         the residual add moves into the
                                                              next layer's norm kernel
What stays the same as the base: the 4-bit weights and their values, every fp16 rounding point in
front of a GEMM, fp32 for the residual stream, norm statistics, RoPE and LoRA adds, the dropout
masks (same kernel, shape, dtype and order of draws) and the attention each real token sees. What
goes away: the padding, fp32 copies of fp16 GEMM outputs, fp32 MLP intermediates, three casts of
each norm output, the separate elementwise kernels, and the saved tensors that came with them.
"""

import math

import torch
import triton
from torch.utils.checkpoint import checkpoint

from .nf4 import NF4Weight, dequantize_stack, nf4_linear, nf4_supported
from .norm import add_rmsnorm, rmsnorm16
from .rope import qkv_rope
from .swiglu import BLOCK as SWIGLU_BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

PLAN_ATTR = "_soupk_plan"
# Saved activations allowed per step; a step that would keep more first drops gate/up of some
# layers (they repeat the gate|up GEMMs in backward), then recomputes whole leading layers.
# Kaggle T4, one session: 1300 MiB -> peak 2,867 MiB; 1000 MiB -> 2,547 MiB for ~2 s more of the
# run; 800 MiB -> 2,347 MiB for ~3 s more. With the reported best runs (~37 s, ~2.2 GB) the scores
# of the three are within noise, and a tie goes to the lower memory: we use 800 (README,
# "Adaptive recomputation").
ACTIVATION_BUDGET_MB = 300
_ATEN = torch.ops.aten
_EFFICIENT_ATTENTION = all(hasattr(_ATEN, name) for name in (
    "_efficient_attention_forward", "_efficient_attention_backward"))


def _base(module):
    return getattr(module, "base_layer", module)


def _lora(module):
    """(dropout, lora_A, lora_B, scaling) of a PEFT LoRA layer with one plain adapter, None for a
    plain 4-bit linear, False for anything else."""
    if not hasattr(module, "lora_A"):
        return None if nf4_supported(module) else False
    names = list(getattr(module, "active_adapters", []))
    if (len(names) != 1 or names[0] not in module.lora_A or getattr(module, "merged", False)
            or getattr(module, "disable_adapters", False) or names[0] in module.lora_variant
            or not nf4_supported(module.base_layer)):
        return False
    name = names[0]
    if module.lora_A[name].bias is not None:  # its GEMM runs inside the norm (AddRMSNorm adapters)
        return False
    return module.lora_dropout[name], module.lora_A[name], module.lora_B[name], module.scaling[name]


class LayerPlan:
    def __init__(self, layer):
        attn, mlp = layer.self_attn, layer.mlp
        self.layer = layer
        self.lora_q, self.lora_v = _lora(attn.q_proj), _lora(attn.v_proj)
        q, k, v, o = (_base(m) for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj))
        self.qkv = [NF4Weight(m.weight) for m in (q, k, v)]
        self.qkv_bias = torch.cat([q.bias, k.bias, v.bias]).detach().to(torch.float16)
        self.o = [NF4Weight(o.weight)]
        self.gate_up = [NF4Weight(_base(mlp.gate_proj).weight), NF4Weight(_base(mlp.up_proj).weight)]
        self.down = [NF4Weight(_base(mlp.down_proj).weight)]
        self.norm1, self.norm2 = layer.input_layernorm, layer.post_attention_layernorm
        self.scaling = attn.scaling

    @staticmethod
    def supported(layer):
        attn, mlp = layer.self_attn, layer.mlp
        projections = (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.gate_proj,
                       mlp.up_proj, mlp.down_proj)
        q, k, v, o, gate, up, down = (_lora(m) for m in projections)
        if q is False or v is False or any(x is not None for x in (k, o, gate, up, down)):
            return False  # LoRA only on q/v, every other projection a plain NF4 linear
        if any(_base(m).bias is not None for m in (attn.o_proj, mlp.gate_proj, mlp.up_proj,
                                                    mlp.down_proj)):
            return False
        if any(_base(m).bias is None for m in (attn.q_proj, attn.k_proj, attn.v_proj)):
            return False
        norms = (layer.input_layernorm, layer.post_attention_layernorm)
        return (all(n.weight.dtype == torch.float32 and not n.weight.requires_grad for n in norms)
                and getattr(attn, "sliding_window", None) is None)


class ModelPlan:
    def __init__(self, model):
        config = model.config
        self.layers = [LayerPlan(layer) for layer in model.layers[:config.num_hidden_layers]]
        self.eps = config.rms_norm_eps
        self.nq, self.nkv = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.nq
        self.final_norm = model.norm
        hidden, inter = config.hidden_size, config.intermediate_size
        q_size, kv_size = self.nq * self.head_dim, self.nkv * self.head_dim
        # bytes a layer keeps for backward per token: the fp32 input of norm1 and, for norm2, the
        # fp16 o_proj output added to it (norm2's input is formed again from both), the adapters'
        # keep masks packed 8 per byte (their fp16 inputs are rebuilt, AddRMSNorm adapters), q,
        # the distinct k/v heads and the attention output in fp16, gate and up in fp16
        self.kept_per_token = (4 * hidden + 2 * hidden + 2 * hidden // 8
                               + 2 * (2 * q_size + 2 * kv_size) + 2 * 2 * inter)
        self.mlp_dropped_per_token = 2 * 2 * inter  # gate and up, for a RecomputedMLP layer
        self.recomputed_per_token = 6 * hidden  # a recomputed layer keeps only its inputs

    def recompute_plan(self, tokens):
        """(whole layers recomputed, layers that only repeat gate|up) for a step of `tokens`.

        Within the budget a step keeps everything. Above it, layers first drop gate and up
        (RecomputedMLP: 4 * inter bytes per token for the gate|up GEMMs again, about half of what
        a whole layer costs to recompute, for two thirds of its bytes); only if every layer has
        done so and the step is still over budget are leading layers recomputed whole. The peak
        memory of the run is set by its largest batches, so shorter ones pay little or nothing."""
        layers = len(self.layers)
        excess = tokens * self.kept_per_token * layers - ACTIVATION_BUDGET_MB * 2**20
        if excess <= 0:
            return 0, 0
        dropped = tokens * self.mlp_dropped_per_token
        mlp = min(layers, math.ceil(excess / dropped))
        excess -= mlp * dropped
        if excess <= 0:
            return 0, mlp
        rest = self.kept_per_token - self.mlp_dropped_per_token - self.recomputed_per_token
        full = min(layers, math.ceil(excess / (tokens * rest)))
        return full, layers - full

    @staticmethod
    def supported(model):
        config = model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return (_EFFICIENT_ATTENTION and config.hidden_act == "silu"
                and config.attention_dropout == 0.0 and config.hidden_size % 8 == 0
                and head_dim in (64, 128) and config.num_attention_heads % config.num_key_value_heads == 0
                and all(t == "full_attention" for t in config.layer_types)
                and getattr(config, "_attn_implementation", "sdpa") == "sdpa"
                and model.norm.weight.dtype == torch.float32
                and all(LayerPlan.supported(layer) for layer in model.layers))


class Unpad:
    """The real tokens of a right-padded batch: every op runs on these rows only.

    Removing padding inside a batch is allowed by the task; rows of different batches are never
    mixed. Attention runs on each row's tokens, causally (the memory-efficient SDPA kernel then
    skips the blocks above the diagonal), which for right padding is exactly what the base run's
    padded, masked SDPA computes for the real tokens, at a fraction of the work. Packing
    variable-length rows this way follows FlashAttention-2's varlen interface (Dao 2023)."""

    def __init__(self, attention_mask, batch, seq, device):
        steps = torch.arange(seq, device=device)
        if attention_mask is None:
            lengths = torch.full((batch,), seq, device=device)
            self.right_padded = True
        else:
            lengths = attention_mask.sum(1)
            self.right_padded = bool(((attention_mask != 0) == (steps < lengths[:, None])).all())
        self.batch, self.seq = batch, seq
        self.pidx = (steps < lengths[:, None]).reshape(-1).nonzero().squeeze(1)
        self.lengths = lengths.tolist()
        # where each row starts among the real tokens, for attention (computed on the GPU: no copy)
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        self.rows = (offsets, max(self.lengths))
        self.ones = None

    def keep(self, p, hidden):
        """The keep mask F.dropout draws for a padded [B, T, hidden] fp32 input: the same kernel on
        the same shape and dtype, so the same RNG draw and the same mask as the base run, as a
        [B*T, hidden] bool tensor over the padded positions."""
        if self.ones is None or self.ones.shape[-1] != hidden:
            self.ones = torch.ones((self.batch, self.seq, hidden), dtype=torch.float32,
                                   device=self.pidx.device)
        return torch.native_dropout(self.ones, p, True)[1].view(-1, hidden)

    def dropout(self, x, p):
        """F.dropout of the padded fp32 input, on the real rows: x * (1/(1-p)) where kept, the
        exact product the dropout kernel forms."""
        keep = self.keep(p, x.shape[-1]).index_select(0, self.pidx)
        return torch.where(keep, x * (1.0 / (1.0 - p)), 0.0)


def _dropout_p(parts):
    """p of an adapter's dropout when it is an active nn.Dropout, else None."""
    dropout = parts[0] if parts else None
    if isinstance(dropout, torch.nn.Dropout) and dropout.training and dropout.p > 0:
        return dropout.p
    return None


def _lora_out(parts, x32, unpad):
    if parts is None:
        return None
    dropout, lora_a, lora_b, scaling = parts
    if isinstance(dropout, torch.nn.Dropout) and dropout.training and dropout.p > 0:
        dropped = unpad.dropout(x32, dropout.p)
    else:
        dropped = dropout(x32)
    return lora_b(lora_a(dropped)) * scaling  # PEFT's own expression, under autocast


class CausalAttention(torch.autograd.Function):
    """Causal attention of every row's real tokens in one call: [n, heads, head_dim] q, k, v
    (token-major, rows one after the other) -> [n, heads * head_dim].

    The memory-efficient kernel that `scaled_dot_product_attention` runs on sm75, called with the
    rows' offsets among the n tokens (FlashAttention-2's varlen interface, which the kernel takes
    as cu_seqlens): each row attends causally to its own tokens only, which for right padding is
    what the base run's padded, masked SDPA computes for the real tokens. One launch per layer and
    direction for the whole batch, the output and gradients written in place. k and v come
    repeated to every query head (rope.py; the kernel has no grouped-query mode). Backward keeps
    only their distinct kv heads and repeats them again when it runs, so the k/v saved per token
    shrink by nq / nkv."""

    @staticmethod
    def forward(ctx, q, k, v, rows, scale, group):
        offsets, longest = rows
        out, lse, seed, offset, _, _ = _ATEN._efficient_attention_forward(
            q[None], k[None], v[None], None, offsets, offsets, longest, longest, 0.0, 1, True,
            scale=scale)  # mask type 1: causal, as SDPA's is_causal
        out = out[0].view(q.shape[0], -1)
        ctx.save_for_backward(q, k[:, ::group].contiguous(), v[:, ::group].contiguous(), out, lse,
                              seed, offset, offsets)
        ctx.longest, ctx.scale, ctx.group = longest, scale, group
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k_kv, v_kv, out, lse, seed, offset, offsets = ctx.saved_tensors
        k = k_kv[:, :, None, :].expand(-1, -1, ctx.group, -1).reshape(k_kv.shape[0], -1, k_kv.shape[-1]).contiguous()
        v = v_kv[:, :, None, :].expand(-1, -1, ctx.group, -1).reshape(v_kv.shape[0], -1, v_kv.shape[-1]).contiguous()
        tokens = (1,) + q.shape
        dq, dk, dv, _ = _ATEN._efficient_attention_backward(
            grad.contiguous().view(tokens), q[None], k[None], v[None], None, out.view(tokens),
            offsets, offsets, ctx.longest, ctx.longest, lse, 0.0, seed, offset, 1, False,
            scale=ctx.scale)
        return dq[0], dk[0], dv[0], None, None, None


def _swiglu_out(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    example_swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](gate, up, out, n, BLOCK=SWIGLU_BLOCK)
    return out


def _swiglu_grads_in_place(grad, gate, up):
    """(dgate, dup) written over gate and up: each element is read before it is written."""
    grad = grad.reshape(gate.shape).to(torch.float16).contiguous()
    n = gate.numel()
    example_swiglu_bwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](grad, gate, up, gate, up, n,
                                                               BLOCK=SWIGLU_BLOCK)
    return gate, up


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up with the starter kit's example kernels (swiglu.py), for a gate and up that
    only this op reads (fresh NF4Linear outputs): backward writes dgate and dup over them, two
    [tokens, intermediate] buffers fewer at the backward peak, the same values."""

    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return _swiglu_out(gate, up)

    @staticmethod
    def backward(ctx, grad):
        return _swiglu_grads_in_place(grad, *ctx.saved_tensors)


class RecomputedMLP(torch.autograd.Function):
    """silu(gate) * up for [gate, up] = m16 @ [Wg; Wu].T, without keeping gate and up for backward.

    m16 is norm2's output: backward rebuilds it from norm2's input x + delta, which norm2 keeps
    anyway (the same kernel on the same values, so the same bits), repeats the gate|up GEMMs, and
    then forms exactly what SwiGLU's and NF4Linear's backward form, with the same kernels and
    GEMMs."""

    @staticmethod
    def forward(ctx, m16, x, delta, norm_weight, eps, weights):
        gate, up = nf4_linear(m16, weights)
        ctx.save_for_backward(x, delta, norm_weight)
        ctx.eps, ctx.weights = eps, weights
        return _swiglu_out(gate, up)

    @staticmethod
    def backward(ctx, grad):
        x, delta, norm_weight = ctx.saved_tensors
        flat = rmsnorm16(x, norm_weight, ctx.eps, delta=delta).view(-1, x.shape[-1])
        dense = dequantize_stack(ctx.weights, into_scratch=True)  # for both GEMMs below
        parts, row = [], 0
        for w in ctx.weights:
            parts.append(dense[row:row + w.rows])
            row += w.rows
        gate, up = _swiglu_grads_in_place(grad, *(flat @ part.t() for part in parts))
        grad_m16 = torch.addmm(gate @ parts[0], up, parts[1])  # NF4Linear's backward GEMMs
        return grad_m16.view(x.shape), None, None, None, None, None


def _layer(plan, layer, x, pending, cos, sin, unpad, recompute_mlp=False):
    p_q, p_v = _dropout_p(layer.lora_q), _dropout_p(layer.lora_v)
    if p_q is not None and p_v is not None:
        # the q then v masks, in the base run's RNG order; the norm kernel applies them, and runs
        # the adapters' A GEMMs too, so their fp16 inputs are not kept (AddRMSNorm adapters)
        hidden = x.shape[-1]
        lora = (unpad.keep(p_q, hidden), unpad.keep(p_v, hidden), unpad.pidx, 1.0 / (1.0 - p_q),
                1.0 / (1.0 - p_v))
        (_, a_q, b_q, s_q), (_, a_v, b_v, s_v) = layer.lora_q, layer.lora_v
        total, h16, _, zq, zv = add_rmsnorm(x, pending, layer.norm1.weight, plan.eps, lora=lora,
                                            adapters=(a_q.weight, a_v.weight))
        lq, lv = b_q(zq) * s_q, b_v(zv) * s_v  # PEFT's expression from lora_A on, under autocast
    else:
        total, h16, h32, _, _ = add_rmsnorm(
            x, pending, layer.norm1.weight, plan.eps,
            want32=layer.lora_q is not None or layer.lora_v is not None)
        lq, lv = _lora_out(layer.lora_q, h32, unpad), _lora_out(layer.lora_v, h32, unpad)
        del h32
    residual = total if pending is not None else x
    qkv = nf4_linear(h16, layer.qkv, layer.qkv_bias)
    q, k, v = qkv_rope(qkv, lq, lv, cos, sin, unpad.pidx, unpad.seq, plan.nq, plan.nkv,
                       plan.head_dim)
    attn = CausalAttention.apply(q, k, v, unpad.rows, layer.scaling, plan.nq // plan.nkv)
    o16 = nf4_linear(attn, layer.o)
    # norm2 keeps its fp16 delta, not the fp32 sum: its x is the layer's input, kept by norm1
    total2, m16, _, _, _ = add_rmsnorm(residual, o16, layer.norm2.weight, plan.eps, keep_sum=False)
    if recompute_mlp:
        act = RecomputedMLP.apply(m16, residual, o16, layer.norm2.weight, plan.eps,
                                  tuple(layer.gate_up))
    else:
        act = SwiGLU.apply(*nf4_linear(m16, layer.gate_up))
    return total2, nf4_linear(act, layer.down)


def decoder_forward(model, plan, input_ids, attention_mask, position_ids):
    """(last hidden state fp16 [n, H] of the n real tokens, unpad) of a labelled training
    forward, or None for a batch that is not right-padded (the caller then runs HF's forward)."""
    batch, seq = input_ids.shape
    unpad = Unpad(attention_mask, batch, seq, input_ids.device)
    if not unpad.right_padded:
        return None
    if position_ids is None:
        position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    # RoPE's cos/sin as HF computes them; the helper reads only the dtype and device of its input
    probe = torch.empty((batch, seq, 1), dtype=torch.float32, device=input_ids.device)
    cos, sin = model.rotary_emb(probe, position_ids)
    x, pending = model.embed_tokens(input_ids.reshape(-1).index_select(0, unpad.pidx)), None
    full, mlp = plan.recompute_plan(unpad.pidx.numel())
    for index, layer in enumerate(plan.layers):
        if index < full:
            # our own recomputation (allowed by the task; Soup's gradient_checkpointing stays off):
            # backward reruns this layer from its inputs; the RNG state is restored for the
            # rerun, so its dropout masks are the ones drawn now
            x, pending = checkpoint(_layer, plan, layer, x, pending, cos, sin, unpad,
                                    use_reentrant=False, preserve_rng_state=True)
        else:  # the next `mlp` layers repeat only their gate|up GEMMs in backward
            x, pending = _layer(plan, layer, x, pending, cos, sin, unpad, index < full + mlp)
    unpad.ones = None  # drawn from for the masks of the last layer; a recomputed layer makes it again
    _, hidden, _, _, _ = add_rmsnorm(x, pending, plan.final_norm.weight, plan.eps)
    return hidden, unpad


def plain_training_call(input_ids, inputs_embeds, past_key_values, use_cache, kwargs):
    """A call the fused decoder handles: token ids on CUDA, no cache, no extra outputs."""
    return (input_ids is not None and inputs_embeds is None and past_key_values is None
            and not use_cache and input_ids.is_cuda and torch.is_grad_enabled()
            and not kwargs.get("output_hidden_states") and not kwargs.get("output_attentions"))


def prepare_decoder(model):
    """Attach a plan to a Qwen2Model whose layers we fully support; False otherwise."""
    if not ModelPlan.supported(model):
        return False
    setattr(model, PLAN_ATTR, ModelPlan(model))
    return True
