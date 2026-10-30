"""The Qwen2 decoder stack of a training forward, built from our kernels, on the real tokens only.

Per layer, the base run is HF's Qwen2DecoderLayer with bitsandbytes 4-bit linears and PEFT LoRA on
q_proj / v_proj, on the padded batch, everything around the GEMMs in separate fp32 kernels. Here,
on the [n, hidden] rows of the n real tokens:

    keep_q, keep_v = dropout masks                        PyTorch's own kernel, base run's RNG order
    s, h16, xq, xv = add_rmsnorm(residual, pending, keeps)    norm.py: add + RMSNorm + dropout
    qkv            = nf4_linear(h16, [Wq; Wk; Wv], bias)      nf4.py: one dequant launch, one GEMM
    lq, lv         = LoRA_B(LoRA_A(xq / xv)) * scaling        PEFT's modules, under autocast
    q, k, v        = qkv_rope(qkv, lq, lv, cos, sin)          rope.py: LoRA add, RoPE, GQA repeat
    attn           = causal SDPA per row                      the padded, masked SDPA's result
    s2, m16        = add_rmsnorm(s, nf4_linear(attn, Wo))
    gate, up       = nf4_linear(m16, [Wg; Wu])
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
from torch.utils.checkpoint import checkpoint

from .nf4 import NF4Weight, nf4_linear, nf4_supported
from .norm import add_rmsnorm
from .rope import qkv_rope
from .swiglu import swiglu

PLAN_ATTR = "_soupk_plan"
# Saved activations allowed per step; a step that would keep more recomputes its first layers.
# Chosen from a sweep on the T4 (2400 / 1400 / 900 / 500 MiB): at 1300 the largest batches give up
# ~2-3% of speed for about a third less peak memory; lower budgets cost more speed than they save.
ACTIVATION_BUDGET_MB = 1300


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
        # bytes a layer keeps for backward per token: two fp32 norm inputs, two fp16 adapter
        # inputs, two keep masks, q/k/v and the attention output in fp16, gate and up in fp16
        self.kept_per_token = 22 * hidden + 4 * inter
        self.recomputed_per_token = 6 * hidden  # a recomputed layer keeps only its inputs

    def recompute_count(self, tokens):
        """How many leading layers this step recomputes in backward to stay within the budget.

        The peak memory of the run is set by its largest batches, so only those pay for
        recomputation; shorter batches run without it."""
        layers = len(self.layers)
        excess = tokens * self.kept_per_token * layers - ACTIVATION_BUDGET_MB * 2**20
        if excess <= 0:
            return 0
        saved_per_layer = tokens * (self.kept_per_token - self.recomputed_per_token)
        return min(layers, math.ceil(excess / saved_per_layer))

    @staticmethod
    def supported(model):
        config = model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return (config.hidden_act == "silu" and config.attention_dropout == 0.0
                and head_dim in (64, 128) and config.num_attention_heads % config.num_key_value_heads == 0
                and all(t == "full_attention" for t in config.layer_types)
                and getattr(config, "_attn_implementation", "sdpa") == "sdpa"
                and model.norm.weight.dtype == torch.float32
                and all(LayerPlan.supported(layer) for layer in model.layers))


class Unpad:
    """The real tokens of a right-padded batch: every op runs on these rows only.

    Removing padding inside a batch is allowed by the task; rows of different batches are never
    mixed. Attention runs per row with is_causal (the memory-efficient SDPA kernel then skips the
    blocks above the diagonal), which for right padding is exactly what the base run's padded,
    masked SDPA computes for the real tokens, at a fraction of the work. Packing variable-length
    rows this way follows FlashAttention-2's varlen interface (Dao 2023)."""

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


def _attention(q, k, v, lengths, scaling):
    """Causal SDPA per row on [n, heads, head_dim] rows; [n, heads * head_dim] out."""
    rows = []
    for qr, kr, vr in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
        out = torch.nn.functional.scaled_dot_product_attention(
            qr.transpose(0, 1)[None], kr.transpose(0, 1)[None], vr.transpose(0, 1)[None],
            dropout_p=0.0, scale=scaling, is_causal=qr.shape[0] > 1)
        rows.append(out[0].transpose(0, 1).reshape(qr.shape[0], -1))
    return torch.cat(rows) if len(rows) > 1 else rows[0]


def _layer(plan, layer, x, pending, cos, sin, unpad):
    p_q, p_v = _dropout_p(layer.lora_q), _dropout_p(layer.lora_v)
    if p_q is not None and p_v is not None:
        # the q then v masks, in the base run's RNG order; the norm kernel applies them
        hidden = x.shape[-1]
        lora = (unpad.keep(p_q, hidden), unpad.keep(p_v, hidden), unpad.pidx, 1.0 / (1.0 - p_q),
                1.0 / (1.0 - p_v))
        total, h16, _, xq, xv = add_rmsnorm(x, pending, layer.norm1.weight, plan.eps, lora=lora)
        (_, a_q, b_q, s_q), (_, a_v, b_v, s_v) = layer.lora_q, layer.lora_v
        lq, lv = b_q(a_q(xq)) * s_q, b_v(a_v(xv)) * s_v  # PEFT's expression, under autocast
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
    attn = _attention(q, k, v, unpad.lengths, layer.scaling)
    residual, m16, _, _, _ = add_rmsnorm(residual, nf4_linear(attn, layer.o), layer.norm2.weight,
                                         plan.eps)
    gate, up = nf4_linear(m16, layer.gate_up)
    return residual, nf4_linear(swiglu(gate, up), layer.down)


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
    recompute = plan.recompute_count(unpad.pidx.numel())
    for index, layer in enumerate(plan.layers):
        if index < recompute:
            # our own recomputation (allowed by the task; Soup's gradient_checkpointing stays off):
            # backward reruns this layer from its inputs; the RNG state is restored for the
            # rerun, so its dropout masks are the ones drawn now
            x, pending = checkpoint(_layer, plan, layer, x, pending, cos, sin, unpad,
                                    use_reentrant=False, preserve_rng_state=True)
        else:
            x, pending = _layer(plan, layer, x, pending, cos, sin, unpad)
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
