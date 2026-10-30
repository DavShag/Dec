"""The Qwen2 decoder stack of a training forward, built from our kernels.

Per layer, the base run is HF's Qwen2DecoderLayer with bitsandbytes 4-bit linears and PEFT LoRA on
q_proj / v_proj, everything around the GEMMs in separate fp32 kernels. Here:

    s, h16, h32 = add_rmsnorm(residual, pending)            norm.py: add + RMSNorm, fp16 out
    qkv         = nf4_linear(h16, [Wq; Wk; Wv], bias)       nf4.py: one dequant pass, one GEMM
    lq, lv      = LoRA_B(LoRA_A(dropout(h32))) * scaling    PEFT's modules, same RNG calls
    q, k, v     = qkv_rope(qkv, lq, lv, cos, sin)           rope.py: LoRA add, RoPE, layouts
    attn        = SDPA(q, k, v, mask)                       as transformers calls it
    s2, m16     = add_rmsnorm(s, nf4_linear(attn, Wo))
    gate, up    = nf4_linear(m16, [Wg; Wu])
    pending     = nf4_linear(swiglu(gate, up), Wd)          the residual add moves into the
                                                            next layer's norm
What stays the same as the base: the 4-bit weights and their values, every fp16 rounding point in
front of a GEMM, fp32 for the residual stream, norm statistics, RoPE and LoRA adds, the dropout
calls (same shapes, dtype and order, so the same masks), the mask, the SDPA call and the layouts
it sees. What goes away: fp32 copies of fp16 GEMM outputs, fp32 MLP intermediates, three casts of
each norm output, the separate elementwise kernels, and the saved tensors that came with them.
"""

import torch

from .nf4 import NF4Weight, nf4_linear, nf4_supported
from .norm import add_rmsnorm
from .rope import qkv_rope
from .swiglu import swiglu

PLAN_ATTR = "_soupk_plan"


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

    def dropout(self, x, p):
        """F.dropout of the padded [B, T, H] fp32 input, on the real rows: the same RNG draw (same
        shape, dtype and op as the base run, so the same keep mask), then x * (1/(1-p)) where kept,
        which is the exact product the dropout kernel forms."""
        if self.ones is None or self.ones.shape[-1] != x.shape[-1]:
            self.ones = torch.ones((self.batch, self.seq, x.shape[-1]), dtype=torch.float32,
                                   device=x.device)
        keep = torch.native_dropout(self.ones, p, True)[1]
        keep = keep.view(-1, x.shape[-1]).index_select(0, self.pidx)
        return torch.where(keep, x * (1.0 / (1.0 - p)), 0.0)


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
    total, h16, h32 = add_rmsnorm(x, pending, layer.norm1.weight, plan.eps,
                                  want32=layer.lora_q is not None or layer.lora_v is not None)
    residual = total if pending is not None else x
    qkv = nf4_linear(h16, layer.qkv, layer.qkv_bias)
    lq, lv = _lora_out(layer.lora_q, h32, unpad), _lora_out(layer.lora_v, h32, unpad)
    del h32
    q, k, v = qkv_rope(qkv, lq, lv, cos, sin, unpad.pidx, unpad.seq, plan.nq, plan.nkv,
                       plan.head_dim)
    attn = _attention(q, k, v, unpad.lengths, layer.scaling)
    residual, m16, _ = add_rmsnorm(residual, nf4_linear(attn, layer.o), layer.norm2.weight,
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
    for layer in plan.layers:
        x, pending = _layer(plan, layer, x, pending, cos, sin, unpad)
    _, hidden, _ = add_rmsnorm(x, pending, plan.final_norm.weight, plan.eps)
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
