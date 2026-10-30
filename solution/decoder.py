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


def _lora_out(parts, x32):
    if parts is None:
        return None
    dropout, lora_a, lora_b, scaling = parts
    return lora_b(lora_a(dropout(x32))) * scaling  # PEFT's own expression, under autocast


def _layer(plan, layer, x, pending, mask, cos, sin):
    batch, seq, _ = x.shape
    rep = plan.nq // plan.nkv if mask is not None else 1
    total, h16, h32 = add_rmsnorm(x, pending, layer.norm1.weight, plan.eps,
                                  want32=layer.lora_q is not None or layer.lora_v is not None)
    residual = total if pending is not None else x
    qkv = nf4_linear(h16, layer.qkv, layer.qkv_bias)
    lq, lv = _lora_out(layer.lora_q, h32), _lora_out(layer.lora_v, h32)
    del h32
    q, k, v = qkv_rope(qkv, lq, lv, cos, sin, plan.nq, plan.nkv, rep, plan.head_dim)
    attn = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, scale=layer.scaling,
        is_causal=mask is None and seq > 1, **({} if rep > 1 else {"enable_gqa": True}))
    attn = attn.transpose(1, 2).reshape(batch, seq, plan.nq * plan.head_dim)
    residual, m16, _ = add_rmsnorm(residual, nf4_linear(attn, layer.o), layer.norm2.weight,
                                   plan.eps)
    gate, up = nf4_linear(m16, layer.gate_up)
    return residual, nf4_linear(swiglu(gate, up), layer.down)


def decoder_forward(model, plan, input_ids, attention_mask, position_ids):
    """last_hidden_state (fp16) of a labelled training forward."""
    from transformers.masking_utils import create_causal_mask

    embeds = model.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
    mask = create_causal_mask(config=model.config, inputs_embeds=embeds,
                              attention_mask=attention_mask, past_key_values=None,
                              position_ids=position_ids)
    cos, sin = model.rotary_emb(embeds, position_ids)
    x, pending = embeds, None
    for layer in plan.layers:
        x, pending = _layer(plan, layer, x, pending, mask, cos, sin)
    _, hidden, _ = add_rmsnorm(x, pending, plan.final_norm.weight, plan.eps)
    return hidden


def prepare_decoder(model):
    """Attach a plan to a Qwen2Model whose layers we fully support; False otherwise."""
    if not ModelPlan.supported(model):
        return False
    setattr(model, PLAN_ATTR, ModelPlan(model))
    return True


def patch_qwen2_model():
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2Model.forward
    if getattr(original, "_soupk", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        plan = getattr(self, PLAN_ATTR, None)
        if (plan is None or not self.training or not torch.is_grad_enabled() or use_cache
                or past_key_values is not None or input_ids is None or inputs_embeds is not None
                or kwargs.get("output_hidden_states") or kwargs.get("output_attentions")
                or not input_ids.is_cuda):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = decoder_forward(self, plan, input_ids, attention_mask, position_ids)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)

    forward._soupk = True
    modeling_qwen2.Qwen2Model.forward = forward
