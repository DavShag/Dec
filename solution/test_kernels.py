"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}
VOCAB, HIDDEN = 151936, 1536


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


def _head_batch(batch, seq, seed=0):
    """Right-padded rows like TRL's collator: prompt (-100), answer, padding (-100, mask 0)."""
    gen = torch.Generator().manual_seed(seed)
    weight = (torch.randn(VOCAB, HIDDEN, generator=gen) * 0.02).half().cuda()
    hidden = (torch.randn(batch, seq, HIDDEN, generator=gen) * 2.0).cuda()
    labels = torch.randint(0, VOCAB, (batch, seq), generator=gen).cuda()
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    for row in range(batch):
        length = seq - 7 * row
        prompt = 5 + 3 * row
        labels[row, :prompt] = -100
        labels[row, length:] = -100
        mask[row, length:] = 0
    return weight, hidden, labels, mask


def _reference(weight, hidden, labels, mask, num_items):
    logits = (hidden.half() @ weight.t()).float()
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    flat_t = targets.reshape(-1)
    token_loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), flat_t,
                                                   ignore_index=-100, reduction="none")
    loss = token_loss.sum() / (num_items if num_items is not None else (flat_t != -100).sum())
    with torch.no_grad():
        entropy = torch.cat([-(lp.exp() * lp).sum(-1) for lp in
                             (torch.log_softmax(part, -1) for part in logits.split(64, dim=1))], 1)
        hits = ((logits.argmax(-1) == targets) & (targets != -100)).sum()
    return loss, entropy, token_loss.view(targets.shape), hits, (targets != -100).sum()


@pytest.mark.parametrize("batch,seq,with_num_items", [(2, 70, False), (3, 200, True)])
def test_fused_head_matches_pytorch(batch, seq, with_num_items):
    from solution.lce import fused_linear_cross_entropy

    weight, hidden, labels, mask = _head_batch(batch, seq)
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    num_items = (targets != -100).sum() + 3 if with_num_items else None
    scale = 65536.0  # the fp16 GradScaler's starting scale

    ref_hidden = hidden.clone().requires_grad_(True)
    ref_loss, ref_ent, ref_tok, ref_hits, ref_total = _reference(weight, ref_hidden, labels, mask,
                                                                 num_items)
    (ref_loss * scale).backward()

    our_hidden = hidden.clone().requires_grad_(True)
    loss, ent, tok, hits, total = fused_linear_cross_entropy(our_hidden, weight, labels, mask,
                                                             num_items)
    (loss * scale).backward()

    torch.testing.assert_close(loss, ref_loss, rtol=2e-5, atol=1e-5)
    real = mask.bool()
    torch.testing.assert_close(ent[real], ref_ent[real], rtol=1e-3, atol=2e-3)
    assert torch.all(ent[~real] == 0)
    torch.testing.assert_close(tok, ref_tok, rtol=1e-4, atol=1e-3)
    assert int(total) == int(ref_total)
    assert abs(int(hits) - int(ref_hits)) <= 1  # argmax ties between GEMM tilings
    ref_grad, our_grad = ref_hidden.grad / scale, our_hidden.grad / scale
    rel = (our_grad - ref_grad).norm() / ref_grad.norm()
    assert rel < 2e-3, f"gradient relative error {rel:.2e}"
    assert our_hidden.grad.dtype == hidden.dtype


def test_fused_head_without_grad():
    from solution.lce import fused_linear_cross_entropy

    weight, hidden, labels, mask = _head_batch(2, 40, seed=1)
    with torch.no_grad():
        loss, ent, tok, hits, total = fused_linear_cross_entropy(hidden, weight, labels, mask)
        ref_loss, ref_ent, _, _, _ = _reference(weight, hidden, labels, mask, None)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(ent[mask.bool()], ref_ent[mask.bool()], rtol=1e-3, atol=2e-3)


def test_embedding_fp32_rows():
    import types

    from solution.integrate import _embedding_fp32_forward

    embed = torch.nn.Embedding(VOCAB, HIDDEN).cuda().half()
    embed.forward = types.MethodType(_embedding_fp32_forward, embed)
    ids = torch.randint(0, VOCAB, (3, 129), device="cuda")
    out = embed(ids)
    assert out.dtype == torch.float32 and out.shape == (3, 129, HIDDEN)
    torch.testing.assert_close(out, embed.weight.float()[ids], rtol=0, atol=0)


def test_nf4_dequant_matches_bitsandbytes():
    import bitsandbytes.functional as bnbf

    from solution.nf4 import NF4Weight, dequantize_stack

    class Param:
        pass

    torch.manual_seed(0)
    parts = []
    for rows, cols in ((256, 1536), (512, 1536), (64, 1536)):
        w = torch.randn(rows, cols, dtype=torch.float16, device="cuda")
        packed, state = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=True,
                                           quant_type="nf4")
        param = Param()
        param.data, param.quant_state = packed, state
        parts.append((NF4Weight(param), bnbf.dequantize_4bit(packed, state)))
    ours = dequantize_stack([p for p, _ in parts])
    torch.testing.assert_close(ours, torch.cat([ref for _, ref in parts]), rtol=0, atol=0)


def _ref_norm(x, delta, weight, eps):
    total = x + delta.float() if delta is not None else x
    var = total.pow(2).mean(-1, keepdim=True)
    return total, weight * (total * torch.rsqrt(var + eps))


@pytest.mark.parametrize("with_delta", [False, True])
def test_add_rmsnorm_matches_pytorch(with_delta):
    from solution.norm import add_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(3, 37, 1536, device="cuda", requires_grad=True)
    delta = (torch.randn(3, 37, 1536, device="cuda") * 3).half().requires_grad_(True) \
        if with_delta else None
    weight = torch.rand(1536, device="cuda") + 0.5
    g16 = torch.randn(3, 37, 1536, device="cuda").half()
    g32 = torch.randn(3, 37, 1536, device="cuda")
    gsum = torch.randn(3, 37, 1536, device="cuda")

    total, y16, y32, _, _ = add_rmsnorm(x, delta, weight, 1e-6, want32=True)
    loss = (y16.float() * g16.float()).sum() + (y32 * g32).sum()
    if with_delta:
        loss = loss + (total * gsum).sum()
    loss.backward()
    ours = [x.grad.clone()] + ([delta.grad.clone()] if with_delta else [])
    x.grad = None
    if with_delta:
        delta.grad = None

    ref_total, ref_y = _ref_norm(x, delta, weight, 1e-6)
    ref_loss = (ref_y * g16.float()).sum() + (ref_y * g32).sum()
    if with_delta:
        ref_loss = ref_loss + (ref_total * gsum).sum()
    ref_loss.backward()
    torch.testing.assert_close(y32, ref_y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y16, ref_y.half(), rtol=1e-3, atol=1e-3)
    if with_delta:
        torch.testing.assert_close(total, ref_total, rtol=0, atol=0)
    torch.testing.assert_close(ours[0], x.grad, rtol=1e-4, atol=1e-4)
    if with_delta:
        torch.testing.assert_close(ours[1].float(), delta.grad.float(), rtol=2e-3, atol=2e-3)


def _ref_qkv(qkv, lq, lv, cos, sin, nq, nkv, d):
    """HF's path: LoRA adds and RoPE in fp32, repeat_kv, cast to fp16; [B, H, T, D]."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    batch, seq, _ = qkv.shape
    rep = nq // nkv
    q = qkv[..., :nq * d].float() + lq.float()
    k = qkv[..., nq * d:(nq + nkv) * d].float()
    v = qkv[..., (nq + nkv) * d:].float() + lv.float()
    q = q.view(batch, seq, nq, d).transpose(1, 2)
    k = k.view(batch, seq, nkv, d).transpose(1, 2)
    v = v.view(batch, seq, nkv, d).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k[:, :, None].expand(batch, nkv, rep, seq, d).reshape(batch, nq, seq, d)
    v = v[:, :, None].expand(batch, nkv, rep, seq, d).reshape(batch, nq, seq, d)
    return q.half(), k.half(), v.half()


@pytest.mark.parametrize("padded", [False, True])
def test_qkv_rope_matches_pytorch(padded):
    from solution.decoder import Unpad
    from solution.rope import qkv_rope

    torch.manual_seed(0)
    batch, seq, nq, nkv, d = 3, 50, 12, 2, 128
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    if padded:
        mask[1, 37:] = 0
        mask[2, 9:] = 0
    unpad = Unpad(mask, batch, seq, "cuda")
    real = mask.bool()
    qkv = (torch.randn(batch, seq, (nq + 2 * nkv) * d, device="cuda") * 2).half().requires_grad_()
    lq = (torch.randn(batch, seq, nq * d, device="cuda") * 0.1).half().requires_grad_()
    lv = (torch.randn(batch, seq, nkv * d, device="cuda") * 0.1).half().requires_grad_()
    pos = torch.arange(seq, device="cuda").float()
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, device="cuda").float() / d))
    emb = torch.cat([pos[:, None] * inv[None], pos[:, None] * inv[None]], -1)[None]
    cos, sin = emb.cos(), emb.sin()
    grads = [torch.randn(batch, seq, nq, d, device="cuda").half() for _ in range(3)]

    rows = [t.reshape(batch * seq, -1).index_select(0, unpad.pidx) for t in (qkv, lq, lv)]
    outs = qkv_rope(*rows, cos, sin, unpad.pidx, seq, nq, nkv, d)
    sum(((o.float() * g[real].float()).sum() for o, g in zip(outs, grads))).backward()
    ours = [t.grad.clone() for t in (qkv, lq, lv)]
    for t in (qkv, lq, lv):
        t.grad = None
    refs = _ref_qkv(qkv, lq, lv, cos, sin, nq, nkv, d)
    sum(((o.transpose(1, 2)[real].float() * g[real].float()).sum()
         for o, g in zip(refs, grads))).backward()
    for o, r in zip(outs, refs):
        torch.testing.assert_close(o, r.transpose(1, 2)[real], rtol=1e-3, atol=1e-3)
    for o, t in zip(ours, (qkv, lq, lv)):
        torch.testing.assert_close(o.float(), t.grad.float(), rtol=2e-3, atol=2e-3)


def _tiny_qlora(tmp_path):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config,
                              Qwen2ForCausalLM)

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=1024, hidden_size=256, intermediate_size=640,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=True,
                         rms_norm_eps=1e-6, attn_implementation="sdpa")
    Qwen2ForCausalLM(config).half().save_pretrained(tmp_path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    model.train()
    return model


def _train_step(model, batch, seed):
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**batch)
    (out.loss * 1024.0).backward()
    grads = {n: p.grad.detach().clone() / 1024.0 for n, p in model.named_parameters()
             if p.requires_grad}
    model.zero_grad(set_to_none=True)
    return out.loss.detach(), grads


@pytest.mark.parametrize("padded,recompute", [(True, False), (False, False), (True, True)])
def test_fused_training_step_matches_hf(tmp_path, padded, recompute, monkeypatch):
    import solution
    from solution import decoder
    from solution.integrate import prepare_model

    monkeypatch.setattr(decoder, "ACTIVATION_BUDGET_MB", 0.0 if recompute else 1e9)

    solution.apply()
    model = _tiny_qlora(tmp_path)
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1024, (3, 40), generator=gen).cuda()
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :6] = -100
    if padded:
        for row, length in ((1, 31), (2, 17)):
            mask[row, length:] = 0
            labels[row, length:] = -100
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels, "use_cache": False}

    ref_loss, ref_grads = _train_step(model, batch, seed=3)
    assert prepare_model(model)
    from solution.decoder import PLAN_ATTR

    assert getattr(model.get_base_model().model, PLAN_ATTR, None) is not None
    loss, grads = _train_step(model, batch, seed=3)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-3, atol=2e-3)
    for name, ref in ref_grads.items():
        rel = (grads[name] - ref).norm() / ref.norm().clamp(min=1e-12)
        assert rel < 5e-3, f"{name}: relative gradient error {rel:.3e}"  # another dropout draw: ~0.5


def test_add_rmsnorm_lora_dropout_matches_pytorch():
    from solution.norm import add_rmsnorm

    torch.manual_seed(0)
    batch, seq, hidden = 2, 30, 1536
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    mask[1, 21:] = 0
    pidx = mask.reshape(-1).nonzero().squeeze(1)
    n = pidx.numel()
    x = torch.randn(n, hidden, device="cuda", requires_grad=True)
    delta = (torch.randn(n, hidden, device="cuda") * 3).half().requires_grad_(True)
    weight = torch.rand(hidden, device="cuda") + 0.5
    keep_q = torch.rand(batch * seq, hidden, device="cuda") > 0.05
    keep_v = torch.rand(batch * seq, hidden, device="cuda") > 0.05
    scale = 1.0 / (1.0 - 0.05)
    g16, gq, gv = (torch.randn(n, hidden, device="cuda").half() for _ in range(3))
    gsum = torch.randn(n, hidden, device="cuda")

    total, y16, _, xq, xv = add_rmsnorm(x, delta, weight, 1e-6,
                                        lora=(keep_q, keep_v, pidx, scale, scale))
    ((y16.float() * g16.float()).sum() + (xq.float() * gq.float()).sum()
     + (xv.float() * gv.float()).sum() + (total * gsum).sum()).backward()
    ours = (x.grad.clone(), delta.grad.clone())
    x.grad = delta.grad = None

    ref_total, ref_y = _ref_norm(x, delta, weight, 1e-6)
    ref_xq = torch.where(keep_q[pidx], ref_y * scale, 0.0)  # F.dropout's product, then autocast
    ref_xv = torch.where(keep_v[pidx], ref_y * scale, 0.0)
    ((ref_y * g16.float()).sum() + (ref_xq * gq.float()).sum() + (ref_xv * gv.float()).sum()
     + (ref_total * gsum).sum()).backward()
    torch.testing.assert_close(xq, ref_xq.half(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(xv, ref_xv.half(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ours[0], x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ours[1].float(), delta.grad.float(), rtol=2e-3, atol=2e-3)
