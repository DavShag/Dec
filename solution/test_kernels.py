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
