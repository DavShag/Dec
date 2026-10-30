"""Fused linear cross-entropy for the output head (lm_head + loss + TRL's metrics + rewind losses).

Base Soup builds the [batch, seq, 151936] logits, upcasts them to fp32 for the loss, keeps the
log-softmax for backward, copies them again for TRL's accuracy, and the accelerate fp16 wrapper
converts them to fp32 once more: several GiB at the peak, and about 7% of the step time.

Here the head runs a chunk of tokens at a time. cuBLAS writes one chunk of fp16 logits; our Triton
kernel streams each row once with an online softmax (Milakov & Gimelshein 2018) and gets, in the
same pass, the log-sum-exp, the Shannon entropy H = lse - sum(p * z) (the weighted sum is rescaled
like the softmax denominator), the argmax and the label's logit. A second pass over the same row
writes (softmax - onehot) back over the logits, so the gradient GEMM reads it in place and no fp32
copy ever exists (the in-place idea is from Liger Kernel, Hsu et al. 2024; chunking the head from
Cut Cross-Entropy, Wijmans et al. 2024, and Soup's own `stream_loss_chunk` design). The head is
frozen, so only the gradient on the hidden states is formed.

Two task-specific choices:
- rows are processed in the order: tokens with a label (loss + gradient), then the other real tokens
  (entropy only: TRL averages entropy over every attention-mask position, prompt included).
  Padding positions are skipped entirely;
- the gradient is formed as (softmax - onehot) * 1024 in fp16, and the 1/num_items and the loss
  scale of the fp16 GradScaler are applied in backward, in fp32.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK = 256        # logits rows alive at once: 256 x 151936 fp16 = 74 MiB
BLOCK_V = 2048
GRAD_SCALE = 1024.0


@triton.jit
def lce_row_kernel(logits_ptr, stride, target_ptr, ent_ptr, loss_ptr, hit_ptr, V, grad_scale,
                   WITH_GRAD: tl.constexpr, BLOCK_V: tl.constexpr):
    """One row of logits: entropy, cross-entropy, argmax hit; with WITH_GRAD also overwrite the
    row with (softmax - onehot) * grad_scale in fp16."""
    row = tl.program_id(0)
    ptr = logits_ptr + row.to(tl.int64) * stride
    target = tl.load(target_ptr + row).to(tl.int32)
    offs = tl.arange(0, BLOCK_V)

    run_max = float("-inf")
    run_sum = 0.0      # sum exp(z - run_max)
    run_dot = 0.0      # sum exp(z - run_max) * z
    best_val = float("-inf")
    best_idx = 0
    for start in range(0, V, BLOCK_V):
        cols = start + offs
        inside = cols < V
        z = tl.load(ptr + cols, mask=inside, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(z, axis=0)
        tile_arg = tl.argmax(z, axis=0)
        best_idx = tl.where(tile_max > best_val, start + tile_arg, best_idx)
        best_val = tl.maximum(best_val, tile_max)
        new_max = tl.maximum(run_max, tile_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.where(inside, tl.exp(z - new_max), 0.0)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(e * tl.where(inside, z, 0.0), axis=0)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    tl.store(ent_ptr + row, lse - run_dot / run_sum)
    labelled = target >= 0
    z_target = tl.load(ptr + tl.maximum(target, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - z_target, 0.0))
    tl.store(hit_ptr + row, (labelled & (best_idx == target)).to(tl.int32))

    if WITH_GRAD:
        for start in range(0, V, BLOCK_V):
            cols = start + offs
            inside = cols < V
            z = tl.load(ptr + cols, mask=inside, other=0.0).to(tl.float32)
            grad = tl.exp(z - lse) - tl.where(cols == target, 1.0, 0.0)
            grad = tl.where(labelled, grad * grad_scale, 0.0)
            tl.store(ptr + cols, grad.to(tl.float16), mask=inside)


class FusedLinearCrossEntropy(torch.autograd.Function):
    """loss = sum of the labelled tokens' cross-entropy / denominator, as transformers'
    ForCausalLMLoss computes it, plus per-token entropy, loss and argmax hits for the metrics."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)  # dtypes are explicit below; autocast rejects out=
    def forward(ctx, hidden, weight, targets, real, denominator):
        shape, dtype = hidden.shape, hidden.dtype
        flat = hidden.reshape(-1, shape[-1])
        targets = targets.reshape(-1)
        n, vocab = flat.shape[0], weight.shape[0]
        labelled = targets != IGNORE_INDEX
        real = labelled | (real.reshape(-1) != 0)
        with_grad = ctx.needs_input_grad[0]

        entropy = torch.zeros(n, dtype=torch.float32, device=flat.device)
        token_loss = torch.zeros(n, dtype=torch.float32, device=flat.device)
        hits = torch.zeros(n, dtype=torch.int32, device=flat.device)
        grad_rows = torch.zeros_like(flat, dtype=torch.float16) if with_grad else None
        buffer = torch.empty((min(CHUNK, n), vocab), dtype=torch.float16, device=flat.device)
        weight_t = weight.t()

        for rows, grad in ((labelled.nonzero().squeeze(1), with_grad),
                           ((real & ~labelled).nonzero().squeeze(1), False)):
            for start in range(0, rows.numel(), CHUNK):
                index = rows[start:start + CHUNK]
                count = index.numel()
                logits = buffer[:count]
                torch.mm(flat.index_select(0, index).to(torch.float16), weight_t, out=logits)
                part_ent = torch.empty(count, dtype=torch.float32, device=flat.device)
                part_loss = torch.empty_like(part_ent)
                part_hit = torch.empty(count, dtype=torch.int32, device=flat.device)
                lce_row_kernel[(count,)](logits, logits.stride(0), targets.index_select(0, index),
                                         part_ent, part_loss, part_hit, vocab, GRAD_SCALE,
                                         WITH_GRAD=grad, BLOCK_V=BLOCK_V, num_warps=8)
                entropy.index_copy_(0, index, part_ent)
                token_loss.index_copy_(0, index, part_loss)
                hits.index_copy_(0, index, part_hit)
                if grad:
                    grad_rows.index_copy_(0, index, torch.mm(logits, weight))
        del buffer

        if denominator is None:
            denominator = labelled.sum()
        loss = token_loss.sum() / denominator
        ctx.shape, ctx.dtype = shape, dtype
        ctx.denominator = denominator
        ctx.save_for_backward(grad_rows if with_grad else torch.empty(0, device=flat.device))
        entropy, token_loss = entropy.view(shape[:-1]), token_loss.view(shape[:-1])
        correct, total = hits.sum(), labelled.sum()
        ctx.mark_non_differentiable(entropy, token_loss, correct, total)
        return loss, entropy, token_loss, correct, total

    @staticmethod
    def backward(ctx, grad_loss, *unused):
        (grad_rows,) = ctx.saved_tensors
        if grad_rows.numel() == 0:
            return None, None, None, None, None
        scale = grad_loss.float() / (ctx.denominator * GRAD_SCALE)
        grad = (grad_rows.float() * scale).view(ctx.shape).to(ctx.dtype)
        return grad, None, None, None, None


def fused_linear_cross_entropy(hidden, weight, labels, attention_mask=None, num_items_in_batch=None,
                               rows=None):
    """ForCausalLMLoss(hidden @ weight.T, labels) without the logits.

    Returns (loss, entropy [B, T], token_loss [B, T], correct, labelled): entropy of every position,
    the cross-entropy of position t against labels[t + 1] (0 where there is none), and the argmax
    hits and count of the labelled positions, which is what TRL and Soup's rewind log read.
    With `rows`, hidden holds only those flattened positions ([n, H], the real tokens); the
    per-position outputs are still [B, T], zero elsewhere.
    """
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)
    targets = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    denominator = num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(hidden.device)
    if rows is None:
        real = attention_mask if attention_mask is not None else torch.ones_like(labels)
        return FusedLinearCrossEntropy.apply(hidden, weight, targets.contiguous(), real,
                                             denominator)
    targets = targets.reshape(-1).index_select(0, rows)
    loss, entropy, token_loss, correct, total = FusedLinearCrossEntropy.apply(
        hidden, weight, targets, torch.ones_like(targets), denominator)
    full = []
    for values in (entropy, token_loss):
        out = torch.zeros(labels.numel(), dtype=values.dtype, device=values.device)
        full.append(out.index_copy_(0, rows, values.reshape(-1)).view(labels.shape))
    return loss, full[0], full[1], correct, total
