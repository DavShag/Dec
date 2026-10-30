# Soup Kernel Challenge: fused QLoRA training on a Kaggle T4

Our `solution/` makes the fixed `soup train` run (Qwen2.5-1.5B-Instruct, NF4, LoRA r=16 on
q_proj/v_proj, 64 steps on a Kaggle T4) **RESULT_SPEEDUP faster and RESULT_MEMORY lighter**, with
the same training: the same batches, the same LoRA dropout masks, the same Soup outputs, and an
adapter whose held-out loss matches the base run's to RESULT_QUALITY.

Seven Triton kernels of our own (`@triton.jit`, all in `solution/`) take RESULT_SHARE of the training
time; everything around them is a handful of patches applied by `solution.apply()`.

| Kaggle T4, our run (`harness/run_task.py --repeats 2`) | Base Soup 0.77.0 | Ours | |
|---|---|---|---|
| Whole `soup train` (start to saved adapter) | RESULT_BASE_TIME s | RESULT_OUR_TIME s | **RESULT_SPEEDUP** |
| Peak GPU memory (nvidia-smi) | RESULT_BASE_MEM MiB | RESULT_OUR_MEM MiB | **RESULT_MEMORY** less |
| Training loop alone (64 steps) | RESULT_BASE_LOOP s | RESULT_OUR_LOOP s | |
| Held-out loss of the adapter (public rows) | RESULT_BASE_LOSS | RESULT_OUR_LOSS | RESULT_QUALITY |
| `harness/check_rules.py` | | all 10 checks pass | |

Links: Kaggle notebook KAGGLE_LINK · final commit COMMIT_LINK

## Where the time and memory went

A `torch.profiler` table of the base run on the T4 (our measurement, `run_task.py --profile`):

| Base step on a T4 (911 ms GPU) | share | what it is |
|---|---|---|
| GEMMs (cuBLAS fp16) | 40% | the 4-bit linears after dequantization, the 152k-vocab head |
| copies and dtype casts | 18% | the residual stream is fp32 (Soup keeps embeddings and norms in fp32), so every 4-bit linear casts its input to fp16 and its output back to fp32 |
| other element-wise ops | 12% | RMSNorm in seven fp32 kernels, RoPE, LoRA dropout/scale/add, residual adds |
| softmax / cross-entropy | 7% | over [batch, seq, 151936] fp32 logits, again for entropy, accuracy and the rewind log |
| attention | 6.5% | padded, masked SDPA |
| NF4 dequantization | 5% | bitsandbytes, forward and backward |

Peak memory (13.4 GiB allocated) was mostly the logits (fp16, then fp32 copies for the loss, for
TRL's accuracy, for accelerate's fp16-output conversion, plus the fp32 log-softmax kept for
backward), saved fp32 activations of 28 layers (no gradient checkpointing in this config), the
fp32 tied embedding / lm_head (890 MiB) and its fp16 copy for the head's matmul.

## What we built

Every kernel is applied on top of the unchanged model: the 4-bit weights, the LoRA modules, the
trainer and its settings are Soup's own. `apply()` patches three places (Soup's model setup, the
Qwen2 causal-LM forward, TRL's `compute_loss`) and checks at run time that it sees exactly the
supported structure; anything else takes the original code path.

### 1. Fused head + loss + metrics (`lce.py`, kernel `lce_row_kernel`)
The [tokens, 151936] logits never exist. The head runs 256 tokens at a time: cuBLAS writes one
chunk of fp16 logits, then our kernel streams each row once with an online softmax
(running max, sum of exponentials) and, in the same pass, the Shannon entropy
H = lse − Σp·z (the weighted sum is rescaled together with the softmax denominator), the argmax
and the label's logit. A second pass over the row writes (softmax − onehot) × 1024 back over the
logits in fp16, and one more GEMM gives the hidden-state gradient; the 1/num_items and the fp16
GradScaler factor are applied in backward in fp32. The head is frozen (tied embedding), so no
weight gradient is formed, and only the labelled (answer) tokens go through the backward GEMM.
TRL's `entropy` (every non-padding token), `mean_token_accuracy` and `num_tokens`, and Soup's
per-row rewind losses are computed by their own formulas from the per-token values; Soup's rewind
recorder already reads per-token losses for a step without logits, so we hand them over that way
(we do not turn on Soup's `stream_loss_chunk` option).
*Ideas:* online softmax — Milakov & Gimelshein 2018; chunked head without logits — Cut
Cross-Entropy (Wijmans et al. 2024) and Liger Kernel's FusedLinearCrossEntropy (Hsu et al. 2024);
the gradient written in place of the logits — Liger Kernel.

### 2. fp16 tied embedding (`integrate.py`, kernel `embed_fp32_kernel`)
`prepare_model_for_kbit_training` upcast the fp16-loaded embedding / lm_head to fp32. We store it
as fp16 again (the same values: it was loaded as fp16) and look rows up with a gather kernel that
returns fp32, so the residual stream sees exactly the base run's values. −890 MiB, and the head
reads the weight without an fp16 copy.

### 3. NF4 dequantization (`nf4.py`, kernel `nf4_dequant_kernel`)
bitsandbytes' layout decoded in one pass: two 4-bit codes per byte (high nibble first), a weight
is code[nibble] × (code2[absmax_q] × absmax2 + offset) with an absmax per 64 weights and an
absmax2 per 256 absmax values (double quantization). Bit-identical to `bnb.dequantize_4bit`
(checked in the tests; the kernel is compiled without FMA contraction because bitsandbytes rounds
the multiply and the add separately). q|k|v and gate|up are dequantized into one buffer in one
launch and share their input: q|k|v becomes one GEMM with the bias in the cuBLAS epilogue. The
outputs stay fp16 (they are fp16 GEMM results either way; the base run kept fp32 copies), and
nothing is saved for backward but the packed weights. *Format:* QLoRA (Dettmers et al. 2023).

### 4. Residual add + RMSNorm + LoRA dropout (`norm.py`, `add_rmsnorm_fwd_kernel` / `_bwd_kernel`)
One program per token: the pending residual add, the fp32 statistics and both multiplies, the fp16
output for the q|k|v (or gate|up) GEMM, and the two adapters' inputs fp16(y × keep × 1/(1−p)),
the exact value PEFT's dropout plus autocast's cast produce. The keep masks come from PyTorch's own
dropout kernel, drawn on a tensor of the same shape and dtype as in the base run and in the same
order, so they are the same masks (verified: the gradients match the HF/PEFT step to 1e-3, while a
different dropout draw moves them by ~50%). Backward rebuilds x̂ from the saved fp32 input and
1/rms and adds both adapters' masked gradients in fp32.
*Derivation:* Liger Kernel's RMSNorm backward (Hsu et al. 2024).

### 5. q/k/v epilogue (`rope.py`, `qkv_rope_fwd_kernel` / `_bwd_kernel`)
Per token: the q|k|v GEMM row plus the LoRA outputs (fp32), RoPE on q and k in fp32 at the token's
position, k/v repeated to all 12 query heads (SDPA's memory-efficient kernel has no GQA mode on
sm75), fp16 out. Replaces three output casts, two clones and adds, RoPE's slices/cat/neg/mul/add
for q and k, repeat_kv and three casts. Backward: inverse rotation, the sum over repeated heads,
one fp16 gradient row whose q and v parts are also the LoRA outputs' gradients.
*RoPE:* Su et al. 2021, in transformers' rotate_half form.

### 6. No padding (`decoder.py`)
43% of the positions in this task's batches are padding (we measured it on the training rows:
TRL pads each batch to its longest row). Every op except attention runs on the real tokens only;
attention runs per row with `is_causal`, which for right padding is exactly what the padded,
masked SDPA of the base run computes for the real tokens, and lets the memory-efficient kernel
skip everything above the diagonal. Rows of different batches are never mixed.
*Idea:* the varlen interface of FlashAttention-2 (Dao 2023).

### 7. Adaptive recomputation (`decoder.py`)
Peak memory is set by the largest batch only. Each step estimates the activations it would keep
(from the model dimensions and its token count) and, if they exceed a budget, recomputes just
enough of the first layers in backward (`torch.utils.checkpoint`, our own code: Soup's
`gradient_checkpointing` stays off). The RNG state is restored for the recomputation, so the
recomputed layers draw the same dropout masks. Short batches pay nothing.

### 8. Compile warm-up (`warmup.py`)
Every measured run starts with an empty Triton cache, and compilation counts. A background thread
compiles every kernel variant (from dtypes alone, no GPU memory, `JITFunction.warmup`) while Soup
loads the model; all flags are runtime arguments so each kernel compiles exactly once.

Kept from the starter kit: the example SwiGLU kernel (`swiglu.py`), now on fp16 MLP activations;
it does not count toward our share.

## How we checked it

- `python -m pytest solution -q` (17 tests, on a GPU): each kernel against plain PyTorch, forward
  and backward (NF4 bit-exact vs bitsandbytes; norm, RoPE epilogue, fused head with padding, real
  vocabulary and the fp16 loss scale), plus whole training steps of a small 4-bit Qwen2 with PEFT
  LoRA against transformers + peft + bitsandbytes (padded / unpadded / recomputed): loss within
  2e-6 relative, LoRA gradients within 5e-3.
- `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`: Soup's 79 tests with our patches.
- `harness/run_task.py` + `harness/check_rules.py` on a Kaggle T4: all 10 checks.

## How to run

On Kaggle: open KAGGLE_LINK (the starter kit's notebook with `REPO_URL` set to this repository and
`COMMIT` to the final commit), accelerator **GPU T4 x2**, internet on, **Run All** (about 10
minutes): base and solution are trained and measured, both adapters scored, and the checks run.

Locally on a Linux machine with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2 --profile
python harness/check_rules.py
```

## Limitations and what we would do next

- About 20 s of the ~RESULT_OUR_TIME s are not training: Python imports, model loading and
  quantization, the trainer's setup, saving. They are the same for the base run, so they cap the
  speedup; we did not touch Soup's loading path.
- The GEMMs (cuBLAS, ~26 TFLOPS on the T4) are now ~65% of a step. A fused W4A16 GEMM in Triton
  would remove the dequantization pass (~10% of a step) but has to match cuBLAS on Turing's
  tensor cores to pay off; we did not get there.
- PyTorch's dropout kernel still runs once per adapter per layer only to draw the masks the base
  run draws; reproducing its Philox stream in Triton would remove it but is fragile.
- Attention is PyTorch's memory-efficient SDPA per row; a varlen flash-attention kernel with GQA
  would avoid repeating k/v and the per-row launches.
- The fused path supports exactly this setup (Qwen2, NF4 with double quantization, LoRA on q/v,
  right padding, CUDA); other configurations run the original code.

## Files

| Path | What |
|---|---|
| `solution/__init__.py` | `apply()` |
| `solution/integrate.py` | patches: model setup (fp16 embedding), causal-LM forward, TRL compute_loss |
| `solution/decoder.py` | the fused decoder stack, unpadding, adaptive recomputation |
| `solution/lce.py` | fused head + loss + metrics |
| `solution/nf4.py`, `norm.py`, `rope.py` | dequantization, add+RMSNorm+dropout, q/k/v epilogue |
| `solution/warmup.py` | background compilation |
| `solution/swiglu.py` | the starter kit's example kernel (unchanged) |
| `solution/test_kernels.py` | tests |

Everything else (`harness/`, `tests/`, `task/`, `templates/`) is the starter kit, unchanged.
