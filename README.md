# Soup Kernel Challenge: fused QLoRA training on a Kaggle T4

Our `solution/` makes the fixed `soup train` run (Qwen2.5-1.5B-Instruct, NF4, LoRA r=16 on
q_proj/v_proj, 64 steps on a Kaggle T4) **1.88x faster and 6.27x lighter**, with
the same training: the same batches, the same LoRA dropout masks, the same Soup outputs, and an
adapter whose held-out loss matches the base run's to +0.001%.

Seven Triton kernels of our own (`@triton.jit`, all in `solution/`) take 11.1% of the training
time; everything around them is a handful of patches applied by `solution.apply()`.

| Kaggle T4, our run (`harness/run_task.py --repeats 2`) | Base Soup 0.77.0 | Ours | |
|---|---|---|---|
| Whole `soup train` (start to saved adapter) | 85.3 s | 45.5 s | **1.88x** |
| Peak GPU memory (nvidia-smi) | 14,345 MiB | 2,287 MiB | **6.27x** less |
| Training loop alone (64 steps) | 61.9 s | 31.0 s | |
| Held-out loss of the adapter (public rows) | 2.04963 | 2.04966 | +0.001% |
| `harness/check_rules.py` | | all 10 checks pass | |

T4 sessions differ by ~10% in speed: in an earlier session the base took 78.3 s and the same
code at the faster activation budget of §7 (1300 MiB) 37.7 s, 2.08x. The memory numbers do not
move between sessions.

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
The [tokens, 151936] logits never exist. The head runs 256 tokens at a time (their logits in one
buffer it shares with the dequantization, §3): cuBLAS writes one
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
the multiply and the add separately). q|k|v are dequantized into one buffer in one launch and
share their input: one GEMM with the bias in the cuBLAS epilogue; gate|up likewise share one
launch and one 55 MB buffer, then a GEMM each. One launch for both instead of one per weight costs
27.5 MB more at once and saves 56 launches per step: on the T4 every launch of our kernels costs
the CPU more than the GPU (Triton's launcher plus the harness's two timing events), and the GPU
was waiting for the CPU ~18% of a step before this and the one-call attention (§6), ~6% after.
Each launch writes into one scratch buffer allocated when the model is prepared, which the fused
head's logits chunk (§1) reuses: those transients are never alive at the same time and follow one
another in one CUDA stream, and as separate allocations of 5-74 MiB they cut holes among the
activations, so PyTorch's allocator reserved ~150 MiB more than it had allocated at the peak (now
~40; the nvidia-smi peak 60 MiB lower). The outputs stay fp16 (they are fp16 GEMM results either
way; the base run kept fp32 copies), and nothing is saved for backward but the packed weights.
*Format:* QLoRA (Dettmers et al. 2023).

### 4. Residual add + RMSNorm + LoRA dropout (`norm.py`, `add_rmsnorm_fwd_kernel` / `_bwd_kernel`)
One program per token: the pending residual add, the fp32 statistics and both multiplies, the fp16
output for the q|k|v (or gate|up) GEMM, and the two adapters' inputs fp16(y × keep × 1/(1−p)),
the exact value PEFT's dropout plus autocast's cast produce. The keep masks come from PyTorch's own
dropout kernel, drawn on a tensor of the same shape and dtype as in the base run and in the same
order, so they are the same masks (verified: the gradients match the HF/PEFT step to 1e-3, while a
different dropout draw moves them by ~50%). The kernel also packs the row's two masks 8 per byte,
and only those are kept for backward: 384 bytes per real token and layer instead of two bool masks
over every padded position (~5 KB per token at this task's 43% padding). The adapters' A GEMMs
(fp16, as PEFT's `lora_A` runs them under autocast) run in the same autograd function, so their
fp16 inputs are not kept either: backward writes them again from the saved input, 1/rms and masks
for the A gradients (bitwise PEFT's outputs and gradients, tested). Backward rebuilds x̂ from the
saved fp32 input and 1/rms and adds both adapters' masked gradients in fp32. The second norm of a
layer adds the fp16 o_proj output to the layer's fp32 input, which the first norm keeps anyway:
it keeps that fp16 output instead of the fp32 sum, and backward forms the sum again with the same
fp32 add (same bits, tested): 3 KB per token and layer less, nothing recomputed but an add.
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
TRL pads each batch to its longest row). Every op runs on the real tokens only. Attention is the
memory-efficient kernel SDPA picks on sm75, called directly once per layer and direction for the
whole batch, with the rows' offsets among the real tokens (its cu_seqlens interface): each row
attends causally to its own tokens, which for right padding is exactly what the padded, masked SDPA
of the base run computes for the real tokens (the same output bits, tested), and the kernel skips
everything above the diagonal. An earlier version called it once per row, with slicing, a concat
and gradient copies around each call: 4x the launches and about a quarter of a step's CPU time
on our machine (cProfile). k and v reach it repeated to all 12 query heads, but backward keeps
only the 2 distinct kv heads and repeats them again when it runs, 1 KB instead of 6 KB per token
and layer. Rows of different batches are never mixed.
*Idea:* the varlen interface of FlashAttention-2 (Dao 2023).

### 7. Adaptive recomputation (`decoder.py`)
Peak memory is set by the largest batch only. Each step estimates the activations it would keep
(from the model dimensions and its token count: 52,608 bytes per token and layer) and, if they
exceed a budget, gives up just enough of them, cheapest bytes first (our own code: Soup's
`gradient_checkpointing` stays off):
1. layers drop gate and up, 35,840 of those bytes (`RecomputedMLP`): backward rebuilds the norm's
   fp16 output from the norm's input it keeps anyway (§4) and repeats the gate|up GEMMs, about
   half the work of recomputing the layer for two thirds of its bytes; the same bits as keeping
   them (tested);
2. only if every layer has done so, leading layers are recomputed whole (`torch.utils.checkpoint`,
   with the RNG state restored, so they draw the same dropout masks).

Short batches pay nothing. The budgets we weighed, one Kaggle session (base 85.3 s /
14,345 MiB), two solution runs each:

| Activation budget | `soup train` | Training loop | Peak GPU memory | Speed / memory vs base |
|---|---|---|---|---|
| 1300 MiB | 41.5 / 44.2 s | 26.8 / 30.0 s | 2,787 MiB | 1.99x / 5.15x |
| 1000 MiB | 44.8 / 44.3 s | 30.2 / 29.9 s | 2,487 MiB | 1.92x / 5.77x |
| 800 MiB | 46.6 / 44.4 s | 32.2 / 29.7 s | 2,287 MiB | 1.88x / 6.27x |

The peak is exact (it follows the budget); the time is not: two runs of one build in one session
differ by up to 3 s, about what 800 costs over 1300 (~2.6 s by the averages here; ~3 s in another
session, 37.7 s → 40.5 s). The score is logarithmic in both ratios and relative to the best team:
with the best runs reported to us so far (~37 s, ~2.2 GB) the three budgets score within noise of
each other, and a tie goes to the lower memory. We use 800.

### 8. Compile warm-up (`warmup.py`)
Every measured run starts with an empty Triton cache, and compilation counts. Background threads
compile every kernel variant (from dtypes alone, no GPU memory, `JITFunction.warmup`) while Soup
loads the model; all flags are runtime arguments so each kernel compiles exactly once. Compiling is
not all of it: Triton builds each kernel's launcher, a small C extension (one C compiler run per
kernel signature), only at the kernel's first launch, which on the T4 was ~2.5 s inside the first
training step even after the warm-up had compiled everything. Two threads now compile and two more
build each launcher into the Triton cache as soon as its kernel is compiled (the first launch only
loads it), starting at the end of `apply()`, once Soup's libraries are imported. First step on
the T4: 3.3 s → ~0.8 s.

### 9. Startup and exit (`startup.py`)
Not kernels, but inside the measured command: costs of the start and the end of `soup train`,
removed without changing any value the run produces. `apply()` itself (the imports of Soup's
libraries it triggers) went from 7.5 s to 2.9 s on the T4:
- trl's import calls `importlib.metadata.packages_distributions()`, which reads the metadata of
  every installed distribution: 2.2 s on Kaggle's image (943 distributions). transformers made the
  same call at its own import, in the harness, moments earlier and in the same environment; trl
  gets a copy of that mapping.
- bitsandbytes wraps its CPU fallback ops in `torch.compile` at import, which imports most of
  Inductor (~1 s) for functions a CUDA run never calls: it imports with `torch.compile` deferring
  each compile to the function's first call.
- transformers imports scikit-learn and SciPy at import when they are installed (Kaggle's image
  has them), for assisted generation, the GLUE metrics and the object-detection losses; those
  modules stand in for the names transformers takes from them and import themselves when one is
  called (~0.8 s). wandb stands in the same way, answering `wandb.run` with None until it is
  imported: Soup does not report to W&B, and `run` is None until `wandb.init()`.
- Python's cyclic GC: the libraries import ~500k objects that live until the process exits. They
  import with the collector paused and are then frozen (`gc.freeze`), so later collections skip
  them; at the exit, one ordinary collection runs and the rest is frozen too, which takes the
  interpreter's shutdown from ~0.8 s to ~0.2 s on our machine. Every output is written and closed
  by then (Soup writes each rewind line under its own `open` and commits its database at once).

And in loading the model and the data:
- `datasets` fingerprints each `Dataset.map` by pickling the mapped function with dill. Soup's
  formatting function holds the tokenizer, so this pickled its 151k-entry vocabulary: ~1.7 s of a
  1.85 s map (cProfile, our machine), the tokenization itself was 0.2 s. The dataset lives in
  memory, so the fingerprint names a cache file that is never written; we pass a random one, which
  is what `datasets` does itself for a function it cannot hash.
- transformers loads each checkpoint tensor with one `.to(device, dtype)`, and for a CPU tensor
  PyTorch converts bf16 to fp16 on the CPU before the copy. We copy first and convert on the GPU:
  the same IEEE conversion, and the loaded 4-bit model is bit-identical (a digest of every packed
  weight, absmax and table). `from_pretrained`: 2.4–3.6 s → 1.1 s on our machine, 2.3 s →
  1.3–1.8 s on the T4 (its timeline, below the map's 3.0 s → 0.5–0.6 s).
- bitsandbytes rebuilt its 16-entry NF4 table, with a host-to-device copy, for each of the 196
  linears it quantizes; we build it once.

Kept from the starter kit: the example SwiGLU kernel (`swiglu.py`), now on fp16 MLP activations;
it does not count toward our share.

## How we checked it

- `python -m pytest solution -q` (32 tests, on a GPU): each kernel against plain PyTorch, forward
  and backward (NF4 bit-exact vs bitsandbytes; norm, RoPE epilogue, fused head with padding, real
  vocabulary and the fp16 loss scale; attention, one call for all rows, against the per-row SDPA
  call it replaces: the same output bits), plus whole training steps of a small 4-bit Qwen2 with
  PEFT LoRA against transformers + peft + bitsandbytes (padded / unpadded / gate|up repeated /
  whole layers recomputed / both; with and without LoRA dropout): loss within 2e-6 relative, LoRA
  gradients within 5e-3; bitwise: norm2 keeping its input's parts instead of the sum, and the
  repeated gate|up, against keeping everything (outputs and every gradient), the adapters' A
  inside the norm against PEFT's `lora_A`; the warm-up compiles exactly the kernel variants a
  training step launches and builds their launchers (the step then runs the C compiler zero
  times); the startup patches: bf16 → fp16 on the GPU is bitwise the CPU conversion (edge values
  included), an in-memory map gives the same rows without hashing its function, trl's package
  mapping is what a scan returns, the collector is paused and frozen around the imports,
  `torch.compile` waits for a function's first call, and a module standing in imports the real
  one when used.
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

- About 14.5 s of the 45.5 s are not training (23.4 s of the base's 85.3 s). ~5 s of it run before
  `apply()` can do anything: the harness imports torch and transformers, and transformers' own
  import scans the metadata of every installed package (2.3 s on Kaggle's image, the scan trl's
  import now reuses). Model loading and 4-bit quantization (~1.3 s), the dataset's chat template
  and tokenization (~0.7 s), PEFT's setup and saving are the base run's code, unchanged.
- The GEMMs (cuBLAS, ~18 TFLOPS on the T4 at these shapes; cuBLASLt was not faster) are ~67% of
  a step's GPU time. A fused W4A16 GEMM in Triton would remove the dequantization pass (~10% of a
  step) but has to match cuBLAS on Turing's tensor cores to pay off; we did not get there.
- The GPU still waits for the CPU ~6% of a step (from ~18%). Each of our ~450 kernel launches per
  step costs ~0.1 ms of CPU time on Kaggle's CPUs, Triton's launcher plus the two CUDA events the
  harness records around it from step 3 on. Dequantizing more weights per launch (q|k|v with o,
  gate|up with down) would remove another 56-112 launches per step; it needs a kernel that reads a
  table of weights and buffers shared between the autograd nodes, which we did not write.
- PyTorch's dropout kernel still runs once per adapter per layer only to draw the masks the base
  run draws; reproducing its Philox stream in Triton would remove it but is fragile.
- Attention: the memory-efficient kernel has no grouped-query mode, so k and v are repeated to all
  12 query heads for it (backward keeps them un-repeated); a varlen flash-attention kernel of our
  own with GQA would avoid that.
- Memory is a choice of the activation budget (§7): every 300 MiB less costs ~1 s of
  recomputation on this task. Beyond the tensors, PyTorch's allocator holds ~40 MiB more than the
  allocated peak (holes the remaining transients leave among the activations), and the CUDA
  context ~140 MiB.
- We tried moving the MLP activations to pinned host memory on a side stream instead of
  recomputing them (copies overlapped with the GEMMs, prefetched in backward). It saved ~2 s per
  run but raised the nvidia-smi peak by up to 2 GiB: blocks freed across streams stay reserved by
  PyTorch's caching allocator until the copies finish. We dropped it.
- Startup: the stand-ins for scikit-learn, SciPy and wandb modules (§9) defer imports only of
  modules that are installed and not yet imported; any use imports the real module. They are
  there because Kaggle's image has those packages; elsewhere they do nothing.
- The T4 sessions differ: across our runs the base took 77–90 s, and the same build's time moves by
  1–3 s between sessions and between runs of one session; the memory ratio is stable (the peak is
  deterministic).
- The fused path supports exactly this setup (Qwen2, NF4 with double quantization, LoRA on q/v,
  right padding, one CUDA device); other configurations run the original code.

## Files

| Path | What |
|---|---|
| `solution/__init__.py` | `apply()` |
| `solution/integrate.py` | patches: model setup (fp16 embedding), causal-LM forward, TRL compute_loss |
| `solution/decoder.py` | the fused decoder stack, unpadding, attention, adaptive recomputation (gate\|up first, then whole layers) |
| `solution/lce.py` | fused head + loss + metrics |
| `solution/nf4.py`, `norm.py`, `rope.py` | dequantization, add+RMSNorm+dropout, q/k/v epilogue |
| `solution/warmup.py` | background compilation of every kernel variant and its launcher |
| `solution/startup.py` | start and exit: `apply()`'s imports (trl's package mapping reused, bitsandbytes' compiles deferred, unused scikit-learn / SciPy / wandb modules standing in, GC paused and frozen), dataset map without the tokenizer pickle, bf16 → fp16 on the GPU while loading, one NF4 table |
| `solution/swiglu.py` | the starter kit's example kernel (unchanged) |
| `solution/test_kernels.py`, `solution/test_startup.py` | tests |

Everything else (`harness/`, `tests/`, `task/`, `templates/`) is the starter kit, unchanged.
