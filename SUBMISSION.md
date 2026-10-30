# Submission: TEAM_NAME

## Links
- Repository, final commit: COMMIT_LINK
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  our README): KAGGLE_LINK
- Demo video (up to 3 minutes): VIDEO_LINK
- Presentation: SLIDES_LINK

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TEAM_MEMBER | | | timaton | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes `dequantize_4bit` for all 4-bit linears, forward and backward; q\|k\|v and gate\|up in one launch | 7.4% |
| `add_rmsnorm_fwd_kernel` | `solution/norm.py` | residual add + Qwen2RMSNorm (7 fp32 kernels) + the casts to fp16 + PEFT's LoRA dropout and cast for q/v | 1.3% |
| `add_rmsnorm_bwd_kernel` | `solution/norm.py` | their backward, with both adapters' masked gradients | 1.3% |
| `qkv_rope_fwd_kernel` | `solution/rope.py` | bnb output casts, LoRA clone/add, RoPE on q and k, repeat_kv, casts for SDPA | 0.5% |
| `qkv_rope_bwd_kernel` | `solution/rope.py` | their backward | 0.3% |
| `lce_row_kernel` | `solution/lce.py` | lm_head logits + fp32 cross-entropy + TRL's entropy and accuracy + rewind's per-row losses (no logits tensor) | 1.1% |
| `embed_fp32_kernel` | `solution/integrate.py` | the fp32 embedding lookup (the weight is now stored in fp16) | <0.1% |

Ideas taken from other work, with links (we wrote the code ourselves):
- Online softmax: Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018 —
  https://arxiv.org/abs/1805.02867
- Chunked head + loss without materialized logits, gradient in place of the logits:
  Hsu et al., *Liger Kernel*, 2024 — https://arxiv.org/abs/2410.10989 ; Wijmans et al.,
  *Cut Your Losses in Large-Vocabulary Language Models*, 2024 — https://arxiv.org/abs/2411.09009
- RMSNorm backward formula: Liger Kernel (above)
- NF4 with double quantization (the format we decode): Dettmers et al., *QLoRA*, 2023 —
  https://arxiv.org/abs/2305.14314
- RoPE: Su et al., *RoFormer*, 2021 — https://arxiv.org/abs/2104.09864
- Variable-length rows instead of padding: Dao, *FlashAttention-2*, 2023 —
  https://arxiv.org/abs/2307.08691
- The hand-off of per-token losses to the rewind recorder follows Soup's own `stream_loss_chunk`
  design (Soup 0.77.0, `soup_cli/utils/stream_loss_chunk.py`); we do not enable that option.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.1 | 54.8 | speedup 1.57x |
| Peak GPU memory, MiB | 14,345 | 2,935 | base/ours 4.89x |
| Eval loss (public rows) | 2.04963 | 2.04961 | -0.001% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (the base profile), how each kernel works, results against the
base (speed, memory, quality, the budget sweep of the speed/memory trade-off), limits and next
steps — see the README.

## Limitations
See the README: ~20 s of fixed startup (imports, model loading) caps the speedup; cuBLAS GEMMs are
now ~65% of a step; PyTorch's dropout kernel still draws the masks; attention is per-row SDPA; the
fused path covers exactly this model/adapter setup and falls back to the original code otherwise.
