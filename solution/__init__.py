"""Our Triton kernels for Soup's QLoRA run on a Kaggle T4; `apply()` plugs them in.

The harness calls `apply()` once, before Soup builds the model and the trainer. See the README for
what each kernel does and where the ideas come from.
"""

from .integrate import patch_causal_lm, patch_setup, patch_sft_trainer
from .swiglu import patch_qwen2_mlp


def apply():
    patch_qwen2_mlp()     # the starter kit's example; kept, it does not count as ours
    patch_setup()         # fp16 tied embedding / head, fp32 embedding rows (our gather kernel)
    patch_causal_lm()     # fused decoder stack (decoder.py) + fused head and loss (lce.py)
    patch_sft_trainer()   # TRL metrics and Soup's rewind log from the fused head
