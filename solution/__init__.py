"""Our Triton kernels for Soup's QLoRA run on a Kaggle T4; `apply()` plugs them in.

The harness calls `apply()` once, before Soup builds the model and the trainer. See the README for
what each kernel does and where the ideas come from.
"""

from . import warmup
from .integrate import patch_causal_lm, patch_setup, patch_sft_trainer
from .startup import library_imports, patch_checkpoint_loading, patch_dataset_map, patch_nf4_table, patch_peft_kbit
from .swiglu import patch_qwen2_mlp


def apply():
    with library_imports():   # startup.py: no collector passes, no repeated scan, less imported
        patch_qwen2_mlp()     # the starter kit's example; kept, it does not count as ours
        patch_setup()         # fp16 tied embedding / head, fp32 embedding rows (our gather kernel)
        patch_causal_lm()     # fused decoder stack (decoder.py) + fused head and loss (lce.py)
        patch_sft_trainer()   # TRL metrics and Soup's rewind log from the fused head
        patch_dataset_map()          # startup.py: no dill pickling of the tokenizer for a fingerprint
        patch_checkpoint_loading()   # startup.py: bf16 -> fp16 on the GPU while loading
        patch_nf4_table()            # startup.py: one NF4 table, not one per quantized linear
        patch_peft_kbit()            # startup.py: keep embedding fp16 during kbit training setup
    warmup.start()   # warmup.py: compile our kernels and their launchers in the background
