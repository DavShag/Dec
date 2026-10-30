"""How the kernels plug into Soup 0.77.0 (transformers 5.16.1, trl 0.29.1, peft 0.20.0).

- `SFTTrainerWrapper._setup_transformers` (Soup): once the 4-bit PEFT model exists, the tied
  embedding / lm_head weight, which `prepare_model_for_kbit_training` upcast from the fp16
  checkpoint load to fp32, is stored in fp16 again (the same values, half the memory; the head's
  matmul ran on an fp16 copy of it anyway). The embedding lookup still returns fp32, through our
  gather kernel, so the residual stream is the same as in the base run.
- `Qwen2ForCausalLM.forward`: a labelled training forward runs the decoder, then the fused head and
  loss (lce.py) instead of materializing the logits.
- `trl.SFTTrainer.compute_loss`: logs entropy, num_tokens and mean_token_accuracy by TRL's own
  formulas from the per-token values the fused head returns, and hands the per-token losses to
  Soup's rewind recorder through the attribute it reads for a step without logits
  (`soup_cli.utils.stream_loss_chunk.STEP_ATTR`). We only reuse that hand-off; Soup's
  `stream_loss_chunk` option stays off.

Every patch checks that it sees a real Qwen2 model on CUDA; anything else (Soup's unit tests use
fake modules) takes the original code path.
"""

import types

import torch
import triton
import triton.language as tl

from .decoder import PLAN_ATTR, decoder_forward, plain_training_call, prepare_decoder
from . import warmup
from .lce import fused_linear_cross_entropy

FUSED_ATTR = "_soupk_fused_head"
_LAST = {}


class _Step:
    """What a fused forward leaves for the trainer: per-token values and two counts."""

    def __init__(self, entropy, token_loss, correct, total):
        self.token_entropy, self.token_loss = entropy, token_loss
        self.correct, self.total = correct, total


@triton.jit
def embed_fp32_kernel(ids_ptr, weight_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
    """out[row] = float32(weight[ids[row]]): the fp16-stored embedding read as the fp32 rows the
    base run used."""
    row = tl.program_id(0)
    token = tl.load(ids_ptr + row).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    inside = cols < H
    values = tl.load(weight_ptr + token * H + cols, mask=inside, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * H + cols, values.to(tl.float32), mask=inside)


def _embedding_fp32_forward(self, input_ids):
    weight = self.weight
    if not input_ids.is_cuda or weight.dtype != torch.float16:
        return torch.nn.functional.embedding(input_ids, weight).float()
    ids = input_ids.reshape(-1).contiguous()
    hidden = weight.shape[1]
    out = torch.empty((ids.numel(), hidden), dtype=torch.float32, device=ids.device)
    if ids.numel():
        embed_fp32_kernel[(ids.numel(),)](ids, weight, out, hidden,
                                          BLOCK_H=triton.next_power_of_2(hidden), num_warps=4)
    return out.view(*input_ids.shape, hidden)


def _causal_lm(model):
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def prepare_model(model):
    """Repack the frozen head/embedding to fp16 and mark the model for the fused head."""
    if not isinstance(model, torch.nn.Module):
        return False
    from transformers.models.qwen2 import modeling_qwen2

    causal = _causal_lm(model)
    if not isinstance(causal, modeling_qwen2.Qwen2ForCausalLM):
        return False
    embed, head = causal.get_input_embeddings(), causal.get_output_embeddings()
    if head is None or head.weight.device.type != "cuda" or head.weight.requires_grad:
        return False
    if embed.weight.requires_grad:
        return False
    if embed.weight.dtype == torch.float32:
        embed.weight.data = embed.weight.data.to(torch.float16).contiguous()
        embed.forward = types.MethodType(_embedding_fp32_forward, embed)
    if head.weight.dtype == torch.float32:  # not tied: the head alone
        head.weight.data = head.weight.data.to(torch.float16).contiguous()
    setattr(causal, FUSED_ATTR, True)
    prepare_decoder(causal.model)
    return True


def patch_setup():
    from soup_cli.trainer import sft

    original = sft.SFTTrainerWrapper._setup_transformers
    if getattr(original, "_soupk", False):
        return

    def _setup_transformers(self, *args, **kwargs):
        warmup.start()  # compile our kernels in the background while the model loads
        result = original(self, *args, **kwargs)
        prepare_model(getattr(self, "model", None))
        return result

    _setup_transformers._soupk = True
    sft.SFTTrainerWrapper._setup_transformers = _setup_transformers


def patch_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2ForCausalLM.forward
    if getattr(original, "_soupk", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if labels is None or not self.training or not getattr(self, FUSED_ATTR, False):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        plan = getattr(self.model, PLAN_ATTR, None)
        fused = None
        if plan is not None and plain_training_call(input_ids, inputs_embeds, past_key_values,
                                                    use_cache, kwargs):
            fused = decoder_forward(self.model, plan, input_ids, attention_mask, position_ids)
        if fused is not None:
            hidden, unpad = fused
            rows, extra = unpad.pidx, {}
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
            hidden, rows = outputs.last_hidden_state, None
            extra = {"past_key_values": outputs.past_key_values,
                     "hidden_states": outputs.hidden_states, "attentions": outputs.attentions}
        loss, entropy, token_loss, correct, total = fused_linear_cross_entropy(
            hidden, self.lm_head.weight, labels, attention_mask,
            kwargs.get("num_items_in_batch"), rows=rows)
        _LAST["step"] = _Step(entropy, token_loss, correct, total)
        return CausalLMOutputWithPast(loss=loss, logits=None, **extra)

    forward._soupk = True
    modeling_qwen2.Qwen2ForCausalLM.forward = forward


def patch_sft_trainer():
    import trl
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    from transformers import Trainer

    cls = trl.SFTTrainer
    original = cls.compute_loss
    if getattr(original, "_soupk", False):
        return

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal = _causal_lm(getattr(self, "model", None))
        if (not isinstance(inputs, dict) or "labels" not in inputs or "shift_labels" in inputs
                or not getattr(causal, FUSED_ATTR, False) or not getattr(model, "training", False)
                or self.args.use_liger_kernel or getattr(self, "num_virtual_tokens", 0) > 0):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        _LAST.clear()
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        step = _LAST.pop("step", None)
        if step is None:
            raise RuntimeError("soupk: the fused head did not run for a fused model")
        gather = self.accelerator.gather_for_metrics
        with torch.no_grad():  # TRL's formulas, from the fused head's per-token values
            mask = inputs["attention_mask"]
            entropy = torch.sum(step.token_entropy * mask) / mask.sum()
            self._metrics[mode]["entropy"].append(gather(entropy).mean().item())
            if mode == "train":
                self._total_train_tokens += gather(mask.sum()).sum().item()
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            total = gather(step.total).sum()
            accuracy = (gather(step.correct).sum() / total).item() if total > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        if self.aux_loss_enabled:
            self._metrics[mode]["aux_loss"].append(gather(outputs.aux_loss).mean().item())
        setattr(self, STEP_ATTR, step)  # Soup's rewind recorder reads step.token_loss
        return (loss, outputs) if return_outputs else loss

    compute_loss._soupk = True
    cls.compute_loss = compute_loss
