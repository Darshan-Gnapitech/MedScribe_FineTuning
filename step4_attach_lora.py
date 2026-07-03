"""
Step 4: Configure and Attach LoRA Adapters
============================================
Freezes base Whisper weights and attaches LoRA to attention projections.
Trainable params must be < 2% of total.
Outputs: LoRA-attached WhisperForConditionalGeneration.
"""

from typing import List, Optional
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def attach_lora(
    model: WhisperForConditionalGeneration,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> WhisperForConditionalGeneration:
    """
    Freeze base Whisper weights and attach LoRA to attention projections.
    Covers encoder self-attn + decoder self-attn + decoder cross-attn.
    Trainable params must be < 2% of total.
    """
    if target_modules is None:
        # add "k_proj","out_proj" optionally
        target_modules = ["q_proj", "v_proj","k_proj","out_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    pct = 100 * trainable / total
    print(
        f"[LoRA] Trainable params: {trainable:,}  ({pct:.3f}% of {total:,} total)")
    assert pct < 5.0, f"Trainable share {pct:.2f}% exceeds 5% — reduce rank or target_modules."

    return model


if __name__ == "__main__":
    from transformers import WhisperForConditionalGeneration
    from step3_load_whisper import load_whisper

    _, model = load_whisper()
    print("\n-- Step 4: Attaching LoRA adapters --")
    lora_model = attach_lora(model)
    print("LoRA attachment complete.")
