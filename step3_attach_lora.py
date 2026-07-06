"""
Step 4: Configure and Attach LoRA Adapters
============================================
Freezes base Whisper weights and attaches LoRA to attention projections.
Trainable params must be < 2% of total.
Outputs: LoRA-attached WhisperForConditionalGeneration.
"""

from typing import List, Optional
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
import os

def attach_lora(
    model: WhisperForConditionalGeneration,
    r: int = 16,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
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

    has_existing = resume_from and os.path.exists(
        os.path.join(resume_from, "adapter_config.json"))

    if has_existing:
        print(
            f"[LoRA] Found existing weights at {resume_from} — loading instead of fresh init")
        # is_trainable=True is NOT the default here — PEFT loads adapters
        # frozen/inference-only unless you explicitly ask otherwise. Miss
        # this and "resumed" training silently does nothing: zero gradient
        # flow, loss looks fine, nothing actually updates.
        model = PeftModel.from_pretrained(
            model, resume_from, is_trainable=True)
    else:
        print("[LoRA] No existing weights found — starting from fresh LoRA init")
        lora_config = LoraConfig(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none",
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

    _, model = load_whisper()
    print("\n-- Step 4: Attaching LoRA adapters --")
    lora_model = attach_lora(model)
    print("LoRA attachment complete.")
