"""
model_export.py
=================
Exports/loads the LoRA adapter weights ALONE — not optimizer state, not
the frozen base model. Separate purpose from save_checkpoint() in
train.py, which saves full training state for resuming an interrupted
run. This is the clean, portable, deployable export.

Export directory layout (standard PEFT format):
    export_dir/
        adapter_config.json
        adapter_model.safetensors      <- a few tens of MB, not GB
"""

import os
import shutil


def export_best_weights(model, export_dir: str) -> None:
    """
    Saves ONLY the LoRA adapter weights, overwriting whatever was
    exported before. Call this every time validation produces a new
    best WER — export_dir always reflects the single current-best
    model, nothing older accumulates.
    """
    if os.path.exists(export_dir):
        # clear stale weights before writing new ones
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    model.save_pretrained(export_dir)   # PEFT writes adapter_config.json +
    # adapter_model.safetensors only —
    # the frozen base weights are NOT saved
    print(f"[export] New best weights saved -> {export_dir}")


def has_exported_weights(export_dir: str) -> bool:
    return os.path.exists(os.path.join(export_dir, "adapter_config.json"))
