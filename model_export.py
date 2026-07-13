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

    Export is atomic: the new weights are written to a temporary
    directory first, and export_dir is only ever replaced by an
    already-complete directory via os.rename. A kill mid-write leaves
    the temporary directory incomplete but export_dir itself untouched
    (either absent, or still holding the previous complete export) —
    never an empty/partial directory that attach_lora(resume_from=...)
    could mistake for "no existing weights".
    """
    tmp_dir = export_dir.rstrip("/\\") + ".tmp"
    bak_dir = export_dir.rstrip("/\\") + ".bak"

    # clean up any leftovers from a previous interrupted export
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if os.path.exists(bak_dir):
        shutil.rmtree(bak_dir)

    os.makedirs(tmp_dir, exist_ok=True)
    model.save_pretrained(tmp_dir)   # PEFT writes adapter_config.json +
    # adapter_model.safetensors only —
    # the frozen base weights are NOT saved

    # swap: move old export aside, move new export in, then discard old.
    # Each step is a fast rename (not a multi-file write), so the window
    # in which export_dir could be observed in a bad state is reduced to
    # a couple of metadata operations instead of the whole save duration.
    if os.path.exists(export_dir):
        os.rename(export_dir, bak_dir)
    os.rename(tmp_dir, export_dir)
    if os.path.exists(bak_dir):
        shutil.rmtree(bak_dir)

    print(f"[export] New best weights saved -> {export_dir}")


def has_exported_weights(export_dir: str) -> bool:
    return os.path.exists(os.path.join(export_dir, "adapter_config.json"))
