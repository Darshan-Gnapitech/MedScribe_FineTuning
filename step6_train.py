# train.py
# Steps 7-10 unified training loop.
# Receives config as a plain dict — dynamic config.py attached later.
#
# Expected config keys (mirrors WhisperFinetuneConfig fields):
#   model_name_or_path, per_device_train_batch_size, per_device_eval_batch_size,
#   gradient_accumulation_steps, learning_rate, warmup_steps, max_steps,
#   fp16, bf16, gradient_checkpointing, freeze_encoder,
#   lora_r, lora_alpha, lora_dropout, lora_target_modules,
#   eval_steps, save_steps, logging_steps, output_dir,
#   max_input_length_seconds, num_workers

import os
import json
import time
import torch
import numpy as np
from dataclasses import dataclass
import random
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from transformers import (
    GenerationConfig,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel
from jiwer import wer as compute_wer
from model_export import export_best_weights

# ============================================================================
# 2. Early stopping state
# ============================================================================


def limit_gpu_memory(max_gb: float, device_idx: int = 0):
    if not torch.cuda.is_available():
        return
    total_gb = torch.cuda.get_device_properties(
        device_idx).total_memory / (1024**3)
    fraction = min(max_gb / total_gb, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)
    print(f"[train] GPU memory capped to {max_gb:.1f} GB "
          f"({fraction:.1%} of {total_gb:.1f} GB total) on device {device_idx}")
    
@dataclass
class EarlyStoppingState:
    patience: int = 5
    min_delta: float = 1e-4
    best_wer: float = float("inf")
    counter: int = 0
    should_stop: bool = False
    best_step: int = 0

    def step(self, current_wer: float, current_step: int) -> bool:
        improved = current_wer < (self.best_wer - self.min_delta)
        if improved:
            self.best_wer = current_wer
            self.counter = 0
            self.best_step = current_step
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# ============================================================================
# 3. Checkpoint helpers
# ============================================================================

def save_checkpoint(
    model: PeftModel,
    optimizer: AdamW,
    scaler: GradScaler,
    step: int,
    metrics: dict,
    save_dir: str,
    tag: str,
) -> None:
    path = os.path.join(save_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "metrics": metrics,
    }, os.path.join(path, "trainer_state.pt"))
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump({"step": step, **metrics}, f, indent=2)


def maybe_resume(model, optimizer, scaler, output_dir, enabled: bool = True):
    if not enabled:
        print("[resume] RESUME_FROM_CHECKPOINT=false — starting fresh, ignoring any existing checkpoint")
        return 0

    state_path = os.path.join(output_dir, "latest", "trainer_state.pt")

    if not os.path.exists(state_path):
        print(f"[resume] No checkpoint found at {state_path} — starting fresh")
        return 0

    # Reload LoRA weights into the existing model in-place
    latest_dir = os.path.join(output_dir, "latest")
    model.load_adapter(latest_dir, adapter_name="default")

    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    print(f"[resume] Resuming from step {state['step']} | "
          f"val_wer {state['metrics'].get('val_wer', '?')}")
    return state["step"]


# ============================================================================
# 4. Validation
# ============================================================================

def run_validation(
    model: PeftModel,
    val_loader: DataLoader,
    processor: WhisperProcessor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """
    Compute validation loss and WER over the full validation set.

    Returns:
        val_loss : float
        val_wer  : float
    """
    model.eval()
    losses, preds, refs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            raw_texts = batch["raw_texts"]

            # Validation loss — teacher forced
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_features=features,
                    labels=labels,
                    return_dict=True,
                )
            losses.append(outputs.loss.item())

            # WER — free autoregressive generation (no teacher forcing)
            model.config.use_cache = True
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                predicted_ids = model.generate(
                    input_features=features,
                    max_new_tokens=200,
                )
            model.config.use_cache = False
            decoded = processor.batch_decode(
                predicted_ids, skip_special_tokens=True)
            preds.extend(decoded)
            refs.extend(raw_texts)

    val_loss = float(np.mean(losses))
    val_wer = compute_wer(refs, preds)
    model.train()

    return val_loss, val_wer


# ============================================================================
# 5. Main training loop — Steps 7 → 10
# ============================================================================

def train(
    model,
    processor,
    train_dataset,
    val_dataset,
    training_config,      # MedicalWhisperTrainingConfig
    data_collator,
) -> dict:
    """
    Returns:
        {
            "best_step"     : int,
            "best_wer"      : float,
            "total_steps"   : int,
            "output_dir"    : str,
        }
    """
    import math
    model.generation_config = GenerationConfig.from_pretrained(training_config.model_name_or_path)

    # ── Device + precision setup ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit_gpu_memory(training_config.max_vram_gb, device_idx=0)
    use_amp = (
        training_config.bf16 or training_config.fp16) and torch.cuda.is_available()

    amp_dtype = (
        torch.bfloat16
        if training_config.bf16
        else torch.float16
    )

    print(f"[train] device={device}  amp={use_amp}  dtype={amp_dtype}")
    # ── Contract assertions — catch Team A integration bugs early ────────────
    assert len(train_dataset) > 0, "train_data is empty"
    assert len(val_dataset) > 0, "val_data is empty"

    sample = train_dataset[0]
    assert "input_features" in sample, "Missing input_features — Team A preprocessing issue"
    assert "labels" in sample, "Missing labels — Team A tokenization issue"
    assert "raw_text" in sample, "Missing raw_text — needed for WER in Step 10"

    mel_shape = torch.tensor(sample["input_features"]).shape

    expected_bins = processor.feature_extractor.feature_size

    assert mel_shape == torch.Size([expected_bins, 3000]), (
    f"Expected mel shape ({expected_bins}, 3000), got {mel_shape}")
    labels = torch.tensor(sample["labels"])
    print(
        processor.tokenizer.decode(
            labels.tolist(),
            skip_special_tokens=True,
        )
    )    
    assert processor.tokenizer.eos_token_id in sample["labels"], (
        "No EOS token in labels — decoder will never learn to stop"
    )
    print(
        f"[contract] ✓ EOS present, {sample['labels'].count(-100)} padded positions")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total
    assert ratio < 0.06, (
        f"Trainable param ratio {ratio:.2%} exceeds 6% — LoRA may not be frozen correctly"
    )

    print(f"[contract] ✓ train_data: {len(train_dataset)} samples")
    print(f"[contract] ✓ val_data:   {len(val_dataset)} samples")
    print(f"[contract] ✓ mel shape:  {mel_shape}")
    print(f"[contract] ✓ trainable:  {trainable:,} / {total:,} ({ratio:.2%})")

    # ── Model ────────────────────────────────────────────────────────────────
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=pin,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=pin,
        collate_fn=data_collator,
    
    )
    subset_size = min(500, len(val_dataset))
    subset_indices = random.Random(42).sample(range(len(val_dataset)), subset_size)
    val_subset = torch.utils.data.Subset(val_dataset, subset_indices)

    val_subset_loader = DataLoader(
        val_subset,
        batch_size=training_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=pin,
        collate_fn=data_collator,
    )
    print(f"[train] Using {subset_size}-row random subset for interim validation "
          f"(full {len(val_dataset):,}-row set reserved for final validation)")
    # ── Optimizer — LoRA params only ─────────────────────────────────────────

    updates_per_epoch = math.ceil(
        len(train_loader)
        / training_config.gradient_accumulation_steps
    )

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        lora_params,
        lr=training_config.learning_rate,
        betas=(
            training_config.adam_beta1,
            training_config.adam_beta2,
        ),
        weight_decay=training_config.weight_decay,
        eps=1e-8,
    )
    max_steps = updates_per_epoch * training_config.num_train_epochs
    training_config.eval_steps=max(1, max_steps // 10)
    optimizer_steps = max_steps
    assert training_config.eval_steps <= max_steps, (
        f"eval_steps ({training_config.eval_steps}) > max_steps ({max_steps}) — "
        f"validation will never run. Lower eval_steps or raise num_train_epochs."
    )
    # ── LR scheduler — linear warmup then linear decay ───────────────────────
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=optimizer_steps,
    )

    # ── Mixed precision scaler ───────────────────────────────────────────────
    # Only active for fp16 — bf16 doesn't need loss scaling
    try:
        scaler = GradScaler(
            device.type,
            enabled=training_config.fp16 and device.type == "cuda",
        )
    except Exception as e:
        scaler = GradScaler(
            enabled=training_config.fp16 and device.type == "cuda",
        )

    # ── Resume if checkpoint exists ──────────────────────────────────────────
    output_dir = training_config.output_dir
    resume_enabled = os.getenv(
        "RESUME_FROM_CHECKPOINT", "true").strip().lower() in ("1", "true", "yes")
    print(f"[train] RESUME_FROM_CHECKPOINT = {resume_enabled}")
    start_step = maybe_resume(model, optimizer, scaler,output_dir, enabled=resume_enabled)
    accum_steps = training_config.gradient_accumulation_steps
    early_stop = EarlyStoppingState(
        patience=training_config.early_stopping_patience)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    # We cycle through the dataloader until max_steps is reached
    # rather than training for a fixed number of epochs, which is
    # standard practice for large datasets with step-based configs.
    # ── Training state ────────────────────────────────────────────────────────
    t0 = time.time()
    global_step = start_step
    batch_idx = 0
    running_loss = 0.0
    grad_norm = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    print(f"\n[train] Starting from step {start_step} → {max_steps}")
    print(f"[train] effective batch = "
          f"{training_config.per_device_train_batch_size} × {accum_steps} = "
          f"{training_config.per_device_train_batch_size * accum_steps}\n")
    last_log_time = time.time()
    last_log_step = start_step

    # FIX — everything inside the for loop, remove the stray comment
    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break

            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_features=features,
                    labels=labels,
                    return_dict=True,
                )

            loss = outputs.loss
            running_loss += loss.item()
            batch_idx += 1
            scaler.scale(loss / accum_steps).backward()

            if batch_idx % accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    lora_params, max_norm=training_config.max_grad_norm
                ).item()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
       
     

                # ── Logging ──────────────────────────────────────────────────────
                if global_step % training_config.logging_steps == 0:
                    now = time.time()
                    steps_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-6)
                    last_log_time, last_log_step = now, global_step
                    remaining = (max_steps -
                                 global_step) / max(steps_sec, 1e-6)
                    avg_loss = running_loss / training_config.logging_steps
                    running_loss = 0.0
                    print(
                        f"step {global_step:>6}/{max_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"grad_norm {grad_norm:.3f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{steps_sec:.1f} steps/s | "
                        f"ETA {remaining/60:.1f} min"
                    )

                # ── STEP 10: Validation ───────────────────────────────────────────
                if global_step % training_config.eval_steps == 0:
                    val_loss, val_wer = run_validation(
                        model, val_subset_loader, processor, device, use_amp, amp_dtype
                    )
                    improved = early_stop.step(val_wer, global_step)
                    print(
                        f"\n[val] step {global_step} | "
                        f"val_loss {val_loss:.4f} | "
                        f"WER {val_wer:.4f} | "
                        f"{'✓ best' if improved else f'no improvement ({early_stop.counter}/{early_stop.patience})'}\n"
                    )
                    metrics = {"val_loss": val_loss, "val_wer": val_wer}
                    if improved:
                        save_checkpoint(model, optimizer, scaler,
                                        global_step, metrics, output_dir, "best")
                        export_best_weights(model, os.path.join(
                            output_dir, "exported_best"))
                    save_checkpoint(model, optimizer, scaler,
                                    global_step, metrics, output_dir, "latest")
                    if early_stop.should_stop:
                        print(f"[train] Early stopping at step {global_step}. "
                              f"Best WER {early_stop.best_wer:.4f} at step {early_stop.best_step}.")
                        break

                    model.train()

                    if early_stop.should_stop:
                        break

    # ── Training complete ─────────────────────────────────────────────────────
    val_loss, val_wer = run_validation(
        model, val_loader, processor, device, use_amp, amp_dtype
    )
    
    
    improved = early_stop.step(val_wer, global_step)

    metrics = {
        "val_loss": val_loss,
        "val_wer": val_wer,
    }

    # Always save the latest state
    save_checkpoint(
        model,
        optimizer,
        scaler,
        global_step,
        metrics,
        output_dir,
        "latest",
    )

    # If this final validation is the best, update the best checkpoint too
    if improved:
        save_checkpoint(
            model,
            optimizer,
            scaler,
            global_step,
            metrics,
            output_dir,
            "best",
        )
        export_best_weights(
            model,
            os.path.join(output_dir, "exported_best"),
        )
    summary = {
            "best_step": early_stop.best_step,
        "best_wer": early_stop.best_wer,
        "total_steps": global_step,
        "output_dir": output_dir,
    }
    print(
        f"\n[train] Done. Best WER {early_stop.best_wer:.4f} at step {early_stop.best_step}.")
    print(
        f"[train] Best checkpoint saved to {os.path.join(output_dir, 'best')}")
    return summary

def main():
    """
    Wires together the pieces owned by the other members and kicks off
    train(). This module only owns the training loop itself (steps 7-10),
    so the model / dataset / LoRA-attachment calls below import from the
    modules those members own — swap the import paths below to match
    wherever those land in the final repo layout.
    """
    import argparse
    from step6_training_config import (
        MedicalWhisperTrainingConfig,
        build_training_components,
    )

    parser = argparse.ArgumentParser(description="Fine-tune Whisper on medical speech data")
    parser.add_argument("--config", default="training_config.json",
                         help="Path to a saved MedicalWhisperTrainingConfig JSON file")
    parser.add_argument("--train-data", default=os.getenv("TRAIN_DATA_PATH", "data/train"),
                         help="Path to the training dataset (Team A owned)")
    parser.add_argument("--val-data", default=os.getenv("VAL_DATA_PATH", "data/val"),
                         help="Path to the validation dataset (Team A owned)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    if os.path.exists(args.config):
        training_config = MedicalWhisperTrainingConfig.load(args.config)
        print(f"[main] Loaded config from {args.config}")
    else:
        training_config = MedicalWhisperTrainingConfig()
        print(f"[main] No config file at {args.config} — using defaults")

    # ── Processor ────────────────────────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(training_config.model_name_or_path)

    # ── Model (base + LoRA) — owned by whichever module attaches the adapter ─
    # e.g. `from lora_setup import attach_lora`
    from lora_setup import attach_lora
    model = attach_lora(training_config, resume_from=os.path.join(
        training_config.output_dir, "latest"))

    # ── Datasets — owned by the data-preprocessing module ───────────────────
    # e.g. `from dataset import MedicalWhisperDataset`
    from dataset import MedicalWhisperDataset
    train_dataset = MedicalWhisperDataset(args.train_data, processor)
    val_dataset = MedicalWhisperDataset(args.val_data, processor)

    # ── Data collator (this module's own step6 config helper) ───────────────
    data_collator, _compute_metrics, training_config = build_training_components(
        processor=processor,
        lora_model=model,
        output_dir=training_config.output_dir,
    )

    summary = train(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=training_config,
        data_collator=data_collator,
    )

    print(f"[main] {summary}")
    return summary


if __name__ == "__main__":
    main()
