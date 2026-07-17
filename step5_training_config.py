"""
Step 6: Configure Training
============================
Builds training arguments, data collator, and WER compute_metrics function.
Outputs: MedicalWhisperTrainingConfig, DataCollatorSpeechSeq2SeqWithPadding,
        and compute_metrics callable — all ready for Member 3's Seq2SeqTrainer.
"""

import json
import os
import torch
import jiwer
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from transformers import WhisperProcessor, Seq2SeqTrainingArguments


# =============================================================================
# Training Config
# =============================================================================

@dataclass
class MedicalWhisperTrainingConfig:
    """Hyper-parameters for the custom training loop (train.py).
    NOT passed to Seq2SeqTrainingArguments — train.py owns the loop."""

    """Hyper-parameters for the custom training loop (train.py).
    NOT passed to Seq2SeqTrainingArguments — train.py owns the loop."""
    model_name_or_path: str =os.getenv("WHISPER_MODEL_NAME", "/home/nisha/whisper-large-v3-hf")
    output_dir: str = "./whisper-medical-lora"
    per_device_train_batch_size: int =64 
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    num_train_epochs: int = 3
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    predict_with_generate: bool = True
    generation_max_length: int = 225
    report_to: str = "none"
    weight_decay: float = 0.01

    # custom fields — NOT part of Seq2SeqTrainingArguments
    num_workers: int =12 
    early_stopping_patience: int = 3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    eval_steps: int = 200
    max_grad_norm: float = 1.0
    max_vram_gb: float = 20.0

    _CUSTOM_FIELDS = {
        "num_workers", "early_stopping_patience", "adam_beta1", "adam_beta2", "max_vram_gb"
    }
    

    def save(self, path: str = "training_config.json"):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[Config] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))

    def to_seq2seq_training_arguments(self) -> Seq2SeqTrainingArguments:
        d = {k: v for k, v in asdict(self).items()
            if k not in self._CUSTOM_FIELDS}
        return Seq2SeqTrainingArguments(**d)


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Dynamic per-batch padding for Seq2SeqTrainer."""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]}
                        for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        batch["raw_texts"] = [
            f["raw_text"]
            for f in features
        ]

        return batch


# =============================================================================
# Compute Metrics
# =============================================================================

def build_compute_metrics(processor: WhisperProcessor):
    """WER metric function compatible with Seq2SeqTrainer."""

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)

        # jiwer.wer errors on empty reference strings, so guard against that
        pred_str = [p if p.strip() else " " for p in pred_str]
        label_str = [l if l.strip() else " " for l in label_str]

        wer = jiwer.wer(label_str, pred_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# =============================================================================
# Convenience builder
# =============================================================================

def build_training_components(
    processor: WhisperProcessor,
    lora_model,
    output_dir: str = "./whisper-medical-lora",
    save_config_path: str = "training_config.json",
):
    """
    Instantiates and returns all training components needed by Member 3.

    Returns
    -------
    training_args   : Seq2SeqTrainingArguments
    data_collator   : DataCollatorSpeechSeq2SeqWithPadding
    compute_metrics : callable
    config          : MedicalWhisperTrainingConfig
    """
    print("\n-- Step 6: Building training config --")
    config = MedicalWhisperTrainingConfig(output_dir=output_dir)
    config.save(save_config_path)

    # training_args = config.to_seq2seq_training_arguments()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=lora_model.config.decoder_start_token_id,
    )

    compute_metrics = build_compute_metrics(processor)

    return data_collator, compute_metrics, config


if __name__ == "__main__":
    print("Step 6 module loaded successfully (no standalone run needed).")
    print("Classes available: MedicalWhisperTrainingConfig, "
        "DataCollatorSpeechSeq2SeqWithPadding")
    print("Functions available: build_compute_metrics, build_training_components")
