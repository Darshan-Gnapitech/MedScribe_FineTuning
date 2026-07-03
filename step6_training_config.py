"""
Step 6: Configure Training
============================
Builds training arguments, data collator, and WER compute_metrics function.
Outputs: MedicalWhisperTrainingConfig, DataCollatorSpeechSeq2SeqWithPadding,
        and compute_metrics callable — all ready for Member 3's Seq2SeqTrainer.
"""

import json
import torch
import evaluate
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

    output_dir: str = "./whisper-medical-lora"
    logging_dir: str = "./logs"
    optim: str = "adamw_torch"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8   # keeps effective batch size ~16
    fp16: bool = True
    bf16: bool = False
    num_train_epochs: int = 10
    max_steps: int = -1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    logging_steps: int = 20
    report_to: str = "tensorboard"
    predict_with_generate: bool = True
    generation_max_length: int = 448
    seed: int = 42

    # Custom fields — used directly by train.py, not passed to HF Trainer
    num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    eval_steps: int = 322

    def save(self, path: str = "training_config.json"):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[Config] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
    _CUSTOM_FIELDS = {"num_workers",
                    "early_stopping_patience", "adam_beta1", "adam_beta2"}

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
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(
            pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)
        return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}

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
