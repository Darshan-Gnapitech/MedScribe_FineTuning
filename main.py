"""
main.py
========
Unified entry point for the full Whisper LoRA fine-tuning pipeline.
Orchestrates Steps 1-10 in order.
"""

import os

from step0_A_dataset_Split import split_chunk_dataset
from step0_build_dataset import build_dataset
from step1_load_dataset import load_chunk_dataset
from step2_load_whisper import load_whisper, run_sanity_check, print_deliverable
from step3_attach_lora import attach_lora
from step4b_lazy_dataset import LazyWhisperDataset
from step5_training_config import build_training_components
from step6_train import train
import soundfile as sf
import gc
from dotenv import load_dotenv

load_dotenv()


def main():
    print("\n" + "=" * 60)
    print("WHISPER MEDICAL LoRA — FULL PIPELINE")
    print("=" * 60)
    EXPORT_DIR = os.getenv(
        "EXPORT_DIR", r"./whisper-medical-lora/exported_best")
    CSV_PATH = os.getenv("CSV_PATH", r"./pocfinal/datasets.xlsx")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", r"./")
    PROCESSED_DATASET_PATH = os.getenv(
        "PROCESSED_DATASET_PATH", r"./processed_whisper_dataset")
    DATASET_PATH = os.getenv("DATASET_PATH", r"./datasets")
    build_info = build_dataset(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
    )
    split_chunk_dataset(input_dir=OUTPUT_DIR, output_dir=DATASET_PATH)
    dataset = load_chunk_dataset(output_dir=DATASET_PATH)
    sample = dataset["train"][0]
    arr, sr = sf.read(
        sample["audio"]["path"],
        dtype="float32"
    )
    # ── Step 3: Load Whisper ─────────────────────────────────────
    processor, model = load_whisper()
    baseline = run_sanity_check(model, processor, arr, sample)
    print_deliverable()

    # ── Step 4: Attach LoRA ──────────────────────────────────────
    model = attach_lora(model, resume_from=EXPORT_DIR)
    # ── Step 5: Lazy preprocessing (no upfront .map(), no OOM/disk risk) ──
    train_dataset = LazyWhisperDataset(dataset["train"], processor)
    val_dataset = LazyWhisperDataset(dataset["validation"], processor)
    print(f"[main] Using lazy on-the-fly preprocessing "
          f"(train={len(train_dataset):,}, val={len(val_dataset):,})")
    gc.collect()

    # ── Step 6: Build training components ────────────────────────
    data_collator, compute_metrics, config = build_training_components(
        processor=processor,
        lora_model=model,
        output_dir="./whisper-medical-lora",
        save_config_path="training_config.json",
    )

    # ── Steps 7-10: Training loop ────────────────────────────────
    summary = train(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=config,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best WER      : {summary['best_wer']:.4f}")
    print(f"  Best step     : {summary['best_step']}")
    print(f"  Total steps   : {summary['total_steps']}")
    print(f"  Checkpoint    : {summary['output_dir']}/best")


if __name__ == "__main__":
    main()
