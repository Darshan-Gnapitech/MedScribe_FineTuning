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
from step4_preprocess import MedicalWhisperPreprocessor
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

    preprocessor = MedicalWhisperPreprocessor(
        processor,
        num_proc=min(8, os.cpu_count() or 1),
    )
    # ── Step 5: Preprocess dataset ────────────────────────────────
    def _processed_cache_is_stale(cache_path, raw_dataset):
        from datasets import load_from_disk
        try:
            cached = load_from_disk(cache_path)
        except Exception as e:
            print(
                f"[main] No usable cache at {cache_path} ({e.__class__.__name__}) — will preprocess fresh")
            return True, None
        for split in ("train", "validation", "test"):
            raw_n, cached_n = len(raw_dataset[split]), len(cached[split])
            print(
                f"[main] cache check [{split}]: raw={raw_n:,} cached={cached_n:,}")
            if raw_n != cached_n:
                print(f"[main] -> STALE on {split}, will reprocess")
                return True, None
        print("[main] cache check passed for all splits — reusing cached processed dataset")
        return False, cached


    is_stale, cached_dataset = _processed_cache_is_stale(
        PROCESSED_DATASET_PATH, dataset)

    if not is_stale:
        print(f"[main] Using cached processed dataset -> {PROCESSED_DATASET_PATH}")
        dataset = cached_dataset
    else:
        dataset = preprocessor(dataset)
        dataset.save_to_disk(PROCESSED_DATASET_PATH)
        print(f"[main] Saved processed dataset -> {PROCESSED_DATASET_PATH}")
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
        train_dataset=dataset["train"],
        val_dataset=dataset["validation"],
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
