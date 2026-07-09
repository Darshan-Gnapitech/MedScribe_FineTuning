"""
main.py
========
Unified entry point for the full Whisper LoRA fine-tuning pipeline.
Orchestrates Steps 1-10 in order.
"""

from matplotlib.pylab import sample
import torch
from step0_build_dataset import build_dataset
from step1_load_dataset import load_chunk_dataset
from step2_load_whisper import load_whisper, run_sanity_check, print_deliverable
from step3_attach_lora import attach_lora
from step4_preprocess import preprocess_dataset
from step5_training_config import build_training_components
from step6_train import train
import soundfile as sf
import gc
def main():
    print("\n" + "=" * 60)
    print("WHISPER MEDICAL LoRA — FULL PIPELINE")
    print("=" * 60)
    EXPORT_DIR = "./whisper-medical-lora/exported_best"
    CSV_PATH = r"./pocfinal/datasets.xlsx"


    OUTPUT_DIR = r"./"

    build_info = build_dataset(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
    )
    dataset = load_chunk_dataset(
        manifest_path=build_info["manifest_path"],
        chunks_dir=build_info["chunks_dir"],
    )   
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

    # ── Step 5: Preprocess dataset ───────────────────────────────
    dataset = preprocess_dataset(dataset, processor)
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
