"""
merge_lora.py
=============

Offline LoRA merge utility.

This version loads ONLY a locally available Hugging Face Whisper model.
It never downloads models, tokenizers, processors, or any Hugging Face
resources.

Pipeline:
    Local Whisper HF Model
            ↓
       Load LoRA Adapter
            ↓
     merge_and_unload()
            ↓
 Save merged HuggingFace model

Compatible with:
    convert_to_ct2.py
    WhisperX export

Example:

python merge_lora.py \
    --base_model /home/nisha/whisper-large-v3-hf \
    --adapter_dir ./whisper-medical-lora/exported_best \
    --output_dir ./whisper-medical-lora/merged_hf
"""

import argparse
import os
import sys
from pathlib import Path

from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ---------------------------------------------------------------------
# Default local Whisper Large-v3 model
# ---------------------------------------------------------------------

LOCAL_WHISPER_MODEL = "/home/nisha/whisper-large-v3-hf"

# ---------------------------------------------------------------------
# Required files for a valid Hugging Face Whisper model
# ---------------------------------------------------------------------

REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
]

OPTIONAL_MODEL_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
]

OPTIONAL_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_local_whisper_model(model_dir: str):
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nLocal Whisper model directory does not exist:\n"
            f"{model_dir}\n\n"
            "Offline mode prohibits downloading models."
        )

    if not model_path.is_dir():
        raise RuntimeError(
            f"{model_dir} is not a directory."
        )

    # ---------------------------------------------------------
    # Required configuration files
    # ---------------------------------------------------------
    missing = [
        f for f in REQUIRED_FILES
        if not (model_path / f).exists()
    ]

    if missing:
        raise RuntimeError(
            "Invalid Whisper model directory.\n"
            "Missing required files:\n"
            + "\n".join(missing)
        )

    # ---------------------------------------------------------
    # Accept BOTH:
    #
    # 1. model.safetensors
    # 2. pytorch_model.bin
    # 3. sharded safetensors
    # ---------------------------------------------------------
    has_single_file = (
        (model_path / "model.safetensors").exists()
        or (model_path / "pytorch_model.bin").exists()
    )

    has_sharded = (
        (model_path / "model.safetensors.index.json").exists()
        and len(list(model_path.glob("model-*.safetensors"))) > 0
    )

    if not (has_single_file or has_sharded):
        raise RuntimeError(
            "Model weights not found.\n"
            "Expected one of:\n"
            "  model.safetensors\n"
            "  pytorch_model.bin\n"
            "  OR\n"
            "  model.safetensors.index.json + model-*.safetensors"
        )

    # ---------------------------------------------------------
    # Tokenizer files
    # ---------------------------------------------------------
    has_tokenizer = any(
        (model_path / f).exists()
        for f in OPTIONAL_TOKENIZER_FILES
    )

    if not has_tokenizer:
        raise RuntimeError(
            "Tokenizer files not found."
        )

    print(f"[merge_lora] Verified local Whisper model: {model_dir}")


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------

def merge_lora(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
) -> str:

    validate_local_whisper_model(base_model)

    adapter_config = os.path.join(
        adapter_dir,
        "adapter_config.json",
    )

    if not os.path.exists(adapter_config):
        raise FileNotFoundError(
            f"No adapter_config.json found in:\n{adapter_dir}"
        )

    print(f"[merge_lora] Loading local Whisper model:")
    print(f"              {base_model}")

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        local_files_only=True,
    )

    processor = WhisperProcessor.from_pretrained(
        base_model,
        local_files_only=True,
    )

    processor.tokenizer.set_prefix_tokens(
        language="en",
        task="transcribe",
    )

    print(f"[merge_lora] Loading LoRA adapter:")
    print(f"              {adapter_dir}")

    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        local_files_only=True,
    )

    print("[merge_lora] Merging LoRA weights...")

    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)

    print(f"[merge_lora] Saving merged model:")

    model.save_pretrained(output_dir)

    processor.save_pretrained(output_dir)

    print(f"[merge_lora] Done.")
    print(f"[merge_lora] Output: {output_dir}")

    return output_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Offline LoRA merge utility."
    )

    parser.add_argument(
        "--base_model",
        default=os.getenv(
            "WHISPER_MODEL_NAME",
            LOCAL_WHISPER_MODEL,
        ),
        help=(
            "Local Hugging Face Whisper model directory. "
            "No online downloads are permitted."
        ),
    )

    parser.add_argument(
        "--adapter_dir",
        default="./whisper-medical-lora/exported_best",
        help="Directory containing adapter_config.json",
    )

    parser.add_argument(
        "--output_dir",
        default="./whisper-medical-lora/merged_hf",
        help="Directory to save merged model",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    args = parse_args()

    try:
        merge_lora(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
        )

    except Exception as e:
        print(f"\n[merge_lora] ERROR:\n{e}", file=sys.stderr)
        sys.exit(1)