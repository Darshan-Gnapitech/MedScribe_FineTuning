"""
convert_to_ct2.py
==================
OPTIONAL, standalone post-training utility.

Converts a merged Hugging Face Whisper model (the output of
merge_lora.py) into CTranslate2 format — the format faster-whisper
and WhisperX expect for inference. Does not touch the training
pipeline or model_export.py.

Requires the `ctranslate2` package, which ships the CLI tool
`ct2-transformers-converter` invoked here as a subprocess:
    pip install ctranslate2

Usage
-----
    python convert_to_ct2.py \
        --model_dir ./whisper-medical-lora/merged_hf \
        --output_dir ./whisper-medical-lora/ct2_model \
        --quantization float16
"""

import argparse
import os
import shutil
import subprocess
import sys


def convert_to_ct2(model_dir: str, output_dir: str, quantization: str, force: bool) -> str:
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"'{model_dir}' doesn't look like a merged HF model directory "
            "(no config.json found). Run merge_lora.py first."
        )

    if shutil.which("ct2-transformers-converter") is None:
        raise RuntimeError(
            "ct2-transformers-converter not found on PATH. Install it with:\n"
            "  pip install ctranslate2"
        )

    if force and os.path.exists(output_dir):
        print(f"[convert_to_ct2] --force set, removing existing {output_dir}")
        shutil.rmtree(output_dir)

    cmd = [
        "ct2-transformers-converter",
        "--model", model_dir,
        "--output_dir", output_dir,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", quantization,
    ]
    if force:
        cmd.append("--force")

    print(f"[convert_to_ct2] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "ct2-transformers-converter failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    print(result.stdout)
    print(f"[convert_to_ct2] CTranslate2 model written -> {output_dir}")
    return output_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a merged HF Whisper model to CTranslate2 format."
    )
    p.add_argument(
        "--model_dir",
        default="./whisper-medical-lora/merged_hf",
        help="Merged HF model directory (output of merge_lora.py).",
    )
    p.add_argument(
        "--output_dir",
        default="./whisper-medical-lora/ct2_model",
        help="Where to write the CTranslate2 model.",
    )
    p.add_argument(
        "--quantization",
        default="float16",
        choices=["float32", "float16", "int8", "int8_float16", "int8_float32"],
        help="CTranslate2 quantization at conversion time. Use float32 or "
             "int8 if you don't have a GPU.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        convert_to_ct2(args.model_dir, args.output_dir, args.quantization, args.force)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[convert_to_ct2] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
