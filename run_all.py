"""

run_all.py

==========
 
Offline export utility.
 
Pipeline:
 
    Local HF Whisper Model

            ↓

       Merge LoRA Adapter

            ↓

     Convert to CTranslate2

            ↓

     Verify using WhisperX
 
This version NEVER downloads:
 
- Whisper models

- Tokenizers

- Processors

- WhisperX models

- HuggingFace resources
 
Everything must already exist locally.

"""
 
import argparse

import importlib.util

import os

import shutil

import subprocess

import sys

from pathlib import Path
 
 
# ---------------------------------------------------------------------

# Local paths

# ---------------------------------------------------------------------
 
LOCAL_WHISPER_MODEL = "/home/nisha/whisper-large-v3-hf"
 
REQUIRED_MODEL_FILES = [

    "config.json",

    "generation_config.json",

    "preprocessor_config.json",

]
 
OPTIONAL_WEIGHT_FILES = [

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
 
    model_dir = Path(model_dir)
 
    if not model_dir.exists():

        raise FileNotFoundError(

            f"\nLocal Whisper model not found:\n{model_dir}\n"

            "Offline mode prohibits automatic downloads."

        )
 
    if not model_dir.is_dir():

        raise RuntimeError(

            f"{model_dir} is not a valid directory."

        )
 
    missing = []
 
    for file in REQUIRED_MODEL_FILES:

        if not (model_dir / file).exists():

            missing.append(file)
 
    if missing:

        raise RuntimeError(

            "Invalid Whisper model directory.\n"

            "Missing files:\n"

            + "\n".join(missing)

        )
 
   # Support both single-file and sharded Hugging Face checkpoints

    has_single_weights = (

        (model_dir / "model.safetensors").exists()

        or (model_dir / "pytorch_model.bin").exists()

)
 
    has_sharded_weights = (

        (model_dir / "model.safetensors.index.json").exists()

        and any(model_dir.glob("model-*.safetensors"))

)
 
    if not (has_single_weights or has_sharded_weights):

        raise RuntimeError(

        "No Whisper model weights found.\n"

        "Expected one of:\n"

        "  model.safetensors\n"

        "  pytorch_model.bin\n"

        "  OR\n"

        "  model.safetensors.index.json + model-*.safetensors"

    )
 
    has_tokenizer = any(

        (model_dir / file).exists()

        for file in OPTIONAL_TOKENIZER_FILES

    )
 
    if not has_tokenizer:

        raise RuntimeError(

            "Tokenizer files are missing."

        )
 
    print(f"[run_all] Verified local Whisper model:")

    print(f"          {model_dir}")
 
 
# ---------------------------------------------------------------------

# Stage 1

# Dependency check

# ---------------------------------------------------------------------
 
def ensure_dependencies(skip_install: bool):
 
    if skip_install:

        print("[run_all] Dependency installation skipped.")

        return
 
    required = {

        "ctranslate2": "ctranslate2",

        "whisperx": "whisperx",

        "transformers": "transformers",

        "peft": "peft",

    }
 
    missing = []
 
    for package, module in required.items():
 
        if importlib.util.find_spec(module) is None:

            missing.append(package)
 
    if missing:

        raise RuntimeError(

            "\nRequired packages are missing:\n"

            + "\n".join(missing)

            + "\n\nOffline mode does NOT install packages.\n"

            "Install them manually before running."

        )
 
    print("[run_all] All required packages are available.")
 
 
# ---------------------------------------------------------------------

# Stage 2

# Merge LoRA

# ---------------------------------------------------------------------
 
def merge_lora(

    base_model: str,

    adapter_dir: str,

    output_dir: str,

):
 
    from transformers import (

        WhisperForConditionalGeneration,

        WhisperProcessor,

    )
 
    from peft import PeftModel
 
    validate_local_whisper_model(base_model)
 
    adapter_config = os.path.join(

        adapter_dir,

        "adapter_config.json",

    )
 
    if not os.path.exists(adapter_config):

        raise FileNotFoundError(

            f"No adapter_config.json found in:\n{adapter_dir}"

        )
 
    print()

    print("[run_all] Loading local Whisper model...")

    print(base_model)
 
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
 
    print("[run_all] Loading LoRA adapter...")

    print(adapter_dir)
 
    model = PeftModel.from_pretrained(

        model,

        adapter_dir,

        local_files_only=True,

    )
 
    print("[run_all] Merging LoRA...")
 
    model = model.merge_and_unload()
 
    os.makedirs(

        output_dir,

        exist_ok=True,

    )
 
    model.save_pretrained(output_dir)

    processor.save_pretrained(output_dir)
 
    print("[run_all] Merged model saved:")

    print(output_dir)
 
    return output_dir

# ---------------------------------------------------------------------

# Stage 3

# Convert merged HF model to CTranslate2

# ---------------------------------------------------------------------
 
def convert_to_ct2(

    model_dir: str,

    output_dir: str,

    quantization: str,

    force: bool,

):
 
    config_file = os.path.join(model_dir, "config.json")
 
    if not os.path.exists(config_file):

        raise FileNotFoundError(

            f"\nMerged HuggingFace model not found:\n"

            f"{model_dir}\n"

            "Merge stage may have failed."

        )
 
    converter = os.path.join(

        os.path.dirname(sys.executable),

        "ct2-transformers-converter",

    )
 
    if not os.path.exists(converter):

        raise RuntimeError(

            f"ct2-transformers-converter not found:\n{converter}"

        )

    if converter is None:

        raise RuntimeError(

            "\nct2-transformers-converter is not installed "

            "or not available in PATH.\n"

            "Offline mode will not attempt installation."

        )
 
    if force and os.path.exists(output_dir):

        print(f"[run_all] Removing existing CT2 directory:")

        print(output_dir)

        shutil.rmtree(output_dir)
 
    cmd = [

        converter,

        "--model",

        model_dir,

        "--output_dir",

        output_dir,

        "--copy_files",

        "tokenizer.json",

        "preprocessor_config.json",

        "--quantization",

        quantization,

    ]
 
    if force:

        cmd.append("--force")
 
    print()

    print("[run_all] Converting HuggingFace model to CTranslate2...")

    print(" ".join(cmd))
 
    result = subprocess.run(

        cmd,

        capture_output=True,

        text=True,

    )
 
    if result.returncode != 0:

        raise RuntimeError(

            "ct2 conversion failed.\n\n"

            f"STDOUT:\n{result.stdout}\n\n"

            f"STDERR:\n{result.stderr}"

        )
 
    print(result.stdout)
 
    required = [

        "config.json",

        "model.bin",

    ]
 
    missing = []
 
    for file in required:

        if not os.path.exists(os.path.join(output_dir, file)):

            missing.append(file)
 
    if missing:

        raise RuntimeError(

            "CT2 conversion completed but required files are missing:\n"

            + "\n".join(missing)

        )
 
    print("[run_all] CT2 model successfully created.")

    print(output_dir)
 
    return output_dir
 
 
# ---------------------------------------------------------------------

# CT2 validation

# ---------------------------------------------------------------------
 
def validate_ct2_model(ct2_dir: str):
 
    if not os.path.isdir(ct2_dir):

        raise FileNotFoundError(

            f"\nCT2 directory does not exist:\n{ct2_dir}"

        )
 
    required = [

        "config.json",

        "model.bin",

    ]
 
    missing = []
 
    for file in required:

        if not os.path.exists(os.path.join(ct2_dir, file)):

            missing.append(file)
 
    if missing:

        raise RuntimeError(

            "Invalid CT2 model.\nMissing files:\n"

            + "\n".join(missing)

        )
 
 
# ---------------------------------------------------------------------

# WhisperX verification

# ---------------------------------------------------------------------
 
def verify_with_whisperx(

    ct2_dir: str,

    audio_file,

    device: str,

    compute_type: str,

    hf_token,

    align_model: str = None,

    align_model_dir: str = None,

):
 
    import whisperx
 
    validate_ct2_model(ct2_dir)
 
    print()

    print("[run_all] Loading WhisperX using local CT2 model...")

    print(ct2_dir)
 
    model = whisperx.load_model(

        ct2_dir,

        device=device,

        compute_type=compute_type,

        download_root=None,

    )
 
    print("[run_all] WhisperX model loaded successfully.")
 
    if audio_file is None:

        print()

        print("[run_all] No audio supplied.")

        print("[run_all] Offline verification completed.")

        return
 
    if not os.path.exists(audio_file):

        raise FileNotFoundError(

            f"Audio file not found:\n{audio_file}"

        )
 
    print()

    print("[run_all] Loading audio...")

    audio = whisperx.load_audio(audio_file)
 
    print("[run_all] Running transcription...")
 
    result = model.transcribe(

        audio,

        batch_size=8,

    )
 
    language = result.get("language", "unknown")
 
    print(f"[run_all] Detected language: {language}")
 
    print()

    print("Sample transcription:")
 
    for segment in result["segments"][:5]:

        print(

            f"[{segment['start']:.2f} - {segment['end']:.2f}] "

            f"{segment['text']}"

        )
 
    print()

    print("[run_all] Loading alignment model...")
 
    try:
 
        align_model_obj, metadata = whisperx.load_align_model(

            language_code=language,

            device=device,

            model_name=align_model,

            model_dir=align_model_dir,

        )
 
    except Exception as e:
 
        raise RuntimeError(

            "\nAlignment model is not available locally.\n"

            "Offline mode does not allow downloading alignment models.\n\n"

            f"Original error:\n{e}"

        )
 
    print("[run_all] Running forced alignment...")
 
    result = whisperx.align(

        result["segments"],

        align_model_obj,

        metadata,

        audio,

        device,

    )
 
    print(

        f"[run_all] Alignment complete."

        f" {len(result['segments'])} segments aligned."

    )
 
    if hf_token:
 
        raise RuntimeError(

            "\nSpeaker diarization has been disabled.\n"

            "Offline mode does not permit downloading "

            "PyAnnote models from Hugging Face."

        )
 
    print()

    print("[run_all] Diarization skipped.")

    print("[run_all] Offline verification completed.")

# ---------------------------------------------------------------------

# Command Line Arguments

# ---------------------------------------------------------------------
 
def parse_args():
 
    parser = argparse.ArgumentParser(

        description=(

            "Offline pipeline:\n"

            "Merge LoRA -> Convert to CTranslate2 -> Verify with WhisperX"

        )

    )
 
    parser.add_argument(

        "--base_model",

        default=os.getenv(

            "WHISPER_MODEL_NAME",

            LOCAL_WHISPER_MODEL,

        ),

        help=(

            "Local HuggingFace Whisper model directory. "

            "No online models are supported."

        ),

    )
 
    parser.add_argument(

        "--adapter_dir",

        default="./whisper-medical-lora/exported_best",

        help="Directory containing the exported LoRA adapter.",

    )
 
    parser.add_argument(

        "--merged_dir",

        default="./whisper-medical-lora/merged_hf",

        help="Directory to save merged HuggingFace model.",

    )
 
    parser.add_argument(

        "--ct2_dir",

        default="./whisper-medical-lora/ct2_model",

        help="Directory to save the converted CTranslate2 model.",

    )
 
    parser.add_argument(

        "--quantization",

        default="float16",

        choices=[

            "float32",

            "float16",

            "int16",

            "int8",

            "int8_float16",

        ],

        help="CTranslate2 quantization type.",

    )
 
    parser.add_argument(

        "--skip_install",

        action="store_true",

        help="Skip dependency validation.",

    )
 
    parser.add_argument(

        "--skip_merge",

        action="store_true",

        help="Reuse an existing merged HuggingFace model.",

    )
 
    parser.add_argument(

        "--skip_convert",

        action="store_true",

        help="Reuse an existing CT2 model.",

    )
 
    parser.add_argument(

        "--audio_file",

        default=None,

        help="Optional audio file for WhisperX verification.",

    )
 
    parser.add_argument(

        "--device",

        default="cpu",

        choices=["cpu", "cuda"],

    )
 
    parser.add_argument(

        "--compute_type",

        default="int8",

        help="WhisperX inference compute type.",

    )
 
    parser.add_argument(

        "--hf_token",

        default=None,

        help=(

            "Ignored in offline mode. "

            "Speaker diarization is disabled."

        ),

    )
 
    parser.add_argument(

        "--align_model",

        default=None,

        help=(

            "Alignment model to use with WhisperX. "

            "Pass 'MMS_FA' for the multilingual torchaudio model, "

            "or leave unset for WhisperX's per-language wav2vec2 defaults."

        ),

    )
 
    parser.add_argument(

        "--align_model_dir",

        default=None,

        help="Local directory containing the cached alignment model (e.g. MMS_FA checkpoint).",

    )
 
    return parser.parse_args()
 
 
# ---------------------------------------------------------------------

# Main

# ---------------------------------------------------------------------
 
def main():
 
    args = parse_args()
 
    try:
 
        print()

        print("=" * 70)

        print(" Offline Whisper Export Pipeline ")

        print("=" * 70)
 
        ensure_dependencies(args.skip_install)
 
        validate_local_whisper_model(args.base_model)
 
        if not args.skip_merge:
 
            merge_lora(

                base_model=args.base_model,

                adapter_dir=args.adapter_dir,

                output_dir=args.merged_dir,

            )
 
        else:
 
            print()

            print("[run_all] Skipping merge.")

            print(f"[run_all] Using existing model:")

            print(args.merged_dir)
 
            if not os.path.isdir(args.merged_dir):

                raise FileNotFoundError(

                    f"Merged model directory not found:\n"

                    f"{args.merged_dir}"

                )
 
        if not args.skip_convert:
 
            convert_to_ct2(

                model_dir=args.merged_dir,

                output_dir=args.ct2_dir,

                quantization=args.quantization,

                force=True,

            )
 
        else:
 
            print()

            print("[run_all] Skipping CT2 conversion.")

            print(f"[run_all] Using existing CT2 model:")

            print(args.ct2_dir)
 
            validate_ct2_model(args.ct2_dir)
 
        verify_with_whisperx(

            ct2_dir=args.ct2_dir,

            audio_file=args.audio_file,

            device=args.device,

            compute_type=args.compute_type,

            hf_token=args.hf_token,

            align_model=args.align_model,

            align_model_dir=args.align_model_dir,

        )
 
        print()

        print("=" * 70)

        print(" Pipeline completed successfully ")

        print("=" * 70)
 
    except (

        FileNotFoundError,

        RuntimeError,

        ValueError,

    ) as e:
 
        print()

        print("=" * 70)

        print(" ERROR ")

        print("=" * 70)

        print(e, file=sys.stderr)

        sys.exit(1)
 
    except ImportError as e:
 
        print()

        print("=" * 70)

        print(" Missing Python dependency ")

        print("=" * 70)

        print(e, file=sys.stderr)

        print(

            "\nOffline mode will not install packages automatically.",

            file=sys.stderr,

        )

        sys.exit(1)
 
    except KeyboardInterrupt:
 
        print()

        print("\nInterrupted by user.")

        sys.exit(130)
 
    except Exception as e:
 
        print()

        print("=" * 70)

        print(" Unexpected Error ")

        print("=" * 70)

        print(e, file=sys.stderr)

        sys.exit(1)
 
 
# ---------------------------------------------------------------------

# Entry Point

# ---------------------------------------------------------------------
 
if __name__ == "__main__":

    main()
 
