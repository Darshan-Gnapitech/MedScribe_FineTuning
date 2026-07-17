"""
export_whisperx.py
==================

Offline WhisperX export utility.

Pipeline

    Merge LoRA
          ↓
    Convert to CT2
          ↓
    Verify using WhisperX

This version NEVER downloads anything.

Requirements

- Local HuggingFace Whisper model
- Local CT2 model
- Installed WhisperX package
- No HuggingFace downloads
- No fallback model downloads
"""

import argparse
import os
import sys
from pathlib import Path

from merge_lora import merge_lora
from convert_to_ct2 import convert_to_ct2


LOCAL_WHISPER_MODEL = "/home/nisha/whisper-large-v3-hf"


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_local_whisper_model(model_dir: str):

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"\nLocal Whisper model not found:\n{model_dir}"
        )

    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
    ]

    missing = []

    for file in required:
        if not (model_dir / file).exists():
            missing.append(file)

    if missing:
        raise RuntimeError(
            "Invalid Whisper model directory.\n"
            "Missing files:\n"
            + "\n".join(missing)
        )


def validate_ct2_model(ct2_dir: str):

    if not os.path.isdir(ct2_dir):
        raise FileNotFoundError(
            f"CT2 directory not found:\n{ct2_dir}"
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
            "Invalid CT2 model.\n"
            "Missing files:\n"
            + "\n".join(missing)
        )


# ---------------------------------------------------------------------
# WhisperX verification
# ---------------------------------------------------------------------

def run_whisperx_smoke_test(
    ct2_dir: str,
    audio_file,
    device: str,
    compute_type: str,
    hf_token,
):

    import whisperx

    validate_ct2_model(ct2_dir)

    print()
    print("[export_whisperx] Loading WhisperX model...")
    print(ct2_dir)

    model = whisperx.load_model(
        ct2_dir,
        device=device,
        compute_type=compute_type,
        download_root=None,
    )

    print("[export_whisperx] WhisperX model loaded successfully.")

    if audio_file is None:

        print()
        print("[export_whisperx] No audio supplied.")
        print("[export_whisperx] Offline verification complete.")

        return

    if not os.path.exists(audio_file):
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_file}"
        )

    print()
    print("[export_whisperx] Loading audio...")

    audio = whisperx.load_audio(audio_file)

    print("[export_whisperx] Running transcription...")

    result = model.transcribe(
        audio,
        batch_size=8,
    )

    language = result.get("language", "unknown")

    print(f"[export_whisperx] Language: {language}")

    print()

    for seg in result["segments"][:5]:
        print(
            f"[{seg['start']:.2f}-{seg['end']:.2f}] "
            f"{seg['text']}"
        )

    print()
    print("[export_whisperx] Running forced alignment...")

    try:

        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

    except Exception as e:

        raise RuntimeError(
            "\nAlignment model is not available locally.\n"
            "Offline mode prohibits downloading models.\n\n"
            f"{e}"
        )

    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
    )

    print(
        f"[export_whisperx] Alignment complete "
        f"({len(result['segments'])} segments)."
    )

    if hf_token:

        raise RuntimeError(
            "\nSpeaker diarization requires downloading "
            "PyAnnote models from Hugging Face.\n"
            "Offline mode disables diarization."
        )

    print("[export_whisperx] Diarization skipped.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Offline WhisperX export utility."
    )

    parser.add_argument(
        "--base_model",
        default=os.getenv(
            "WHISPER_MODEL_NAME",
            LOCAL_WHISPER_MODEL,
        ),
        help="Local HuggingFace Whisper model directory.",
    )

    parser.add_argument(
        "--adapter_dir",
        default="./whisper-medical-lora/exported_best",
    )

    parser.add_argument(
        "--merged_dir",
        default="./whisper-medical-lora/merged_hf",
    )

    parser.add_argument(
        "--ct2_dir",
        default="./whisper-medical-lora/ct2_model",
    )

    parser.add_argument(
        "--quantization",
        default="float16",
    )

    parser.add_argument(
        "--skip_merge",
        action="store_true",
    )

    parser.add_argument(
        "--skip_convert",
        action="store_true",
    )

    parser.add_argument(
        "--audio_file",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--compute_type",
        default="int8",
    )

    parser.add_argument(
        "--hf_token",
        default=None,
        help="Ignored in offline mode.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    validate_local_whisper_model(args.base_model)

    if not args.skip_merge:

        merge_lora(
            args.base_model,
            args.adapter_dir,
            args.merged_dir,
        )

    else:

        print(
            f"[export_whisperx] Using existing merged model:\n"
            f"{args.merged_dir}"
        )

    if not args.skip_convert:

        convert_to_ct2(
            args.merged_dir,
            args.ct2_dir,
            args.quantization,
            force=True,
        )

    else:

        print(
            f"[export_whisperx] Using existing CT2 model:\n"
            f"{args.ct2_dir}"
        )

    run_whisperx_smoke_test(
        args.ct2_dir,
        args.audio_file,
        args.device,
        args.compute_type,
        args.hf_token,
    )


if __name__ == "__main__":

    try:
        main()

    except (FileNotFoundError, RuntimeError) as e:

        print(f"\n[export_whisperx] ERROR:\n{e}", file=sys.stderr)
        sys.exit(1)

    except ImportError:

        print(
            "\n[export_whisperx] ERROR:\n"
            "WhisperX is not installed in the current environment.",
            file=sys.stderr,
        )

        sys.exit(1)