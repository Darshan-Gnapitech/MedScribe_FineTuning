"""
Step 3: Load Pretrained Whisper Model
=======================================
Loads the WhisperProcessor and WhisperForConditionalGeneration.
Runs a sanity-check inference on the first training sample.
Outputs: processor, model, baseline transcription string.
"""

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dotenv import load_dotenv
import os 
load_dotenv()  # load environment variables from .env file

MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-large-v3")  # default to small model
# MODEL_NAME = "openai/whisper-large-v3"  # uncomment for full run


def load_whisper(model_name: str = MODEL_NAME):
    
    print("\n" + "=" * 60)
    print("STEP 3: Loading Whisper model + processor")
    print("=" * 60)

    processor = WhisperProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    print(f"\n  Processor loaded  ({model_name})")

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded  |  parameters: {total:,}")

    return processor, model


def run_sanity_check(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    arr: np.ndarray,
    sample: dict,
) -> str:
    """
    Run a single forward pass and print base Whisper output vs ground truth.

    Returns
    -------
    baseline : str   raw (pre-LoRA) transcription
    """
    print("\nSanity check: base Whisper on first training sample ...")
    input_features = processor(
        arr,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    baseline = processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0]
    print(f"\n  Base Whisper : {baseline[:200]}")
    print(f"  Ground truth : {sample['sentence'][:200]}")

    return baseline


def print_deliverable():
    print("\n" + "=" * 60)
    print("DELIVERABLE -- Member 1 complete")
    print("=" * 60)
    print(
        "\nHand-off to Member 2:\n"
        "  dataset   -> DatasetDict  train(322) / validation(10) / test(10)\n"
        "               audio_array  : float32 list at 16 kHz\n"
        "               sentence     : doctor-corrected ground truth (TARGET)\n"
        "  model     -> WhisperForConditionalGeneration (base, LoRA added next)\n"
        "  processor -> WhisperProcessor (feature extractor + tokenizer)\n"
    )


if __name__ == "__main__":
    from step1_load_dataset import load_primock_med
    import numpy as np

    dataset = load_primock_med()
    dataset, sample, arr = decode_dataset(dataset)
    processor, model = load_whisper()
    baseline = run_sanity_check(model, processor, arr, sample)
    print_deliverable()
