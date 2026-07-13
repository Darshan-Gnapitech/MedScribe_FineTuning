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
 
MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-large-v3")
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import soundfile as sf
import numpy as np

 
def load_whisper(model_name: str = MODEL_NAME, device: torch.device = DEVICE):
 
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
 
    model.to(device)
    model.eval()  # disable dropout for deterministic, representative inference
    approx_gb = total * 4 / 1e9  # fp32 unless torch_dtype is set on from_pretrained
    print(f"  Model loaded  |  parameters: {total:,}  (~{approx_gb:.1f} GB fp32)  |  device: {device}")
 
    return processor, model
 
 
def run_sanity_check(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    arr: np.ndarray,
    sample: dict,
    device: torch.device = DEVICE,
) -> str:
    """
    Run a single forward pass and print base Whisper output vs ground truth.
 
    Returns
    -------
    baseline : str   raw (pre-LoRA) transcription
    """
    print(f"\nSanity check: base Whisper on first training sample (device={device}) ...")
    input_features = processor(
        arr,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)
 
    model.eval()  # belt-and-suspenders: safe even if load_whisper's caller mutated model state
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
    from step1_load_dataset import load_chunk_dataset
    from datasets import Audio
    import numpy as np
    import soundfile as sf
    import os

    OUTPUT_DIR = os.getenv("CHUNK_DATASET_DIR", "./datasets")
    dataset = load_chunk_dataset(output_dir=OUTPUT_DIR)
    sample = dataset["train"][0]
    print(sample["audio"]["path"])
    arr, sr = sf.read(sample["audio"]["path"], dtype="float32")
    duration_sec = len(arr) / sr
    print(f"Audio duration: {duration_sec:.2f}s")
    print(f"Ground truth word count: {len(sample['sentence'].split())}")
    print(f"Ground truth: {sample['sentence']}")
    import numpy as np


    processor, model = load_whisper()          # <-- was missing

    inputs = processor(arr, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)      # note: DEVICE (module constant), not device
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        max_new_tokens=200,
        num_beams=5,
    )
    lead_samples = int(sr * 1.5)
    lead_only = arr[:lead_samples]

    lead_inputs = processor(lead_only, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    lead_input_features = lead_inputs.input_features.to(DEVICE)
    lead_attention_mask = lead_inputs.attention_mask.to(DEVICE)

    lead_ids = model.generate(
    lead_input_features,
    attention_mask=lead_attention_mask,
    max_new_tokens=50,
    num_beams=5,
    )
    print("Transcription of JUST the first 1.5s:", processor.batch_decode(lead_ids, skip_special_tokens=True)[0])
    baseline = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"\nImproved decode: {baseline}")

    print_deliverable()
