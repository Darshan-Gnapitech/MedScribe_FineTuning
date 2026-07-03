import io
import numpy as np
import soundfile as sf
from datasets import DatasetDict, Dataset


TARGET_SR = 16_000


def decode_audio(example: dict) -> dict:
    audio_field = example["audio"]
    raw_bytes = audio_field.get("bytes")
    if raw_bytes is not None and len(raw_bytes) > 0:
        arr, sr = sf.read(io.BytesIO(raw_bytes),
                          dtype="float32", always_2d=False)
    elif audio_field.get("path"):
        arr, sr = sf.read(audio_field["path"],
                          dtype="float32", always_2d=False)
    else:
        raise ValueError(
            f"Audio entry has neither bytes nor a valid path: {audio_field}")

    return {"audio_array": arr.tolist(), "sampling_rate": sr}


def decode_dataset(dataset: DatasetDict) -> tuple:
    print("\n" + "=" * 60)
    print("STEP 2: Decoding audio with soundfile (16 kHz)")
    print("=" * 60)

    for split in ("train", "validation", "test"):
        print(f"  Decoding {split} ...")
        dataset[split] = dataset[split].map(decode_audio, writer_batch_size=8)

    # All three splits now come from the same pre-chunked ~30s train
    # data (step1 carves val/test out of train) — nothing left to
    # re-chunk or slice here.

    sample = dataset["train"][0]
    arr = np.array(sample["audio_array"], dtype=np.float32)
    audio_len_sec = len(arr) / TARGET_SR
    print(audio_len_sec)

    print(f"\n  audio shape   : {arr.shape}")
    print(f"  sampling rate : {sample['sampling_rate']} Hz  OK")
    print(f"  ground truth  : {sample['sentence'][:200]}")

    print(f"\nSplit sizes:")
    for split in dataset:
        print(f"  {split:12s}: {len(dataset[split])} examples")

    return dataset, sample, arr
