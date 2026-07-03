"""
Step 1: Load Primock_med Dataset
=================================
Downloads Na0s/Primock_med from HuggingFace, then re-splits it:
the original validation/test splits are full, unchunked consultations
(different format than train), so we discard them and carve val/test
directly out of train instead — which is already pre-chunked to
~30s pieces, so everything stays in the same format. No re-chunking
logic needed anywhere in the pipeline anymore.

Split sizes: train 302 / validation 10 / test 10 (from train's
original 322 examples).
"""

from datasets import load_dataset, Audio, DatasetDict
from huggingface_hub import hf_hub_download
import os
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"


REPO_ID = "Na0s/Primock_med"
VAL_SIZE = 10
TEST_SIZE = 10
SEED = 42


def load_primock_med() -> DatasetDict:
    print("=" * 60)
    print("STEP 1: Loading Na0s/Primock_med from HuggingFace")
    print("=" * 60)

    parquet_files = {}
    for split in ("train", "validation", "test"):
        print(f"  Downloading {split} split ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        parquet_files[split] = path
        print(f"    cached at {path}")

    raw = load_dataset("parquet", data_files=parquet_files)
    raw = raw.cast_column("audio", Audio(decode=False))

    print(f"\nOriginal HF splits : train={len(raw['train'])}  "
          f"validation={len(raw['validation'])}  test={len(raw['test'])}")
    print("  -> discarding original validation/test, carving new ones "
          "out of train instead (same pre-chunked ~30s format)")

    shuffled = raw["train"].shuffle(seed=SEED)
    n = len(shuffled)
    assert n >= VAL_SIZE + TEST_SIZE, (
        f"train only has {n} examples, need at least {VAL_SIZE + TEST_SIZE}"
    )

    dataset = DatasetDict({
        "validation": shuffled.select(range(VAL_SIZE)),
        "test": shuffled.select(range(VAL_SIZE, VAL_SIZE + TEST_SIZE)),
        "train": shuffled.select(range(VAL_SIZE + TEST_SIZE, n)),
    })

    print(f"\nNew splits : train={len(dataset['train'])}  "
          f"validation={len(dataset['validation'])}  test={len(dataset['test'])}")
    print(f"Columns    : {dataset['train'].column_names}")

    row = dataset["train"][0]
    print(f"\nfile_name  : {row['file_name']}")
    print(f"sentence   : {row['sentence'][:200]}")

    return dataset


if __name__ == "__main__":
    dataset = load_primock_med()
