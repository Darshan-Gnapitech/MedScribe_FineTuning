"""
load_chunk_dataset.py
========================
Loads the pre-split train/validation/test CSVs produced by
split_chunk_dataset() into a HuggingFace DatasetDict. No shuffling or
splitting happens here anymore — that's already baked into the CSVs.
"""

from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import os


def _load_split(split_name, output_dir):
    split_dir = os.path.join(output_dir, split_name)
    csv_path = os.path.join(split_dir, f"{split_name}.csv")
    audio_dir = os.path.join(split_dir, "audio_files")

    df = pd.read_csv(csv_path)

    df["audio"] = df["audio_file"].apply(
        lambda x: os.path.join(audio_dir, x)
    )

    df.rename(columns={"transcript": "sentence"}, inplace=True)
    df = df[["audio", "sentence"]]

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000, decode=False)
    )

    return dataset


def load_chunk_dataset(output_dir):
    print("=" * 60)
    print("STEP 1: Loading pre-split chunk dataset")
    print("=" * 60)

    dataset = DatasetDict({
        "train": _load_split("train", output_dir),
        "validation": _load_split("validation", output_dir),
        "test": _load_split("test", output_dir),
    })

    print(
        f"train={len(dataset['train'])} "
        f"val={len(dataset['validation'])} "
        f"test={len(dataset['test'])}"
    )

    return dataset


if __name__ == "__main__":
    load_chunk_dataset(output_dir="path/to/output")
