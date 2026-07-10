"""
split_chunk_dataset.py
========================
Reads chunks_manifest.csv + chunks/ from input_dir (already produced by
align_and_chunk.py) and writes train/validation/test splits into
output_dir. input_dir and output_dir can be the same path or different.
"""

import os
import shutil
import csv
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _get_split_percentages():
    train_pct = float(os.getenv("TRAIN_PERCENT", 70))
    val_pct = float(os.getenv("VALIDATION_PERCENT", 15))
    test_pct = float(os.getenv("TEST_PERCENT", 15))

    total = train_pct + val_pct + test_pct
    if abs(total - 100) > 1e-6:
        raise ValueError(
            f"TRAIN_PERCENT + VALIDATION_PERCENT + TEST_PERCENT must equal 100, "
            f"got {total} (train={train_pct}, val={val_pct}, test={test_pct})"
        )

    return train_pct / 100, val_pct / 100, test_pct / 100


def _write_split(split_name, rows, output_dir, chunks_dir):
    split_dir = os.path.join(output_dir, split_name)
    audio_dir = os.path.join(split_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)

    csv_path = os.path.join(split_dir, f"{split_name}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file", "transcript"])

        for audio_file, transcript in rows:
            src = os.path.join(chunks_dir, audio_file)
            dst = os.path.join(audio_dir, audio_file)

            if not os.path.exists(src):
                print(f"  [skip] {audio_file}: not found in {chunks_dir}")
                continue

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            writer.writerow([audio_file, transcript])

    print(f"  [{split_name}] {len(rows)} rows -> {csv_path}")


def split_chunk_dataset(input_dir, output_dir=None):
    """
    input_dir  : folder containing chunks_manifest.csv and chunks/
                 (i.e. wherever align_and_chunk.py wrote its output —
                 for you, this is your current directory)
    output_dir : folder to write train/validation/test into.
                 Defaults to input_dir if not given.
    """
    if output_dir is None:
        output_dir = input_dir

    print("=" * 60)
    print("Splitting chunk dataset")
    print("=" * 60)

    manifest_path = os.path.join(input_dir, "chunks_manifest.csv")
    chunks_dir = os.path.join(input_dir, "chunks")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"chunks_manifest.csv not found at {manifest_path}")
    if not os.path.isdir(chunks_dir):
        raise FileNotFoundError(f"chunks/ folder not found at {chunks_dir}")

    seed = int(os.getenv("SEED", 42))
    train_frac, val_frac, test_frac = _get_split_percentages()

    df = pd.read_csv(manifest_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    val_size = int(n * val_frac)
    test_size = int(n * test_frac)

    val_df = df.iloc[:val_size]
    test_df = df.iloc[val_size:val_size + test_size]
    train_df = df.iloc[val_size + test_size:]

    print(f"Total chunks: {n}  (seed={seed})")
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    os.makedirs(output_dir, exist_ok=True)

    _write_split("train", list(train_df.itertuples(
        index=False, name=None)), output_dir, chunks_dir)
    _write_split("validation", list(val_df.itertuples(
        index=False, name=None)), output_dir, chunks_dir)
    _write_split("test", list(test_df.itertuples(
        index=False, name=None)), output_dir, chunks_dir)

    print("Done. Original chunks/ left untouched.")

    return {
        "train_csv": os.path.join(output_dir, "train", "train.csv"),
        "validation_csv": os.path.join(output_dir, "validation", "validation.csv"),
        "test_csv": os.path.join(output_dir, "test", "test.csv"),
    }


if __name__ == "__main__":
    # your case: manifest + chunks/ are in the current directory
    split_chunk_dataset(input_dir=".", output_dir="./datasets")
