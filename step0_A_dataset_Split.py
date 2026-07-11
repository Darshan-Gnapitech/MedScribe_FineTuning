import os
import shutil
import csv
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# rows per manifest read chunk — keeps memory flat regardless of manifest size
CHUNK_READ_SIZE = 100_000


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
    print(
        f"[split] ratios -> train={train_pct}% val={val_pct}% test={test_pct}%")
    return train_pct / 100, val_pct / 100, test_pct / 100


def _load_already_assigned(output_dir):
    """
    Reads only the audio_file column from each split CSV — not the whole
    row — so this stays cheap even at millions of rows (a set of filename
    strings, not a full DataFrame).
    """
    assigned = set()
    for split_name in ("train", "validation", "test"):
        csv_path = os.path.join(output_dir, split_name, f"{split_name}.csv")
        if os.path.exists(csv_path):
            t0 = time.time()
            n_before = len(assigned)
            for chunk in pd.read_csv(csv_path, usecols=["audio_file"], chunksize=CHUNK_READ_SIZE):
                assigned.update(chunk["audio_file"].tolist())
            print(f"[split] read {split_name}.csv -> +{len(assigned) - n_before} ids "
                  f"({time.time() - t0:.2f}s)")
    print(
        f"[split] total already-assigned across all splits: {len(assigned):,}")
    return assigned


def _write_split_append(split_name, rows, output_dir, chunks_dir):
    split_dir = os.path.join(output_dir, split_name)
    audio_dir = os.path.join(split_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)

    csv_path = os.path.join(split_dir, f"{split_name}.csv")
    file_exists = os.path.exists(csv_path)

    t0 = time.time()
    written, skipped = 0, 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["audio_file", "transcript"])
            print(f"[split] creating new {csv_path}")

        for audio_file, transcript in rows:
            src = os.path.join(chunks_dir, audio_file)
            dst = os.path.join(audio_dir, audio_file)

            if not os.path.exists(src):
                skipped += 1
                continue

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            writer.writerow([audio_file, transcript])
            written += 1

            if written % 50_000 == 0:
                print(f"  [{split_name}] ...{written:,} rows written so far")

    print(f"[split] [{split_name}] +{written:,} new rows, {skipped} skipped "
          f"(missing source file) -> {csv_path}  ({time.time() - t0:.2f}s)")


def split_chunk_dataset(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    print("=" * 60)
    print("Splitting chunk dataset (incremental, memory-lean)")
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

    already_assigned = _load_already_assigned(output_dir)

    # Stream the manifest in chunks and keep only unassigned rows —
    # avoids holding the full manifest in memory at once for huge datasets.
    print(
        f"[split] scanning manifest in chunks of {CHUNK_READ_SIZE:,} rows ...")
    t0 = time.time()
    new_chunks = []
    total_seen = 0
    for chunk in pd.read_csv(manifest_path, chunksize=CHUNK_READ_SIZE):
        total_seen += len(chunk)
        filtered = chunk[~chunk["audio_file"].isin(already_assigned)]
        if not filtered.empty:
            new_chunks.append(filtered)
        print(f"  [manifest] scanned {total_seen:,} rows so far, "
              f"{sum(len(c) for c in new_chunks):,} new")

    print(
        f"[split] manifest scan complete ({time.time() - t0:.2f}s), total rows = {total_seen:,}")

    if not new_chunks:
        print("No new chunks to split — all manifest rows already assigned.")
        return {
            "train_csv": os.path.join(output_dir, "train", "train.csv"),
            "validation_csv": os.path.join(output_dir, "validation", "validation.csv"),
            "test_csv": os.path.join(output_dir, "test", "test.csv"),
        }

    new_df = pd.concat(new_chunks, ignore_index=True)
    del new_chunks
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(new_df)
    val_size = int(n * val_frac)
    test_size = int(n * test_frac)

    val_df = new_df.iloc[:val_size]
    test_df = new_df.iloc[val_size:val_size + test_size]
    train_df = new_df.iloc[val_size + test_size:]

    print(f"[split] new chunks: {n:,}  (seed={seed})")
    print(
        f"[split] -> train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    os.makedirs(output_dir, exist_ok=True)

    _write_split_append("train", list(train_df.itertuples(
        index=False, name=None)), output_dir, chunks_dir)
    del train_df
    _write_split_append("validation", list(val_df.itertuples(
        index=False, name=None)), output_dir, chunks_dir)
    del val_df
    _write_split_append("test", list(test_df.itertuples(
        index=False, name=None)), output_dir, chunks_dir)
    del test_df

    print("[split] Done. Original chunks/ left untouched.")

    return {
        "train_csv": os.path.join(output_dir, "train", "train.csv"),
        "validation_csv": os.path.join(output_dir, "validation", "validation.csv"),
        "test_csv": os.path.join(output_dir, "test", "test.csv"),
    }


if __name__ == "__main__":
    split_chunk_dataset(input_dir=".", output_dir="./datasets")
