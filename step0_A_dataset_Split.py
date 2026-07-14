import os

import shutil

import csv

import time

import errno

import pandas as pd

from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor, as_completed
 
load_dotenv()
 
# rows per manifest read chunk — keeps memory flat regardless of manifest size

CHUNK_READ_SIZE = 100_000
 
# how to place audio files into each split's audio_files/ dir:

#   "hardlink" (default) - same inode, zero copy cost, requires same filesystem

#   "symlink"             - pointer to original, works across filesystems, but

#                           breaks if the source chunks/ dir is ever moved/deleted

#   "copy"                - real duplicate, needed only if the destination is

#                           genuinely a separate export/filesystem boundary

SPLIT_LINK_MODE = os.getenv("SPLIT_LINK_MODE", "hardlink")
 
# thread pool size for file placement (I/O-bound, so oversubscribing cores is fine)

SPLIT_MAX_WORKERS = int(os.getenv("SPLIT_MAX_WORKERS", "0")) or min(64, (os.cpu_count() or 8) * 2)
 
 
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
 
 
def _link_or_copy(src, dst, mode):

    """

    Places src at dst using the cheapest method available.

    Falls back to a real copy if a hardlink can't cross a filesystem boundary.

    Returns the method actually used, for diagnostics.

    """

    if mode == "hardlink":

        try:

            os.link(src, dst)

            return "hardlink"
        except FileExistsError:
            return "hardlink(race_winner_was_another_thread)"

        except OSError as e:

            if e.errno != errno.EXDEV:

                raise

            shutil.copy2(src, dst)

            return "copy(fallback:exdev)"

    elif mode == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst)
            return "symlink"
        except FileExistsError:
            return "symlink(race_winner_was_another_thread)"
        except OSError:
            shutil.copy2(src,dst)
            return "copy(fallback:hardlink_unsupported)"

    else:

        shutil.copy2(src, dst)

        return "copy"
 
 
def _place_file(row, chunks_dir, audio_dir, mode):

    """

    Worker function run in the thread pool. Does the filesystem work only —

    no CSV writing here, since csv.writer is not thread-safe.

    """

    audio_file, transcript = row

    src = os.path.join(chunks_dir, audio_file)

    dst = os.path.join(audio_dir, audio_file)
 
    if not os.path.exists(src):

        return (audio_file, transcript, "skipped", None)
 
    if not os.path.exists(dst):

        try:

            method = _link_or_copy(src, dst, mode)

        except OSError as e:

            return (audio_file, transcript, "error", str(e))

    else:

        method = "already_present"
 
    return (audio_file, transcript, "ok", method)
 
 
def _write_split_append(split_name, rows, output_dir, chunks_dir,

                         max_workers=SPLIT_MAX_WORKERS, link_mode=SPLIT_LINK_MODE):

    split_dir = os.path.join(output_dir, split_name)

    audio_dir = os.path.join(split_dir, "audio_files")

    os.makedirs(audio_dir, exist_ok=True)
 
    csv_path = os.path.join(split_dir, f"{split_name}.csv")

    file_exists = os.path.exists(csv_path)
 
    n_rows = len(rows)

    if n_rows == 0:

        print(f"[split] [{split_name}] nothing to place, skipping")

        return
 
    print(f"[split] [{split_name}] placing {n_rows:,} audio files "

          f"with {max_workers} workers (mode={link_mode}) ...")
 
    t0 = time.time()

    results = [None] * n_rows

    method_counts = {}
 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {

            executor.submit(_place_file, row, chunks_dir, audio_dir, link_mode): i

            for i, row in enumerate(rows)

        }

        done_count = 0

        for future in as_completed(futures):

            i = futures[future]

            result = future.result()

            results[i] = result

            method_counts[result[3]] = method_counts.get(result[3], 0) + 1

            done_count += 1

            if done_count % 50_000 == 0:

                print(f"  [{split_name}] ...{done_count:,} files placed so far")
 
    place_elapsed = time.time() - t0

    print(f"[split] [{split_name}] file placement done in {place_elapsed:.2f}s "

          f"({method_counts})")
 
    # Serial CSV write, in original row order — avoids concurrent-write

    # issues and keeps output deterministic given the same seed.

    t1 = time.time()

    written, skipped = 0, 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow(["audio_file", "transcript"])

            print(f"[split] creating new {csv_path}")
 
        for audio_file, transcript, status, detail in results:

            if status == "skipped":

                skipped += 1

                continue

            if status == "error":

                print(f"[split] [{split_name}] WARNING: {audio_file} -> {detail}")

                skipped += 1

                continue

            writer.writerow([audio_file, transcript])

            written += 1
 
    print(f"[split] [{split_name}] +{written:,} new rows, {skipped} skipped "

          f"(missing source file) -> {csv_path}  "

          f"(place={place_elapsed:.2f}s, csv_write={time.time() - t1:.2f}s)")
 
 
def split_chunk_dataset(input_dir, output_dir=None):

    if output_dir is None:

        output_dir = input_dir
 
    print("=" * 60)

    print("Splitting chunk dataset (incremental, memory-lean, parallel placement)")

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
 
