"""
inspect_dataset.py
====================
Raw inspection of the Na0s/Primock_med dataset — reads directly from
the cached parquet files (same source as step1_load_dataset.py).
Does NOT decode, chunk, slice, or filter anything — this only reads
and reports what's actually stored, so you can confirm whether
transcripts are genuinely truncated in the data or not.

Per split, writes a CSV with every row's:
  file_name, duration_sec, char_count, sentence (FULL, untruncated)

And prints a console summary: row counts, audio/transcript presence,
duration min/max/avg, and how many exceed 30s.

Usage:
    python inspect_dataset.py
"""

import io
import os
import csv
import soundfile as sf
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio

REPO_ID = "Na0s/Primock_med"
OUTPUT_DIR = "./inspection_report"


def load_raw():
    parquet_files = {}
    for split in ("train", "validation", "test"):
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        parquet_files[split] = path

    dataset = load_dataset("parquet", data_files=parquet_files)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def get_duration_sec(audio_field: dict) -> float:
    """Reads audio header only (via soundfile.info) — does not decode
    or modify the audio, just measures actual stored duration."""
    raw_bytes = audio_field.get("bytes") if audio_field else None
    if raw_bytes:
        info = sf.info(io.BytesIO(raw_bytes))
    elif audio_field and audio_field.get("path"):
        info = sf.info(audio_field["path"])
    else:
        return -1.0
    return info.frames / info.samplerate


def inspect_split(dataset_split, split_name: str, output_dir: str):
    n = len(dataset_split)
    rows = []
    missing_audio = 0
    missing_text = 0

    for i in range(n):
        ex = dataset_split[i]
        file_name = ex.get("file_name", f"<missing file_name at row {i}>")
        sentence = ex.get("sentence") or ""
        audio_field = ex.get("audio")

        if not sentence.strip():
            missing_text += 1

        duration = get_duration_sec(audio_field)
        if duration < 0:
            missing_audio += 1

        rows.append({
            "file_name": file_name,
            "duration_sec": round(duration, 3),
            "char_count": len(sentence),
            # full, untruncated — no [:N] slicing anywhere
            "sentence": sentence,
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}_inspection.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name", "duration_sec", "char_count", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    durations = [r["duration_sec"] for r in rows if r["duration_sec"] >= 0]

    print(f"\n[{split_name}]")
    print(f"  total rows          : {n}")
    print(f"  rows with audio     : {n - missing_audio}")
    print(f"  rows with transcript: {n - missing_text}")
    if durations:
        print(f"  duration min/max/avg: "
              f"{min(durations):.2f}s / {max(durations):.2f}s / "
              f"{sum(durations)/len(durations):.2f}s")
        over_30 = sum(d > 30.0 for d in durations)
        print(f"  rows over 30s       : {over_30}")
    print(f"  full detail written -> {out_path}")

    return rows


def main():
    print("=" * 60)
    print("RAW DATASET INSPECTION — Na0s/Primock_med")
    print("(reads parquet directly, does not decode/chunk/filter anything)")
    print("=" * 60)

    dataset = load_raw()

    all_file_names = set()
    duplicate_names = 0
    for split in ("train", "validation", "test"):
        rows = inspect_split(dataset[split], split, OUTPUT_DIR)
        for r in rows:
            if r["file_name"] in all_file_names:
                duplicate_names += 1
            all_file_names.add(r["file_name"])

    total = sum(len(dataset[s]) for s in ("train", "validation", "test"))

    print("\n" + "=" * 60)
    print(f"Total rows across all splits         : {total}")
    print(f"Unique file_names across all splits  : {len(all_file_names)}")
    print(f"Duplicate file_names (same name twice): {duplicate_names}")
    print(f"Full per-row detail in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
