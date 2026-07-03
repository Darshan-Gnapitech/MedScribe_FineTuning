"""
build_dataset_csv.py
=====================
Reads all *_expected_text.txt files from a transcripts folder and pairs
each one with its matching audio file, producing a 2-column CSV:
    audio_file, expected_text

Assumes naming convention:
    transcripts/conv_035_expected_text.txt  <->  audio/conv_035.mp3

Usage:
    python build_dataset_csv.py --transcripts ./transcripts --audio ./audio --out dataset.csv
"""

import os
import csv
import glob
import argparse


def build_csv(transcript_dir: str, audio_dir: str, output_csv: str,
               audio_ext: str = ".mp3", suffix: str = "_expected_text.txt") -> None:

    pattern = os.path.join(transcript_dir, f"*{suffix}")
    txt_files = sorted(glob.glob(pattern))

    if not txt_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found.")

    rows = []
    missing_audio = []

    for txt_path in txt_files:
        fname = os.path.basename(txt_path)
        conv_id = fname[:-len(suffix)]          # "conv_035_expected_text.txt" -> "conv_035"

        audio_filename = conv_id + audio_ext
        audio_path = os.path.join(audio_dir, audio_filename)

        if not os.path.exists(audio_path):
            missing_audio.append(audio_filename)
            continue   # skip rows with no matching audio

        with open(txt_path, "r", encoding="utf-8") as f:
            expected_text = f.read().strip()

        rows.append({"audio_file": audio_filename, "expected_text": expected_text})

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_file", "expected_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows -> {output_csv}")
    if missing_audio:
        print(f"[warn] {len(missing_audio)} transcript(s) had no matching audio file, skipped:")
        for m in missing_audio[:10]:
            print(f"   - {m}")
        if len(missing_audio) > 10:
            print(f"   ... and {len(missing_audio) - 10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", required=True, help="Folder containing *_expected_text.txt files")
    parser.add_argument("--audio", required=True, help="Folder containing the audio files")
    parser.add_argument("--out", default="dataset.csv", help="Output CSV path")
    parser.add_argument("--audio_ext", default=".mp3", help="Audio file extension (default: .mp3)")
    args = parser.parse_args()

    build_csv(args.transcripts, args.audio, args.out, audio_ext=args.audio_ext)