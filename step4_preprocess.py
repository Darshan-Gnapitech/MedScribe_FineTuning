import os
import sys
import time
import shutil
import tempfile
import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict
from transformers import WhisperProcessor

MODEL_PATH = os.getenv("WHISPER_MODEL_NAME", "/home/nisha/whisper-large-v3-hf")

_worker_processor = None
# per-worker counter, resets each process — used only for progress prints
_examples_seen = 0


def get_processor():
    global _worker_processor
    if _worker_processor is None:
        print(
            f"[preprocess][pid={os.getpid()}] loading processor from {MODEL_PATH}")
        _worker_processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        _worker_processor.tokenizer.set_prefix_tokens(
            language="en", task="transcribe")
    return _worker_processor


class MedicalWhisperPreprocessor:
    TARGET_SR = 16_000
    MAX_LABEL_TOKENS = 448

    def __init__(
        self,
        processor: WhisperProcessor,
        audio_column="audio",
        transcript_column="sentence",
        num_proc=None,
        writer_batch_size=1000,
        map_batch_size=32,
    ):
        # Fail fast in the main process rather than deep inside a worker.
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(
                f"[preprocess] MODEL_PATH does not exist: {MODEL_PATH!r}. "
                f"Set WHISPER_MODEL_NAME or populate this path before running."
            )

        self.processor = processor
        self.audio_column = audio_column
        self.transcript_column = transcript_column

        # Raised ceiling: feature extraction + sf.read is CPU/disk-bound and
        # independent per example, so this scales well on a 24-physical/
        # 48-logical-core box. Tune between 24-32 for your workload; leave a
        # couple of cores free for the main process / disk I/O.
        cpu_count = os.cpu_count() or 1
        self.num_proc = num_proc or min(32, cpu_count)

        # smaller writer_batch_size = less RAM held before flush to Arrow,
        # at the cost of more frequent (small) disk writes. 1000 is a
        # reasonable default; drop to 200-500 if you see OOM at millions of rows.
        self.writer_batch_size = writer_batch_size

        # Size of each batch handed to preprocess_batch per map() call.
        # Larger batches amortize Arrow (de)serialization / IPC overhead
        # across num_proc workers; tune alongside num_proc and available RAM.
        self.map_batch_size = map_batch_size

    def preprocess_batch(self, examples):
        """Batched map function: examples is a dict of column -> list of values."""
        global _examples_seen
        processor = get_processor() if self.num_proc > 1 else self.processor

        audio_entries = examples[self.audio_column]
        transcripts = examples[self.transcript_column]

        batch_features = []
        batch_labels = []
        batch_raw_text = []

        for audio_entry, transcript in zip(audio_entries, transcripts):
            audio_path = audio_entry["path"]
            audio, sr = sf.read(audio_path, dtype="float32")

            features = processor.feature_extractor(
                audio, sampling_rate=self.TARGET_SR, return_tensors="np",
            ).input_features[0]

            labels = processor.tokenizer(
                transcript,
                truncation=True,
                max_length=self.MAX_LABEL_TOKENS,
            ).input_ids

            batch_features.append(features)
            batch_labels.append(labels)
            batch_raw_text.append(transcript)

            # audio array goes out of scope right after each iteration —
            # nothing holds onto the raw waveform beyond this loop body
            _examples_seen += 1
            if _examples_seen % 5000 == 0:
                print(
                    f"[preprocess][pid={os.getpid()}] processed {_examples_seen:,} examples in this worker")

        return {
            "input_features": batch_features,
            "labels": batch_labels,
            "raw_text": batch_raw_text,
        }

    def __call__(self, dataset_dict: DatasetDict):
        print("[preprocess] Processing dataset...")
        print(
            f"[preprocess] num_proc={self.num_proc}  writer_batch_size={self.writer_batch_size}  "
            f"map_batch_size={self.map_batch_size}")
        processed = DatasetDict()
        for split, dataset in dataset_dict.items():
            print(f"\n[preprocess] --- {split} ---")
            print(f"[preprocess] examples to process: {len(dataset):,}")
            t0 = time.time()
            processed[split] = dataset.map(
                self.preprocess_batch,
                batched=True,
                batch_size=self.map_batch_size,
                num_proc=self.num_proc,
                remove_columns=dataset.column_names,
                desc=f"Processing {split}",
                writer_batch_size=self.writer_batch_size,
                load_from_cache_file=True,
            )
            elapsed = time.time() - t0
            rate = len(dataset) / elapsed if elapsed > 0 else 0
            print(f"[preprocess] {split} done: {len(dataset):,} examples in {elapsed:.1f}s "
                  f"(~{rate:.1f} examples/sec)")
        print("\n[preprocess] Finished preprocessing all splits.")
        return processed


def _make_fake_dataset_dict(tmp_dir, n_train=12, n_val=4, sr=16_000, seconds=1.0):
    """Generate tiny synthetic wav files + transcripts to smoke-test the pipeline
    without needing a real corpus. Returns a DatasetDict shaped like the real one
    (an `audio` column with a {"path": ...} dict, and a `sentence` column)."""
    audio_dir = os.path.join(tmp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    def build_split(n, prefix):
        paths, sentences = [], []
        for i in range(n):
            # short sine tone at a slightly different frequency per example,
            # just so files aren't byte-identical
            freq = 220 + (i * 10)
            t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
            wave = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            path = os.path.join(audio_dir, f"{prefix}_{i:03d}.wav")
            sf.write(path, wave, sr)
            paths.append(path)
            sentences.append(f"this is fake transcript number {i} for {prefix}")
        return Dataset.from_dict({
            "audio": [{"path": p} for p in paths],
            "sentence": sentences,
        })

    return DatasetDict({
        "train": build_split(n_train, "train"),
        "validation": build_split(n_val, "val"),
    })


def main():
    """Smoke test: builds a tiny synthetic dataset, runs it through
    MedicalWhisperPreprocessor, and sanity-checks the output.

    Usage:
        python step4_preprocess.py                # small num_proc, quick check
        python step4_preprocess.py --num-proc 4    # override worker count
    """
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for step4_preprocess.py")
    parser.add_argument("--num-proc", type=int, default=2,
                         help="Workers to use for this test run (keep small; this is a smoke test, not a benchmark).")
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--n-val", type=int, default=4)
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="whisper_preprocess_smoketest_")
    ok = True
    try:
        print(f"[smoketest] MODEL_PATH={MODEL_PATH}")
        print(f"[smoketest] generating synthetic dataset in {tmp_dir}")
        raw_dataset = _make_fake_dataset_dict(
            tmp_dir, n_train=args.n_train, n_val=args.n_val)

        # Load the processor once in the main process — matches how the
        # real pipeline constructs MedicalWhisperPreprocessor.
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

        preprocessor = MedicalWhisperPreprocessor(
            processor=processor,
            num_proc=args.num_proc,
            writer_batch_size=8,
            map_batch_size=4,
        )

        processed = preprocessor(raw_dataset)

        # --- Sanity checks ---
        for split, expected_n in [("train", args.n_train), ("validation", args.n_val)]:
            ds = processed[split]
            assert len(ds) == expected_n, (
                f"[smoketest] {split}: expected {expected_n} rows, got {len(ds)}")

            assert set(ds.column_names) == {"input_features", "labels", "raw_text"}, (
                f"[smoketest] {split}: unexpected columns {ds.column_names}")

            row0 = ds[0]
            feat = np.array(row0["input_features"])
            assert feat.ndim == 2, f"[smoketest] {split}: expected 2D log-mel features, got shape {feat.shape}"
            assert feat.shape[0] > 0 and feat.shape[1] > 0

            assert isinstance(row0["labels"], list) and len(row0["labels"]) > 0, (
                f"[smoketest] {split}: labels empty or wrong type")

            assert isinstance(row0["raw_text"], str) and len(row0["raw_text"]) > 0

            print(f"[smoketest] {split}: OK — {len(ds)} rows, "
                  f"input_features shape={feat.shape}, sample labels len={len(row0['labels'])}")

        print("\n[smoketest] ALL CHECKS PASSED ✅")

    except Exception as e:
        ok = False
        print(f"\n[smoketest] FAILED ❌ — {type(e).__name__}: {e}")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
