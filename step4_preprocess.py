import os
import sys
import gc
import time
import shutil
import tempfile
import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
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

    # Examples per chunk. This is the hard ceiling on how much processed
    # data (input_features + labels + raw_text) is ever resident at once,
    # regardless of total dataset size. At ~1.7MB/example (measured via
    # pyarrow's memory pool on this pipeline), 20_000 examples caps a
    # single chunk's arrow-side footprint at ~34GB *before* it's written
    # to disk and released -- tune down if that's still too much for your
    # box, tune up if you have RAM to spare and want fewer, larger shards.
    DEFAULT_CHUNK_SIZE = 20_000

    def __init__(
        self,
        processor: WhisperProcessor,
        audio_column="audio",
        transcript_column="sentence",
        num_proc=None,
        writer_batch_size=1000,
        map_batch_size=32,
        chunk_size=None,
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

        cpu_count = os.cpu_count() or 1
        self.num_proc = num_proc or min(32, cpu_count)

        self.writer_batch_size = writer_batch_size
        self.map_batch_size = map_batch_size

        self.chunk_size = chunk_size or int(
            os.getenv("PREPROCESS_CHUNK_SIZE", self.DEFAULT_CHUNK_SIZE))

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

            _examples_seen += 1
            if _examples_seen % 5000 == 0:
                print(
                    f"[preprocess][pid={os.getpid()}] processed {_examples_seen:,} examples in this worker")

        return {
            "input_features": batch_features,
            "labels": batch_labels,
            "raw_text": batch_raw_text,
        }

    def _process_chunk(self, chunk_dataset, chunk_shard_path):
        """
        Runs .map() over a single bounded chunk and writes it straight to
        disk at chunk_shard_path. Nothing from this chunk is returned or
        kept referenced by the caller -- the on-disk shard is the only
        thing that survives past this call.

        Note on why this actually bounds memory even though we never
        diagnosed *why* datasets.map()/pyarrow doesn't release memory
        internally: when num_proc > 1, each .map() call spawns a brand
        new worker pool that is torn down when that call returns. Killing
        those OS processes hands all their memory -- including whatever
        pyarrow was holding onto -- back to the OS unconditionally. So
        chunking bounds memory in two independent ways: (a) each chunk is
        small, and (b) for num_proc>1 the worker processes literally die
        between chunks. Only the main process (which stays alive across
        chunks) needs chunk_size itself to be the safety margin.
        """
        processed_chunk = chunk_dataset.map(
            self.preprocess_batch,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.num_proc,
            remove_columns=chunk_dataset.column_names,
            desc="  chunk",
            writer_batch_size=self.writer_batch_size,
            load_from_cache_file=True,
        )
        processed_chunk.save_to_disk(chunk_shard_path)

        # Drop every reference we hold to this chunk's in-process data
        # before moving on, so the *main* process's own memory (relevant
        # even when num_proc>1, since results still get pulled back into
        # it to build processed_chunk) doesn't carry it into the next
        # iteration.
        del processed_chunk
        del chunk_dataset
        gc.collect()

    def __call__(self, dataset_dict: DatasetDict, output_dir: str):
        """
        Processes each split in bounded chunks of self.chunk_size examples,
        writing each chunk to a temporary shard directory and releasing it
        before starting the next chunk. Once every chunk in a split is
        done, the shards are concatenated (mmap-backed, so this does not
        require loading the split into RAM) and the combined split is
        saved to output_dir/<split>. Temporary chunk shards are deleted
        afterward.

        Returns a DatasetDict loaded via load_from_disk(output_dir), i.e.
        memory-mapped, not held in RAM.
        """
        print("[preprocess] Processing dataset in bounded chunks...")
        print(
            f"[preprocess] num_proc={self.num_proc}  writer_batch_size={self.writer_batch_size}  "
            f"map_batch_size={self.map_batch_size}  chunk_size={self.chunk_size:,}")

        os.makedirs(output_dir, exist_ok=True)
        chunks_root = os.path.join(output_dir, "_chunks_tmp")
        os.makedirs(chunks_root, exist_ok=True)

        for split, dataset in dataset_dict.items():
            split_final_path = os.path.join(output_dir, split)
            split_chunks_dir = os.path.join(chunks_root, split)

            if os.path.isdir(split_final_path):
                print(f"\n[preprocess] --- {split} --- already exists at "
                      f"{split_final_path}, skipping (delete it to reprocess)")
                continue

            os.makedirs(split_chunks_dir, exist_ok=True)

            n = len(dataset)
            n_chunks = (n + self.chunk_size - 1) // self.chunk_size
            print(f"\n[preprocess] --- {split} --- "
                  f"{n:,} examples -> {n_chunks} chunk(s) of up to {self.chunk_size:,}")

            t0 = time.time()
            shard_paths = []
            for chunk_idx in range(n_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, n)
                shard_path = os.path.join(split_chunks_dir, f"chunk_{chunk_idx:05d}")
                shard_paths.append(shard_path)

                if os.path.isdir(shard_path):
                    print(f"[preprocess]   chunk {chunk_idx + 1}/{n_chunks} "
                          f"({start:,}-{end:,}) already on disk, skipping")
                    continue

                print(f"[preprocess]   chunk {chunk_idx + 1}/{n_chunks} "
                      f"({start:,}-{end:,}) processing...")
                ct0 = time.time()
                chunk_dataset = dataset.select(range(start, end))
                self._process_chunk(chunk_dataset, shard_path)
                print(f"[preprocess]   chunk {chunk_idx + 1}/{n_chunks} done "
                      f"in {time.time() - ct0:.1f}s")

            # Reassemble the split from its on-disk shards. load_from_disk
            # is mmap-backed and concatenate_datasets on mmap-backed
            # datasets does not require materializing the data in RAM --
            # it builds a combined table referencing the existing arrow
            # files directly.
            print(f"[preprocess] {split}: concatenating {len(shard_paths)} shard(s) -> {split_final_path}")
            shards = [load_from_disk(p) for p in shard_paths]
            combined = concatenate_datasets(shards)
            combined.save_to_disk(split_final_path)
            del combined, shards
            gc.collect()

            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            print(f"[preprocess] {split} done: {n:,} examples in {elapsed:.1f}s "
                  f"(~{rate:.1f} examples/sec)")

        shutil.rmtree(chunks_root, ignore_errors=True)
        print("\n[preprocess] Finished preprocessing all splits.")

        return DatasetDict({
            split: load_from_disk(os.path.join(output_dir, split))
            for split in dataset_dict.keys()
        })


def _make_fake_dataset_dict(tmp_dir, n_train=12, n_val=4, sr=16_000, seconds=1.0):
    """Generate tiny synthetic wav files + transcripts to smoke-test the pipeline
    without needing a real corpus. Returns a DatasetDict shaped like the real one
    (an `audio` column with a {"path": ...} dict, and a `sentence` column)."""
    audio_dir = os.path.join(tmp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    def build_split(n, prefix):
        paths, sentences = [], []
        for i in range(n):
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
    MedicalWhisperPreprocessor's chunked pipeline, and sanity-checks the output.

    Usage:
        python step4_preprocess.py                        # small num_proc, quick check
        python step4_preprocess.py --num-proc 4            # override worker count
        python step4_preprocess.py --chunk-size 5           # force multiple chunks on a tiny set
    """
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for step4_preprocess.py")
    parser.add_argument("--num-proc", type=int, default=2,
                         help="Workers to use for this test run (keep small; this is a smoke test, not a benchmark).")
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=5,
                         help="Deliberately small so the 12/4-row smoke test still exercises multiple chunks.")
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="whisper_preprocess_smoketest_")
    output_dir = os.path.join(tmp_dir, "processed")
    ok = True
    try:
        print(f"[smoketest] MODEL_PATH={MODEL_PATH}")
        print(f"[smoketest] generating synthetic dataset in {tmp_dir}")
        raw_dataset = _make_fake_dataset_dict(
            tmp_dir, n_train=args.n_train, n_val=args.n_val)

        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

        preprocessor = MedicalWhisperPreprocessor(
            processor=processor,
            num_proc=args.num_proc,
            writer_batch_size=8,
            map_batch_size=4,
            chunk_size=args.chunk_size,
        )

        processed = preprocessor(raw_dataset, output_dir=output_dir)

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
