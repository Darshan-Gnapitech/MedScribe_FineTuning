import os
import time
import soundfile as sf

from datasets import DatasetDict
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
        writer_batch_size=100,
    ):
        self.processor = processor
        self.audio_column = audio_column
        self.transcript_column = transcript_column
        self.num_proc = num_proc or min(2, os.cpu_count() or 1)
        # smaller writer_batch_size = less RAM held before flush to Arrow,
        # at the cost of more frequent (small) disk writes. 1000 is a
        # reasonable default; drop to 200-500 if you see OOM at millions of rows.
        self.writer_batch_size = writer_batch_size

    def preprocess_example(self, example):
        global _examples_seen
        processor = get_processor() if self.num_proc > 1 else self.processor

        audio_path = example[self.audio_column]["path"]
        audio, sr = sf.read(audio_path, dtype="float32")

        features = processor.feature_extractor(
            audio, sampling_rate=self.TARGET_SR, return_tensors="np",
        ).input_features[0]

        labels = processor.tokenizer(
            example[self.transcript_column],
            truncation=True,
            max_length=self.MAX_LABEL_TOKENS,
        ).input_ids

        _examples_seen += 1
        if _examples_seen % 5000 == 0:
            print(
                f"[preprocess][pid={os.getpid()}] processed {_examples_seen:,} examples in this worker")

        # audio array goes out of scope right after this return —
        # nothing holds onto the raw waveform beyond this function call
        return {
            "input_features": features,
            "labels": labels,
            "raw_text": example[self.transcript_column],
        }

    def __call__(self, dataset_dict: DatasetDict):
        print("[preprocess] Processing dataset...")
        print(
            f"[preprocess] num_proc={self.num_proc}  writer_batch_size={self.writer_batch_size}")

        processed = DatasetDict()

        for split, dataset in dataset_dict.items():
            print(f"\n[preprocess] --- {split} ---")
            print(f"[preprocess] examples to process: {len(dataset):,}")
            t0 = time.time()

            processed[split] = dataset.map(
                self.preprocess_example,
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
