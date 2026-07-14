"""
step4b_lazy_dataset.py
========================
Lazy (on-the-fly) preprocessing for large datasets (100K+ examples).

Problem this solves
--------------------
step4_preprocess.py (MedicalWhisperPreprocessor) computes and stores the
full log-Mel spectrogram (128 x 3000, ~1.5MB) for EVERY example BEFORE
training starts. For ~632K training rows that's ~950GB of feature data
materialized upfront -> RAM/disk exhaustion and server crashes.

This module instead wraps the raw (audio_path, sentence) pairs in a
standard torch.utils.data.Dataset. Each __getitem__ call reads ONE audio
file and computes ITS features on demand, right when the DataLoader asks
for it. Nothing is precomputed. Combined with DataLoader's num_workers,
several examples are prepared in parallel, but at any moment only
(batch_size x num_workers) examples' features exist in memory --
never the full dataset.

Drop-in compatible with the existing DataCollatorSpeechSeq2SeqWithPadding
in step5_training_config.py -- each item still returns exactly:
    {"input_features": np.ndarray, "labels": List[int], "raw_text": str}

USAGE (in a new main_presplit.py, NOT in the existing main.py):
-----------------------------------------------------------------
    from step4b_lazy_dataset import LazyWhisperDataset

    train_dataset = LazyWhisperDataset(dataset["train"], processor)
    val_dataset   = LazyWhisperDataset(dataset["validation"], processor)

    summary = train(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=config,
        data_collator=data_collator,
    )

NOTE ON num_workers
--------------------
This is exactly where DataLoader(num_workers=N) starts to matter a lot.
Each worker is a separate process that calls __getitem__ independently,
so set training_config.num_workers to something like 4-8 (not 0) to keep
the GPU fed without a single CPU process becoming the bottleneck. Test
with a small subset first to find a value that doesn't itself overload
the server's CPU/RAM.
"""

import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from transformers import WhisperProcessor


class LazyWhisperDataset(Dataset):
    """
    Wraps a HuggingFace DatasetDict split (with "audio" and "sentence"
    columns, audio NOT yet decoded/featurized -- i.e. straight out of
    a step1b-style loader, before step4_preprocess.py would normally run)
    and computes Whisper features on demand, one example at a time.
    """

    TARGET_SR = 16_000
    MAX_LABEL_TOKENS = 448

    def __init__(
        self,
        hf_split_dataset,
        processor: WhisperProcessor,
        audio_column: str = "audio",
        transcript_column: str = "sentence",
    ):
        self.dataset = hf_split_dataset
        self.processor = processor
        self.audio_column = audio_column
        self.transcript_column = transcript_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio_path = example[self.audio_column]["path"]
        text = example[self.transcript_column]

        # Read raw audio bytes for ONLY this one file
        audio, sr = sf.read(audio_path, dtype="float32")

        # Compute log-Mel spectrogram for ONLY this one file
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors="np",
            padding="max_length",
            truncation=True,
        ).input_features[0]

        label_ids = self.processor.tokenizer(
            text,
            max_length=self.MAX_LABEL_TOKENS,
            truncation=True,
        ).input_ids

        return {
            "input_features": features,
            "labels": label_ids,
            "raw_text": text,
        }


if __name__ == "__main__":
    print("step4b_lazy_dataset module loaded. "
          "Import LazyWhisperDataset and wrap your train/val splits with it "
          "instead of calling preprocess_dataset() from step4_preprocess.py.")
