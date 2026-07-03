"""
Step 5: Preprocess the Dataset
================================
Converts raw audio arrays + corrected transcripts into model-ready tensors.

Input columns expected from Step 2:
    audio      : dict {"array": np.float32 array, "sampling_rate": int}
    transcript : str  (doctor-corrected ground truth)

Output columns (replaces all raw columns):
    input_features : log-Mel spectrogram, shape (80, 3000)
    labels         : token ids, padding replaced with -100
"""

from datasets import DatasetDict, Dataset
from datasets import DatasetDict
from transformers import WhisperProcessor
from datasets import Dataset
from datasets import disable_caching
disable_caching()
"""
Step 5: Preprocess the Dataset
================================
"""


class MedicalWhisperPreprocessor:

    TARGET_SR = 16_000
    MAX_LABEL_TOKENS = 448

    def __init__(
        self,
        processor: WhisperProcessor,
        audio_column: str = "audio_array",
        transcript_column: str = "sentence",
    ):
        self.processor = processor
        self.audio_column = audio_column
        self.transcript_column = transcript_column

    def _process_split(self, split_dataset, split_name: str) -> Dataset:
        n = len(split_dataset)

        def gen():
            for i in range(n):
                if i % 100 == 0:
                    print(f"    {split_name}: {i}/{n}")

                example = split_dataset[i]
                audio = example[self.audio_column]
                text = example[self.transcript_column]
                audio = audio if isinstance(audio, list) else audio.tolist()

                features = self.processor.feature_extractor(
                    audio,
                    sampling_rate=self.TARGET_SR,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                ).input_features[0]        # keep as numpy — don't .tolist() it

                label_ids = self.processor.tokenizer(
                    text,
                    max_length=self.MAX_LABEL_TOKENS,
                    truncation=True,
                ).input_ids                 # no padding — collator handles it

                yield {
                    "input_features": features,
                    "labels": label_ids,
                    "raw_text": text,
                }

        return Dataset.from_generator(gen, writer_batch_size=8)

    def __call__(self, dataset_dict: DatasetDict, batch_size: int = 8) -> DatasetDict:
        print("[Preprocess] Processing splits (audio -> log-Mel, text -> tokens) ...")
        out = {}
        for split in dataset_dict:
            print(f"  Processing {split} ...")
            out[split] = self._process_split(dataset_dict[split], split)

        dataset_dict = DatasetDict(out)

        print("[Preprocess] Done.")
        for split, ds in dataset_dict.items():
            print(f"  {split:12s}: {len(ds):,} examples")
        return dataset_dict


def preprocess_dataset(
    dataset: DatasetDict,
    processor: WhisperProcessor,
    skip_preprocessing: bool = False,
) -> DatasetDict:
    if skip_preprocessing:
        print("\n-- Step 5: Skipped (data already pre-processed) --")
        return dataset

    print("\n-- Step 5: Preprocessing dataset --")
    return MedicalWhisperPreprocessor(processor)(dataset)


if __name__ == "__main__":
    import numpy as np
    from datasets import Dataset, DatasetDict
    from transformers import WhisperProcessor

    MODEL_NAME  = "openai/whisper-small"
    SAMPLE_TEXT = "Patient was prescribed 500 mg amoxicillin twice daily."
    SR          = 16_000

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="English", task="transcribe"
    )

    def make_split(n):
        silence = np.zeros(SR * 3, dtype=np.float32)
        feats = processor.feature_extractor(
            [silence] * n, sampling_rate=SR,
            return_tensors="np", padding="max_length", truncation=True,
        ).input_features
        tok = processor.tokenizer(
            [SAMPLE_TEXT] * n, max_length=448, truncation=True, padding="max_length"
        ).input_ids
        pad_id = processor.tokenizer.pad_token_id
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in tok]
        return {"input_features": [feats[i] for i in range(n)], "labels": labels}

    dataset = DatasetDict({
        "train"     : Dataset.from_dict(make_split(6)),
        "validation": Dataset.from_dict(make_split(3)),
        "test"      : Dataset.from_dict(make_split(3)),
    })

    processed = preprocess_dataset(dataset, processor, skip_preprocessing=True)
    print("Step 5 standalone test complete.")
    print(f"  columns: {processed['train'].column_names}")