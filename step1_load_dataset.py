from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import os


SEED = 42


def load_chunk_dataset(
    manifest_path,
    chunks_dir,
):
    print("=" * 60)
    print("STEP 1: Loading chunk dataset")
    print("=" * 60)

    df = pd.read_csv(manifest_path)

    df["audio"] = df["audio_file"].apply(
        lambda x: os.path.join(chunks_dir, x)
    )

    df.rename(
        columns={"transcript": "sentence"},
        inplace=True
    )

    df = df[["audio", "sentence"]]

    dataset = Dataset.from_pandas(
        df,
        preserve_index=False
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000,decode=False)
    )

    shuffled = dataset.shuffle(seed=SEED)

    n = len(shuffled)
    val_size = int(n * 0.15)
    test_size = int(n * 0.15)

    dataset = DatasetDict({
        "validation": shuffled.select(range(val_size)),
        "test": shuffled.select(
            range(val_size, val_size + test_size)
        ),
        "train": shuffled.select(
            range(val_size + test_size, n)
        ),
    })

    print(
        f"train={len(dataset['train'])} "
        f"val={len(dataset['validation'])} "
        f"test={len(dataset['test'])}"
    )

    return dataset
