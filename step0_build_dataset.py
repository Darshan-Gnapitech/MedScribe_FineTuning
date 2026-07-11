"""
align_and_chunk.py
====================
Chunks long-form audio (e.g. 40-min doctor-patient recordings) into
<=30s pieces aligned to their ground-truth transcript, using:

  0. ffmpeg             -> converts ANY input audio format (mp3, m4a,
                           flac, etc.) to a standardized 16kHz mono WAV
                           before anything else touches it
  1. Silero VAD        -> finds natural silence/pause boundaries in the audio
  2. torchaudio MMS_FA  -> wav2vec2-based CTC forced alignment, gives a
                           (start, end) timestamp for every word in the
                           ground-truth transcript against the audio

Chunk boundaries are chosen at word gaps that fall inside a VAD-detected
silence, so cuts land on natural pauses instead of mid-word/mid-sentence.

INPUT LAYOUT (expected)
-----------------------
input_dir/
    patient_001.mp3       <- any audio extension works: mp3, wav, m4a, flac...
    patient_001.txt       <- flat ground-truth paragraph, verbatim
    patient_002.wav
    patient_002.txt
    ...

OUTPUT
------
output_dir/
    converted_wavs/
        patient_001.wav        <- standardized 16kHz mono, written by ffmpeg
        patient_002.wav
    chunks/
        patient_001_chunk000.wav
        patient_001_chunk001.wav
        ...
    chunks_manifest.csv              <- columns: audio_file,transcript
    chunking_validation_report.csv   <- per-file chunking QA report

REQUIREMENTS
------------
pip install torch torchaudio soundfile
# torchaudio >= 2.1 required for torchaudio.pipelines.MMS_FA
# Silero VAD is pulled via torch.hub (needs internet access on first run,
# then it's cached under ~/.cache/torch/hub)
# ffmpeg must be installed and on PATH (used for universal audio conversion):
#   Windows : winget install ffmpeg
#   Linux   : conda install -c conda-forge ffmpeg   (no-sudo option)
#   Verify  : ffmpeg -version

USAGE
-----
python align_and_chunk.py --input_dir /path/to/raw --output_dir /path/to/out
"""

from num2words import num2words
import pandas as pd
import argparse
import csv
import os
import re
import shutil
import subprocess

import torch
import torchaudio
import soundfile as sf
import numpy as np
import threading 
import itertools
import time 
import socket 
from pathlib import Path
TARGET_SR = 16_000
MAX_CHUNK_SEC = 28.0          # keep buffer under Whisper's 30s window
MIN_CHUNK_SEC = 2.0           # discard degenerate tiny leftover chunks
GAP_SEARCH_WINDOW_SEC = 3.0   # how far to look for a nearby VAD silence gap
DOWNLOAD_TIMEOUT=300
SPINNER_INTERVAL=0.15

TORCH_HUB=Path.home()/".cache"/"torch"/"hub"
SILERO_CACHE=next(TORCH_HUB.glob("snakers4_silero-vad*"),None)
MMS_CACHE=TORCH_HUB/"checkpoints"/"model.pt"
# =============================================================================
# 0. Universal audio -> standardized WAV conversion (ffmpeg)
# =============================================================================

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac",
                    ".ogg", ".aac", ".wma", ".opus")

class Spinner:

    def __init__(self, text):

        self.text = text

        self.running = False

        self.thread = None

    def start(self):

        self.running = True

        def run():

            for c in itertools.cycle("|/-\\"):

                if not self.running:

                    break

                print(f"\r{self.text} {c}", end="", flush=True)

                time.sleep(SPINNER_INTERVAL)

        self.thread = threading.Thread(target=run)

        self.thread.daemon = True

        self.thread.start()

    def stop(self, msg="Done"):

        self.running = False

        if self.thread:

            self.thread.join()

        print(f"\r{msg}{' '*40}")


def replace_numbers(text):
    def repl(match):
        return num2words(int(match.group(0)))
    return re.sub(r'\d+', repl, text)


def find_pairs_from_csv(csv_path: str, converted_dir: str) -> list:
    """
    Supports both CSV and XLSX manifests.

    Required columns:
        audio_path
        transcript

    Optional columns:
        conversation_id  -> if missing (as a column, or blank/NaN on a
                             given row), falls back to a 1-based sequential
                             id (1, 2, 3, ...) so the script still works on
                             manifests that don't track a conversation id.
                             This id is only used for [skip]/[error] log
                             messages -- it does NOT affect file naming.

    File naming: the converted wav (and every chunk derived from it) keeps
    the source audio file's own name, e.g. 1.mp3 -> 1.wav -> 1_chunk000.wav,
    1_chunk001.wav, ... 2.mp3 -> 2.wav -> 2_chunk000.wav, ...
    """
    manifest_dir = os.path.dirname(os.path.abspath(csv_path))
    check_ffmpeg_available()
    os.makedirs(converted_dir, exist_ok=True)

    ext = os.path.splitext(csv_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(csv_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(csv_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Use .csv or .xlsx"
        )

    has_conv_id_col = "conversation_id" in df.columns

    # Zero-pad width scales with the number of rows (minimum 3 digits), so
    # ids stay sortable/aligned as conv_001, conv_002, ... conv_123, etc.
    id_width = max(3, len(str(len(df))))

    def pad_id(raw_id: str) -> str:
        return raw_id.zfill(id_width) if raw_id.isdigit() else raw_id

    pairs = []

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        raw_conv_id = str(row["conversation_id"]
                          ).strip() if has_conv_id_col else ""
        if raw_conv_id and raw_conv_id.lower() != "nan":
            conv_id = pad_id(raw_conv_id)
        else:
            # No conversation_id column, or blank/NaN for this row ->
            # fall back to sequential numbering (001, 002, 003, ...).
            conv_id = pad_id(str(row_idx))

        src_path = str(row["audio_path"]).strip()

        if not os.path.isabs(src_path):
            src_path = os.path.join(manifest_dir, src_path)

        src_path = os.path.normpath(src_path)
        transcript = str(row["transcript"]).strip()

        if not transcript or transcript.lower() == "nan":
            print(
                f"  [skip] : empty transcript in column 'transcript'"
            )
            continue

        if not os.path.exists(src_path):
            print(
                f"  [skip] {conv_id}: audio file not found at '{src_path}'"
            )
            continue

        # Name the converted wav (and everything derived from it, i.e. every
        # chunk) after the source audio file itself, e.g. 1.mp3 -> 1.wav ->
        # 1_chunk000.wav, 1_chunk001.wav, ... rather than a conv_{id} prefix.
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        wav_path = os.path.join(converted_dir, base_name + ".wav")

        if os.path.exists(wav_path):
            print(
                f"  [skip-convert] {os.path.basename(wav_path)} already exists"
            )
        else:
            print(
                f"  [convert] {src_path} -> {os.path.basename(wav_path)}"
            )
            try:
                convert_to_wav(src_path, wav_path)
            except RuntimeError as e:
                print(f"  [error] {conv_id}: {e}")
                continue

        pairs.append((base_name, wav_path, transcript))

    return pairs


def check_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. It's required to convert audio "
            "files to WAV. Install it with:\n"
            "  Windows : winget install ffmpeg\n"
            "  Linux   : conda install -c conda-forge ffmpeg\n"
            "Then re-open your terminal and re-run this script."
        )


def convert_to_wav(src_path: str, dst_path: str):
    """
    Converts any input audio file to a standardized 16kHz mono WAV using
    ffmpeg. Fails loudly (raises) rather than silently producing bad output,
    so a broken/corrupt source file is caught immediately instead of causing
    a confusing failure later in VAD/alignment.
    """
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite dst_path if it already exists
        "-i", src_path,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-loglevel", "error",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to convert {src_path}:\n{result.stderr}"
        )


def find_audio_transcript_pairs(input_dir: str, converted_dir: str) -> list:
    """
    Scans input_dir for any supported audio file that has a matching .txt
    transcript (same base filename), converts each audio file to a
    standardized WAV in converted_dir, and returns a list of
    (base_name, wav_path, transcript_path) tuples ready for processing.
    """
    check_ffmpeg_available()
    os.makedirs(converted_dir, exist_ok=True)

    all_files = os.listdir(input_dir)
    audio_files = sorted(
        f for f in all_files if f.lower().endswith(AUDIO_EXTENSIONS)
    )

    pairs = []
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        src_path = os.path.join(input_dir, audio_file)
        transcript_path = os.path.join(input_dir, base_name + ".txt")

        if not os.path.exists(transcript_path):
            print(f"  [skip] {base_name}: no matching .txt transcript found")
            continue

        wav_path = os.path.join(converted_dir, base_name + ".wav")
        print(f"  [convert] {audio_file} -> {os.path.basename(wav_path)}")
        try:
            convert_to_wav(src_path, wav_path)
        except RuntimeError as e:
            print(f"  [error] {base_name}: {e}")
            continue

        pairs.append((base_name, wav_path, transcript_path))

    return pairs


# =============================================================================
# 1. Audio loading
# =============================================================================

def load_audio(path: str) -> torch.Tensor:
    """
    Reads a WAV file using soundfile (no external backend/codec dependency).
    Files reaching this function have already been standardized to 16kHz
    mono WAV by convert_to_wav(), so the resample/mono-mixdown logic below
    is only a safety net in case a raw .wav was fed in directly.
    """
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = torch.from_numpy(np.asarray(arr, dtype=np.float32))

    if wav.ndim > 1:
        wav = wav.mean(dim=-1)  # mixdown to mono if somehow stereo

    if sr != TARGET_SR:
        wav = wav.unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.squeeze(0)

    return wav


# =============================================================================
# 2. Silero VAD -> silence gap boundaries
# =============================================================================

def load_vad():
    if next(TORCH_HUB.glob("snakers4_silero-vad*"), None):
        print("✓ Silero VAD found in cache.")
        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
            force_reload=False,
            onnx=False,
        )
        return model, utils[0]
    print("Silero VAD not found in cache.")
    print("Downloading Silero VAD...")
    spinner = Spinner("Downloading")
    spinner.start()
    result = {}
    error = {}
    def worker():
        try:
            model, utils = torch.hub.load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    trust_repo=True,
                    force_reload=False,
                    onnx=False,
                    )
            result["model"] = model
            result["utils"] = utils
        except Exception as e:
            error["e"] = e
    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=DOWNLOAD_TIMEOUT)
    spinner.stop()
    if t.is_alive():
        raise RuntimeError(
            f"""

Silero VAD download timed out after {DOWNLOAD_TIMEOUT} seconds.
Possible reasons:
• Internet connection unavailable
• GitHub temporarily unreachable
• Firewall/proxy blocking GitHub
Please connect to the internet and try again.
"""     )
    if error:
        raise RuntimeError(f"Failed to download Silero VAD:\n{error['e']}")
    print("✓ Silero VAD downloaded successfully.")
    return result["model"], result["utils"][0]

def get_silence_gaps(wav: torch.Tensor, vad_model, get_speech_timestamps) -> list:
    """Returns list of (gap_start_sec, gap_end_sec) between detected speech regions."""
    speech_segments = get_speech_timestamps(
        wav, vad_model, sampling_rate=TARGET_SR, return_seconds=True
    )
    gaps = []
    for i in range(len(speech_segments) - 1):
        gaps.append((speech_segments[i]["end"],
                    speech_segments[i + 1]["start"]))
    return gaps


# =============================================================================
# 3. wav2vec2 forced alignment (torchaudio MMS_FA)
# =============================================================================

import torch

import torchaudio
import torchaudio.pipelines._wav2vec2.utils as utils
mms_fa="/root/models/mms-fa/model.pt"
def load_aligner(device):
    try:
        print("Loading local MMS_FA model...")
        original = utils.load_state_dict_from_url
        def local_loader(url, *args, **kwargs):
            print(f"Loading MMS_FA from {mms_fa}")
            return torch.load(mms_fa, map_location=device)
        # Create MMS_FA model architecture
        try:
            utils.load_state_dict_from_url = local_loader
            bundle = torchaudio.pipelines.MMS_FA
            model = bundle.get_model()
        finally:
            utils.load_state_dict_from_url = original
        # Load your .pt weights
        # Load weights
        model = model.to(device)
        model.eval()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        print("✓ MMS_FA loaded from local .pt")
        return model, tokenizer, aligner
    except Exception as e:
        raise RuntimeError(f"Failed to load local MMS_FA model:\n{e}")
 


def clean_transcript(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_emission_windowed(wav, model, device, window_sec=30.0, overlap_sec=1.0):
    total_samples = wav.shape[0]
    window_samples = int(window_sec * TARGET_SR)
    overlap_samples = int(overlap_sec * TARGET_SR)
    step_samples = max(1, window_samples - overlap_samples)

    total_windows = max(1, (total_samples // step_samples) + 1)
    emissions = []
    start = 0
    window_idx = 0
    with torch.inference_mode():
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            window = wav[start:end].unsqueeze(0).to(device)
            MIN_WINDOW_SAMPLES = 8000
            if window.shape[-1] < MIN_WINDOW_SAMPLES:
                break
            window_idx += 1
            if window_idx == 1 or window_idx % 10 == 0 or end >= total_samples:
                print(f"    [align] window {window_idx}/{total_windows} "
                    f"({window.shape[-1] / TARGET_SR:.1f}s)")
            emission, _ = model(window)
            frames_per_sample = emission.shape[1] / window.shape[1]
            if end < total_samples:
                drop_frames = int(overlap_samples * frames_per_sample)
                emission = emission[:, : emission.shape[1] - drop_frames, :]
            emissions.append(emission[0])
            start += step_samples

    return torch.cat(emissions, dim=0)         # (total_frames, vocab)


def force_align(wav: torch.Tensor, transcript: str, model, tokenizer, aligner, device: str):
    """
    Returns list of (word, start_sec, end_sec) for every word in transcript,
    aligned against wav using CTC forced alignment.
    """
    transcript = replace_numbers(transcript)
    words = clean_transcript(transcript).split()
    if not words:
        return []

    emission = compute_emission_windowed(wav, model, device)

    num_frames = emission.shape[0]
    audio_dur_sec = wav.shape[0] / TARGET_SR
    sec_per_frame = audio_dur_sec / num_frames

    token_spans = aligner(emission, tokenizer(words))

    word_times = []
    for word, spans in zip(words, token_spans):
        start_sec = spans[0].start * sec_per_frame
        end_sec = spans[-1].end * sec_per_frame
        word_times.append((word, start_sec, end_sec))
    return word_times


# =============================================================================
# 4. Merge VAD gaps + word timestamps -> chunk boundaries
# =============================================================================

def nearest_gap_midpoint(gaps: list, target_time: float, window: float = GAP_SEARCH_WINDOW_SEC):
    best_mid, best_dist = None, window
    for g_start, g_end in gaps:
        mid = (g_start + g_end) / 2
        dist = abs(mid - target_time)
        if dist < best_dist:
            best_mid, best_dist = mid, dist
    return best_mid


def build_chunks(word_times: list, silence_gaps: list, max_chunk_sec: float = MAX_CHUNK_SEC):
    """
    Greedily groups consecutive words into chunks under max_chunk_sec.
    When a chunk would exceed the limit, looks for a nearby VAD silence gap
    to cut on instead of cutting at an arbitrary word boundary.
    """
    chunks = []
    i, n = 0, len(word_times)

    while i < n:
        chunk_start_time = word_times[i][1]
        j = i
        while j < n and (word_times[j][2] - chunk_start_time) <= max_chunk_sec:
            j += 1

        if j >= n:
            cut_idx = n
        else:
            target_time = word_times[j - 1][2]
            gap_mid = nearest_gap_midpoint(silence_gaps, target_time)
            if gap_mid is not None:
                cut_idx = j
                while cut_idx > i + 1 and word_times[cut_idx - 1][2] > gap_mid:
                    cut_idx -= 1
            else:
                cut_idx = j  # no nearby pause found; hard cut at the length limit

        chunk_words = word_times[i:cut_idx]
        if chunk_words and (chunk_words[-1][2] - chunk_words[0][1]) >= MIN_CHUNK_SEC:
            chunks.append(chunk_words)
        i = cut_idx if cut_idx > i else i + 1  # safety: always advance

    # Guarantee at least one chunk for any file that has a transcript at
    # all. Without this, a short recording (e.g. under MIN_CHUNK_SEC, or
    # generally under the max_chunk_sec limit) whose single natural chunk
    # gets filtered above would silently produce ZERO chunks -> the file
    # never gets a chunk000, never lands in chunks/, and never appears in
    # chunks_manifest.csv. Short files must still surface as one chunk.
    if not chunks and n > 0:
        chunks.append(word_times[:])

    return chunks


# =============================================================================
# 4b. Validate chunking quality against ground truth
# =============================================================================

def assess_chunking(
    word_times: list,
    chunks: list,
    silence_gaps: list,
    max_chunk_sec: float = MAX_CHUNK_SEC,
    pause_tolerance_sec: float = 0.3,
) -> dict:
    """
    Checks the chunking output against the ground-truth word list.
    Does NOT judge transcription quality — only whether the chunking
    mechanics behaved correctly:

      - word coverage   : every ground-truth word ended up in exactly
                           one chunk, in the original order (no drops,
                           no duplicates, no reordering)
      - duration compliance : no chunk exceeds max_chunk_sec
      - pause alignment : did each cut point land inside/near an actual
                           VAD silence gap, or was it a hard mid-speech cut
    """
    ground_truth_words = [w for w, _, _ in word_times]
    chunked_words = [w for chunk in chunks for w, _, _ in chunk]

    dropped = len(ground_truth_words) - len(chunked_words)
    order_preserved = chunked_words == ground_truth_words
    coverage_pct = (
        round(100 * len(chunked_words) / len(ground_truth_words), 2)
        if ground_truth_words else 0.0
    )

    durations = [chunk[-1][2] - chunk[0][1] for chunk in chunks]
    over_limit = [d for d in durations if d > max_chunk_sec]

    # For every internal cut point (between chunk i and chunk i+1), check
    # whether the cut landed inside/near a VAD-detected silence gap.
    cuts_on_pause = 0
    total_internal_cuts = max(0, len(chunks) - 1)
    for i in range(total_internal_cuts):
        cut_time = chunks[i][-1][2]        # end of previous chunk's last word
        next_start = chunks[i + 1][0][1]    # start of next chunk's first word
        landed_on_pause = any(
            (g_start - pause_tolerance_sec) <= cut_time <= (g_end + pause_tolerance_sec)
            or (g_start - pause_tolerance_sec) <= next_start <= (g_end + pause_tolerance_sec)
            for g_start, g_end in silence_gaps
        )
        if landed_on_pause:
            cuts_on_pause += 1

    pause_alignment_pct = (
        round(100 * cuts_on_pause / total_internal_cuts, 2)
        if total_internal_cuts else 100.0  # single chunk, no cuts to judge
    )

    passed = (
        dropped == 0
        and order_preserved
        and len(over_limit) == 0
    )

    return {
        "passed": passed,
        "total_words": len(ground_truth_words),
        "chunked_words": len(chunked_words),
        "dropped_words": dropped,
        "order_preserved": order_preserved,
        "word_coverage_pct": coverage_pct,
        "num_chunks": len(chunks),
        "chunks_over_limit": len(over_limit),
        "longest_chunk_sec": round(max(durations), 2) if durations else 0.0,
        "avg_chunk_sec": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "total_internal_cuts": total_internal_cuts,
        "cuts_on_pause": cuts_on_pause,
        "pause_alignment_pct": pause_alignment_pct,
    }


# =============================================================================
# 5. Slice audio + write chunk wavs + build manifest rows
# =============================================================================

def process_file(
    audio_path: str,
    transcript: str,
    chunks_dir: str,
    vad_model,
    get_speech_timestamps,
    align_model,
    tokenizer,
    aligner,
    device: str,
) -> list:
    import time

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    t0 = time.time()
    print(f"  [{base_name}] loading audio ...", flush=True)
    wav = load_audio(audio_path)
    audio_dur_min = (wav.shape[0] / TARGET_SR) / 60
    print(f"  [{base_name}] loaded: {audio_dur_min:.1f} min audio "
          f"({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    print(f"  [{base_name}] running VAD ...", flush=True)
    silence_gaps = get_silence_gaps(wav, vad_model, get_speech_timestamps)
    print(f"  [{base_name}] VAD done: {len(silence_gaps)} silence gaps found "
          f"({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    print(f"  [{base_name}] running forced alignment "
          f"(single pass over full audio, can take a few minutes on CPU) ...",
          flush=True)
    word_times = force_align(
        wav, transcript, align_model, tokenizer, aligner, device)
    print(f"  [{base_name}] alignment done: {len(word_times)} words aligned "
          f"({time.time() - t0:.1f}s)", flush=True)

    if not word_times:
        print(
            f"  [skip] {base_name}: empty transcript or alignment failed", flush=True)
        return [], {"source_file": base_name, "passed": False, "error": "empty_transcript_or_alignment_failed"}

    print(f"  [{base_name}] building chunks ...", flush=True)
    chunks = build_chunks(word_times, silence_gaps)
    print(f"  [{base_name}] {len(chunks)} chunks planned", flush=True)

    validation = assess_chunking(word_times, chunks, silence_gaps)
    status = "PASS" if validation["passed"] else "FAIL"
    print(f"  [validate:{status}] {base_name}: "
          f"coverage={validation['word_coverage_pct']}%  "
          f"dropped={validation['dropped_words']}  "
          f"order_ok={validation['order_preserved']}  "
          f"chunks={validation['num_chunks']}  "
          f"over_limit={validation['chunks_over_limit']}  "
          f"pause_aligned={validation['pause_alignment_pct']}%")

    rows = []
    num_chunks = len(chunks)
    for idx, chunk_words in enumerate(chunks):
        start = time.perf_counter()
        start_sec = max(0.0, chunk_words[0][1] - 0.3)   # small padding
        end_sec = min(wav.shape[0] / TARGET_SR, chunk_words[-1][2] + 0.5)
        start_sample = int(start_sec * TARGET_SR)
        end_sample = int(end_sec * TARGET_SR)

        chunk_audio = wav[start_sample:end_sample]
        chunk_filename = f"{base_name}_chunk{idx:03d}.wav"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        sf.write(chunk_path, chunk_audio.numpy(), TARGET_SR)

        chunk_text = " ".join(w for w, _, _ in chunk_words)
        rows.append((chunk_filename, chunk_text))
        print(
            f"[time] {idx + 1}/{num_chunks}: {(time.perf_counter() - start):.2f}s")
        print(f"  [{base_name}] saved chunk {idx + 1}/{num_chunks}: "
              f"{end_sec - start_sec:5.1f}s  \"{chunk_text[:60]}\"", flush=True)

    validation["source_file"] = base_name
    return rows, validation


# =============================================================================
# 6. Main
# =============================================================================

def build_dataset(
    csv_path: str,
    output_dir: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build] device: {device}")

    converted_dir = os.path.join(output_dir, "converted_wavs")
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "chunks_manifest.csv")
    validation_report_path = os.path.join(
        output_dir, "chunking_validation_report.csv")

    print("[build] reading manifest and converting audio (ffmpeg) ...")
    pairs = find_pairs_from_csv(csv_path, converted_dir)
    print(
        f"[build] {len(pairs)} valid audio+transcript pairs after conversion")

    if not pairs:
        print("\nNo valid pairs found. Check that:\n"
              "  1) --csv_path points to a CSV with columns: Audio, "
              "'transcript' (conversation_id is optional)\n"
              "  2) the 'Audio' column holds paths that actually exist on disk\n"
              "  3) ffmpeg is installed and on PATH (ffmpeg -version to check)")
        return

    try:
        print("[build] loading Silero VAD ...")
        vad_model, get_speech_timestamps = load_vad()
        print("[build] loading MMS forced alignment model ...")
        align_model, tokenizer, aligner = load_aligner(device)
    except RuntimeError as e:
        print("\n" + "=" * 70)
        print("MODEL INITIALIZATION FAILED")
        print("=" * 70)
        print(e)
        print("""
The required models could not be loaded.
If this is your first run:
    • Connect to the internet
    • Re-run the script
After the first successful download, everything will run completely offline.
""")
        return

    # ── Already-processed check, chunked read (memory-lean at millions of rows) ──
    processed_files = set()
    if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
        try:
            print(
                f"[build] scanning existing manifest for already-processed files ...")
            t0 = time.time()
            for chunk in pd.read_csv(manifest_path, usecols=["audio_file"], chunksize=100_000):
                for fname in chunk["audio_file"]:
                    processed_files.add(re.sub(r"_chunk\d+\.wav$", "", fname))
            print(f"[build] {len(processed_files):,} source files already processed "
                  f"({time.time() - t0:.2f}s)")
        except pd.errors.EmptyDataError:
            print("[build] manifest exists but is empty — treating as fresh start")

    # ── Open both output files ONCE, in append mode, and flush after every file ──
    manifest_exists = os.path.exists(
        manifest_path) and os.path.getsize(manifest_path) > 0
    report_exists = os.path.exists(
        validation_report_path) and os.path.getsize(validation_report_path) > 0

    fieldnames = [
        "source_file", "passed", "total_words", "chunked_words", "dropped_words",
        "order_preserved", "word_coverage_pct", "num_chunks", "chunks_over_limit",
        "longest_chunk_sec", "avg_chunk_sec", "total_internal_cuts",
        "cuts_on_pause", "pause_alignment_pct", "error",
    ]

    total_files = len(pairs)
    total_chunks_written = 0
    total_passed = 0
    total_failed = 0

    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_f, \
            open(validation_report_path, "a", newline="", encoding="utf-8") as report_f:

        manifest_writer = csv.writer(manifest_f)
        if not manifest_exists:
            manifest_writer.writerow(["audio_file", "transcript"])

        report_writer = csv.DictWriter(report_f, fieldnames=fieldnames)
        if not report_exists:
            report_writer.writeheader()

        for file_idx, (base_name, wav_path, transcript) in enumerate(pairs, start=1):
            expected_chunk = os.path.join(
                chunks_dir, f"{base_name}_chunk000.wav")
            if base_name in processed_files and os.path.exists(expected_chunk):
                print(
                    f"[skip] ({file_idx}/{total_files}) {base_name} already processed")
                continue

            print(
                f"\n=== Processing file {file_idx}/{total_files}: {base_name} ===", flush=True)
            rows, validation = process_file(
                wav_path, transcript, chunks_dir,
                vad_model, get_speech_timestamps,
                align_model, tokenizer, aligner, device,
            )

            # Flush THIS file's results to disk immediately — nothing sits
            # in memory across files, and a crash on file N doesn't lose
            # files 1..N-1's completed work.
            manifest_writer.writerows(rows)
            report_writer.writerow({k: validation.get(k, "")
                                   for k in fieldnames})
            manifest_f.flush()
            report_f.flush()

            total_chunks_written += len(rows)
            if validation.get("passed"):
                total_passed += 1
            else:
                total_failed += 1

            print(f"[build] progress: {file_idx}/{total_files} files | "
                  f"{total_chunks_written:,} chunks written so far | "
                  f"{total_passed} passed / {total_failed} failed validation")

    print(
        f"\n[build] Done. {total_chunks_written:,} chunks written to {chunks_dir}")
    print(f"[build] Manifest saved to {manifest_path}")
    print(f"[build] Validation report saved to {validation_report_path}")
    print(
        f"[build] Validation: {total_passed}/{total_passed + total_failed} files passed")

    return {
        "manifest_path": manifest_path,
        "chunks_dir": chunks_dir,
        "validation_report": validation_report_path,
        "num_chunks": total_chunks_written,
    }


if __name__ == "__main__":
    build_dataset(csv_path="path/to/your.csv", output_dir="path/to/output")
