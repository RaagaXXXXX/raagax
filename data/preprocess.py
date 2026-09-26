"""Audio Preprocessing Pipeline for Raagax.

Pipeline Order:
    silence-strip → resample → slice → normalize → write

Docstring & Overview:
    1. Silence Stripping: Strips leading and trailing silence from the raw audio waveform
       using `librosa.effects.trim` based on energy thresholding (default 30 dB).
    2. Resampling & Channel Conforming: Forces audio to mono and resamples to target
       sample rate (default 22050 Hz from config.yaml).
    3. Slicing & Padding: Divides continuous audio into non-overlapping 5.0-second clips.
       Design Choice (Padding vs. Dropping):
       Any final trailing clip >= 1.0s is zero-padded with silence to exactly 5.0 seconds
       (110,250 samples at 22,050 Hz). This guarantees uniform tensor dimensions for
       downstream CQT feature extraction and CNN-BiLSTM batch processing. Remnants < 1.0s
       are dropped as they lack sufficient musical/sargam context.
    4. Loudness Normalization: Normalizes each clip to -23.0 LUFS using `pyloudnorm`
       (ITU-R BS.1770-4 compliance), ensuring consistent perceptual loudness across
       recordings and instruments.
    5. WAV & Metadata Export: Writes 16-bit PCM WAV files loadable by librosa, and logs
       clip metadata (filename, raga, tonic, duration, source) to `metadata.csv`.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raagax.preprocess")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
METADATA_COLUMNS = ["filename", "raga", "tonic", "duration", "source"]


def find_config_path(explicit_path: Optional[str] = None) -> Path:
    """Find config.yaml either from user argument or standard project locations."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Config file not found at: {explicit_path}")

    # Standard candidate locations
    candidates = [
        Path("config.yaml"),
        Path("../config.yaml"),
        Path(__file__).resolve().parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    logger.warning("config.yaml not found in candidate paths. Using default configuration.")
    return Path("config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load audio and path settings from config.yaml, falling back to defaults."""
    defaults = {
        "audio": {
            "sample_rate": 22050,
            "duration": 5.0,
            "mono": True,
        },
        "paths": {
            "raw_data": "data/raw/",
            "processed": "data/processed/",
        },
    }

    try:
        resolved_path = find_config_path(config_path)
        if resolved_path.is_file():
            with open(resolved_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # Merge with defaults
            if "audio" in loaded:
                defaults["audio"].update(loaded["audio"])
            if "paths" in loaded:
                defaults["paths"].update(loaded["paths"])
            logger.info("Loaded configuration from %s", resolved_path)
    except Exception as e:
        logger.warning("Error reading config file (%s). Proceeding with defaults.", e)

    return defaults


def strip_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """
    Step 1: Strip leading and trailing silence from the audio signal.

    Args:
        y: Audio time series as a 1D numpy array.
        top_db: The threshold (in decibels) below reference to consider as silence.

    Returns:
        Trimmed audio time series.
    """
    if len(y) == 0:
        return y
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def load_and_resample(
    file_path: Path,
    target_sr: int = 22050,
    mono: bool = True,
    top_db: float = 30.0,
) -> Tuple[np.ndarray, int]:
    """
    Load raw audio file, strip leading/trailing silence, and resample to target_sr.

    Pipeline Order Followed:
        silence-strip → resample

    Args:
        file_path: Path to the raw audio file.
        target_sr: Target sample rate in Hz.
        mono: Force mono channel if True.
        top_db: Energy threshold for silence trimming.

    Returns:
        Tuple of (resampled audio array, target_sr).
    """
    # 1. Load native audio (forcing mono if requested)
    y, sr = librosa.load(str(file_path), sr=None, mono=mono)

    # 2. Strip leading/trailing silence before resampling/slicing
    y_trimmed = strip_silence(y, top_db=top_db)
    if len(y_trimmed) == 0:
        return np.array([], dtype=np.float32), target_sr

    # 3. Resample to target sample rate if different
    if sr != target_sr:
        y_resampled = librosa.resample(y_trimmed, orig_sr=sr, target_sr=target_sr)
    else:
        y_resampled = y_trimmed

    return y_resampled.astype(np.float32), target_sr


def slice_audio(
    y: np.ndarray,
    sr: int,
    duration: float = 5.0,
    min_pad_duration: float = 1.0,
) -> List[np.ndarray]:
    """
    Step 3: Slice audio into non-overlapping clips of length `duration` seconds.

    Design Choice (Padding vs. Dropping):
        Any trailing clip shorter than `duration` is zero-padded with silence to exactly
        `duration` seconds if its length is >= `min_pad_duration` seconds (default 1.0s).
        This guarantees uniform clip lengths (sample_rate * duration samples) for downstream
        CQT feature extraction without shape mismatches.
        Trailing fragments < min_pad_duration seconds are dropped due to lack of musical context.

    Args:
        y: Audio time series.
        sr: Sample rate.
        duration: Target clip duration in seconds.
        min_pad_duration: Minimum seconds required to pad rather than drop a trailing clip.

    Returns:
        List of audio clip arrays, each of length int(round(sr * duration)).
    """
    clip_samples = int(round(sr * duration))
    min_samples = int(round(sr * min_pad_duration))
    total_samples = len(y)

    clips: List[np.ndarray] = []
    if total_samples < min_samples:
        return clips

    start = 0
    while start < total_samples:
        end = start + clip_samples
        chunk = y[start:end]

        if len(chunk) == clip_samples:
            clips.append(chunk)
        elif len(chunk) >= min_samples:
            # Pad the trailing short clip with silence to reach exact clip_samples
            padded = np.pad(
                chunk,
                (0, clip_samples - len(chunk)),
                mode="constant",
                constant_values=0.0,
            )
            clips.append(padded)
        # Remnants < min_samples are discarded
        start = end

    return clips


def normalize_loudness(
    clip: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """
    Step 4: Normalize loudness to target LUFS using pyloudnorm.

    Args:
        clip: 1D audio clip array.
        sr: Sample rate in Hz.
        target_lufs: Integrated loudness target in LUFS (default: -23.0).

    Returns:
        Loudness-normalized audio array.
    """
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(clip)
    except Exception:
        loudness = float("-inf")

    # If the clip is silent or extremely quiet, integrated loudness cannot be normalized
    if np.isneginf(loudness) or np.isnan(loudness) or loudness < -70.0:
        peak = np.max(np.abs(clip))
        if peak > 0:
            # Modest headroom peak scaling for low-energy signals
            return clip / peak * 0.1
        return clip

    # Normalize to target LUFS
    normalized = pyln.normalize.loudness(clip, loudness, target_lufs)

    # Prevent digital clipping beyond [-1.0, 1.0]
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak * 0.99

    return normalized.astype(np.float32)


def write_clip(output_path: Path, clip: np.ndarray, sr: int) -> None:
    """
    Step 5: Write audio clip to disk as uncompressed 16-bit PCM WAV.

    Trivially readable by librosa and downstream feature extractors.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), clip, sr, subtype="PCM_16")


def parse_metadata_from_filename(
    raw_path: Path,
    lookup_table: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Parse raga, tonic, and source from the raw filename or optional lookup table.

    Naming convention supported:
        <raga>_<tonic>_<source>[_<optional_id>].ext
        e.g., "yaman_C_sarangi_01.wav" -> raga="yaman", tonic="C", source="sarangi"
        e.g., "bhairavi_D#_vocal.mp3"  -> raga="bhairavi", tonic="D#", source="vocal"

    If the filename does not follow this format, tokens default to "unknown".
    """
    stem = raw_path.stem
    filename = raw_path.name

    # Check lookup table first if provided
    if lookup_table:
        if filename in lookup_table:
            return lookup_table[filename]
        if stem in lookup_table:
            return lookup_table[stem]

    # Delimiter-based parsing (supports '_' or '-')
    parts = stem.replace("-", "_").split("_")

    raga = parts[0] if len(parts) >= 1 and parts[0] else "unknown"
    tonic = parts[1] if len(parts) >= 2 and parts[1] else "unknown"
    source = parts[2] if len(parts) >= 3 and parts[2] else "unknown"

    return {
        "raga": raga,
        "tonic": tonic,
        "source": source,
    }


def load_lookup_table(lookup_file: Optional[Path]) -> Optional[Dict[str, Dict[str, str]]]:
    """Load an optional metadata lookup table from CSV or YAML."""
    if not lookup_file or not lookup_file.is_file():
        return None

    try:
        if lookup_file.suffix.lower() in {".yaml", ".yml"}:
            with open(lookup_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif lookup_file.suffix.lower() == ".csv":
            df = pd.read_csv(lookup_file)
            lookup = {}
            for _, row in df.iterrows():
                key = str(row.get("filename", "")).strip()
                if key:
                    lookup[key] = {
                        "raga": str(row.get("raga", "unknown")),
                        "tonic": str(row.get("tonic", "unknown")),
                        "source": str(row.get("source", "unknown")),
                    }
            return lookup
    except Exception as e:
        logger.warning("Failed to load lookup table from %s: %s", lookup_file, e)

    return None


def append_to_metadata_csv(metadata_path: Path, new_records: List[Dict[str, Any]]) -> None:
    """
    Append processed clip records to data/processed/metadata.csv.

    Columns: filename, raga, tonic, duration, source
    """
    if not new_records:
        return

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_records, columns=METADATA_COLUMNS)

    if metadata_path.exists():
        # Read existing file to check for duplicate entries
        try:
            existing_df = pd.read_csv(metadata_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate by filename keeping the most recently processed
            combined_df = combined_df.drop_duplicates(subset=["filename"], keep="last")
            combined_df.to_csv(metadata_path, index=False)
        except Exception:
            # Fallback append
            new_df.to_csv(metadata_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(metadata_path, index=False)

    logger.info("Saved %d metadata entries to %s", len(new_records), metadata_path)


def process_audio_file(
    file_path: Path,
    output_dir: Path,
    target_sr: int,
    clip_duration: float,
    target_lufs: float = -23.0,
    top_db: float = 30.0,
    lookup_table: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Process a single raw audio file through the complete pipeline:
        silence-strip → resample → slice → normalize → write
    """
    meta_info = parse_metadata_from_filename(file_path, lookup_table=lookup_table)
    raw_stem = file_path.stem

    # 1 & 2: Load, strip silence, and resample
    try:
        y, sr = load_and_resample(file_path, target_sr=target_sr, mono=True, top_db=top_db)
    except Exception as e:
        logger.error("Failed to load/resample %s: %s", file_path.name, e)
        return []

    if len(y) == 0:
        logger.warning("Skipping %s: audio is completely silent or empty.", file_path.name)
        return []

    # 3: Slice into 5.0s clips (with zero-padding for final clip >= 1.0s)
    clips = slice_audio(y, sr=sr, duration=clip_duration)
    if not clips:
        logger.warning("Skipping %s: trimmed audio shorter than minimum slice threshold.", file_path.name)
        return []

    clip_records: List[Dict[str, Any]] = []

    # 4 & 5: Normalize loudness, write WAV, prepare metadata
    for idx, clip in enumerate(clips):
        normalized_clip = normalize_loudness(clip, sr=sr, target_lufs=target_lufs)
        clip_filename = f"{raw_stem}_clip{idx:03d}.wav"
        clip_output_path = output_dir / clip_filename

        write_clip(clip_output_path, normalized_clip, sr=sr)

        clip_records.append({
            "filename": clip_filename,
            "raga": meta_info["raga"],
            "tonic": meta_info["tonic"],
            "duration": clip_duration,
            "source": meta_info["source"],
        })

    logger.info("Processed %s -> generated %d clips", file_path.name, len(clips))
    return clip_records


def preprocess_all(
    input_dir: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    lookup_file: Optional[Path] = None,
) -> None:
    """
    Preprocess all raw audio files in input_dir and export clips to output_dir.
    """
    if config is None:
        config = load_config()

    audio_cfg = config.get("audio", {})
    target_sr = int(audio_cfg.get("sample_rate", 22050))
    clip_duration = float(audio_cfg.get("duration", 5.0))

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup_table = load_lookup_table(lookup_file)

    # Gather all supported raw audio files
    audio_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        logger.warning("No audio files found in %s matching extensions: %s", input_dir, SUPPORTED_EXTENSIONS)
        return

    logger.info("Found %d audio files in %s. Starting preprocessing...", len(audio_files), input_dir)

    all_records: List[Dict[str, Any]] = []
    for audio_file in sorted(audio_files):
        records = process_audio_file(
            file_path=audio_file,
            output_dir=output_dir,
            target_sr=target_sr,
            clip_duration=clip_duration,
            lookup_table=lookup_table,
        )
        all_records.extend(records)

    # Append to metadata.csv
    metadata_csv_path = output_dir / "metadata.csv"
    append_to_metadata_csv(metadata_csv_path, all_records)

    logger.info("Preprocessing complete! %d total clips written to %s", len(all_records), output_dir)


def main() -> None:
    """CLI Entry point for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="Raagax Audio Preprocessing Pipeline (silence-strip -> resample -> slice -> normalize -> write)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to searching project root)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing raw audio files (overrides config.yaml paths.raw_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save processed clips (overrides config.yaml paths.processed)",
    )
    parser.add_argument(
        "--lookup",
        type=str,
        default=None,
        help="Optional path to metadata lookup file (.csv or .yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})

    input_dir = Path(args.input) if args.input else Path(paths_cfg.get("raw_data", "data/raw/"))
    output_dir = Path(args.output) if args.output else Path(paths_cfg.get("processed", "data/processed/"))
    lookup_file = Path(args.lookup) if args.lookup else None

    preprocess_all(
        input_dir=input_dir,
        output_dir=output_dir,
        config=cfg,
        lookup_file=lookup_file,
    )


if __name__ == "__main__":
    main()
