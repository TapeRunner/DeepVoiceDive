"""Audio loading and validation utilities."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

# eGeMAPS / openSMILE are designed around 16 kHz audio; we resample to match.
DEFAULT_SAMPLE_RATE = 16000


def load_wav(path, sr: int = DEFAULT_SAMPLE_RATE):
    """Load an audio file as a mono waveform at the target sample rate.

    Parameters
    ----------
    path:
        Path to an audio file (WAV recommended, uncompressed).
    sr:
        Target sample rate. Audio is resampled to this rate.

    Returns
    -------
    (y, sr): tuple of (1-D float32 numpy array, int sample rate).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    try:
        y, sr_out = librosa.load(str(p), sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the CLI
        raise ValueError(f"Could not read audio file '{p}': {exc}") from exc
    if y.size == 0:
        raise ValueError(f"Audio file '{p}' contains no samples.")
    return y.astype(np.float32), int(sr_out)
