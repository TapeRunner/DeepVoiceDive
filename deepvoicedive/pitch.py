"""Fundamental-frequency (pitch / register) analysis.

For voice swapping, two voices match better when they share a similar pitch
range. These helpers estimate a robust F0 range per recording and measure how
much two ranges overlap.
"""
from __future__ import annotations

import librosa
import numpy as np

# Human singing/speech roughly spans these notes (C2–C6).
_FMIN = 65.0
_FMAX = 1047.0


def f0_stats(y, sr) -> dict:
    """Estimate the fundamental-frequency range of a recording.

    Returns a dict with ``median``, ``low`` (10th percentile) and ``high``
    (90th percentile) of the voiced F0 in Hz. If no voiced frames are found, all
    values are ``None``.
    """
    f0, voiced_flag, _ = librosa.pyin(y, fmin=_FMIN, fmax=_FMAX, sr=sr)
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return {"median": None, "low": None, "high": None}
    return {
        "median": float(np.median(voiced)),
        "low": float(np.percentile(voiced, 10)),
        "high": float(np.percentile(voiced, 90)),
    }


def f0_stats_from_file(path) -> dict:
    from .audio_io import load_wav

    y, sr = load_wav(path)
    return f0_stats(y, sr)


# Pitch difference (in semitones) at which two voices are treated as fully
# register-incompatible. One octave = 12 semitones.
_OCTAVE_SEMITONES = 12.0


def pitch_compatibility(a: dict, b: dict) -> float:
    """How register-compatible two voices are, as a fraction in [0, 1].

    Based on the distance between the two median pitches in semitones: identical
    pitch → 1.0, decaying linearly to 0.0 at one octave apart or more. This is
    robust to narrow F0 ranges and matches what makes a voice swap work well
    (similar singing register). Returns 0.0 if either median is unknown.
    """
    if not a or not b:
        return 0.0
    ma, mb = a.get("median"), b.get("median")
    if not ma or not mb or ma <= 0 or mb <= 0:
        return 0.0
    semitones = abs(12.0 * np.log2(ma / mb))
    return float(max(0.0, 1.0 - semitones / _OCTAVE_SEMITONES))
