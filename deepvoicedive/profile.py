"""Voice profile: an averaged, reusable fingerprint of your own voice.

A profile is built once from several clean clips of your voice and reused to
screen many candidate stems later. It stores both the averaged speaker embedding
and the averaged pitch (F0) range.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio_io import load_wav
from .embedding import embedding_from_file
from .pitch import f0_stats


def build_profile(paths, method: str = "mfcc") -> dict:
    """Build a voice profile from one or more recordings.

    Averages the per-clip embeddings (then L2-normalises) and averages the F0
    range across clips. Returns a dict with ``method``, ``embedding`` and the
    pitch range (``f0_median``, ``f0_low``, ``f0_high``).
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("Need at least one clip to build a voice profile.")

    embeddings = [embedding_from_file(p, method=method) for p in paths]
    mean_emb = np.mean(np.vstack(embeddings), axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    medians, lows, highs = [], [], []
    for p in paths:
        y, sr = load_wav(p)
        stats = f0_stats(y, sr)
        if stats["median"] is not None:
            medians.append(stats["median"])
            lows.append(stats["low"])
            highs.append(stats["high"])

    f0_median = float(np.mean(medians)) if medians else None
    f0_low = float(np.mean(lows)) if lows else None
    f0_high = float(np.mean(highs)) if highs else None

    return {
        "method": method,
        "embedding": mean_emb,
        "f0_median": f0_median,
        "f0_low": f0_low,
        "f0_high": f0_high,
    }


def profile_pitch(profile: dict) -> dict:
    """Return the profile's F0 range in the shape used by :func:`pitch.overlap`."""
    return {
        "median": profile.get("f0_median"),
        "low": profile.get("f0_low"),
        "high": profile.get("f0_high"),
    }


def save_profile(profile: dict, path) -> None:
    """Write a profile to JSON (embedding stored as a plain list)."""
    data = dict(profile)
    data["embedding"] = np.asarray(profile["embedding"], dtype=float).tolist()
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_profile(path) -> dict:
    """Load a profile previously written by :func:`save_profile`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["embedding"] = np.asarray(data["embedding"], dtype=np.float64)
    return data
