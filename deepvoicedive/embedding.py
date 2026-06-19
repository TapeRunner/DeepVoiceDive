"""Voice embedding: a compact, L2-normalised acoustic fingerprint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio_io import load_wav
from .features import N_MFCC, extract_mfcc

# The embedding has one dimension per MFCC coefficient.
EMBEDDING_DIM = N_MFCC


def compute_embedding(y, sr):
    """Compute a 40-dim L2-normalised voice embedding from a waveform.

    The embedding is the time-averaged MFCC vector, normalised to unit length so
    that cosine comparisons become scale-invariant and stable across recordings
    of different loudness or duration.
    """
    mfcc = extract_mfcc(y, sr)
    vec = mfcc.mean(axis=1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float64)


def embedding_from_file(path, method: str = "mfcc"):
    """Load an audio file and return its voice embedding.

    Parameters
    ----------
    path:
        Path to an audio file.
    method:
        ``"mfcc"`` (default) uses the offline MFCC fingerprint. ``"neural"`` uses
        the SpeechBrain ECAPA speaker-embedding model (requires the ``neural``
        extra; see :mod:`deepvoicedive.neural`).
    """
    if method == "neural":
        from .neural import neural_embedding

        return neural_embedding(path)
    if method == "mfcc":
        y, sr = load_wav(path)
        return compute_embedding(y, sr)
    raise ValueError(f"Unknown embedding method: {method!r} (use 'mfcc' or 'neural').")


def save_embedding(vec, json_path=None, npy_path=None):
    """Persist an embedding as JSON and/or a NumPy ``.npy`` file."""
    if npy_path is not None:
        np.save(str(npy_path), vec)
    if json_path is not None:
        Path(json_path).write_text(json.dumps(vec.tolist()), encoding="utf-8")
