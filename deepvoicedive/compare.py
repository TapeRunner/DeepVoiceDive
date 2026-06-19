"""Biometric voice comparison via cosine distance."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine

from .embedding import embedding_from_file


def cosine_distance(emb_a, emb_b) -> float:
    """Cosine distance in [0, 2]; 0 means identical direction."""
    return float(cosine(np.asarray(emb_a, dtype=np.float64),
                        np.asarray(emb_b, dtype=np.float64)))


def voice_match(emb_a, emb_b):
    """Compare two embeddings.

    Returns
    -------
    (distance, similarity, match_percent):
        distance      -- cosine distance in [0, 2].
        similarity    -- ``1 - distance`` in [-1, 1].
        match_percent -- clamped similarity mapped to [0, 100], rounded to 2 dp.
    """
    dist = cosine_distance(emb_a, emb_b)
    similarity = 1.0 - dist
    match = max(0.0, similarity) * 100.0
    return dist, similarity, round(match, 2)


def compare_files(path_a, path_b):
    """Compute the voice match between two audio files."""
    emb_a = embedding_from_file(path_a)
    emb_b = embedding_from_file(path_b)
    return voice_match(emb_a, emb_b)
