"""Compare many recordings at once: all-pairs voice similarity."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .compare import voice_match
from .embedding import embedding_from_file


def similarity_matrix(paths):
    """Compute the all-pairs Voice Match matrix for several recordings.

    Parameters
    ----------
    paths:
        Two or more paths to audio files.

    Returns
    -------
    (labels, matrix):
        ``labels`` is the list of file names; ``matrix`` is an ``(n, n)`` array
        of match percentages where ``matrix[i, j]`` is the Voice Match between
        recording ``i`` and recording ``j``. The matrix is symmetric and its
        diagonal is 100 (every recording matches itself).
    """
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("Need at least two recordings to compare.")

    embeddings = [embedding_from_file(p) for p in paths]
    n = len(embeddings)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        matrix[i, i] = 100.0
        for j in range(i + 1, n):
            _, _, match = voice_match(embeddings[i], embeddings[j])
            matrix[i, j] = match
            matrix[j, i] = match

    labels = [p.name for p in paths]
    return labels, matrix


def write_similarity_report(paths, output_dir) -> dict:
    """Compute the similarity matrix and write a CSV table and heatmap PNG.

    Returns a dict with the labels, the matrix and the produced file paths.
    """
    import pandas as pd

    from .visualize import render_similarity_matrix

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels, matrix = similarity_matrix(paths)

    csv_path = out / "similarity_matrix.csv"
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_path)

    png_path = out / "similarity_matrix.png"
    render_similarity_matrix(labels, matrix, png_path)

    return {
        "labels": labels,
        "matrix": matrix,
        "csv": csv_path,
        "png": png_path,
    }
