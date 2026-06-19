"""Assemble the full analysis report for a single recording."""
from __future__ import annotations

from pathlib import Path

from .audio_io import load_wav
from .embedding import compute_embedding, save_embedding
from .visualize import render_report


def analyze_file(path, output_dir, with_egemaps: bool = True) -> dict:
    """Run the full analysis and write artefacts to ``output_dir``.

    Produces:
      - ``embedding.json`` / ``embedding.npy`` -- the voice fingerprint,
      - ``report.png``                         -- spectrogram + MFCC matrix,
      - ``egemaps_report.csv``                 -- 88 eGeMAPS parameters (optional).

    Returns a dict describing the produced files and the embedding vector.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y, sr = load_wav(path)
    embedding = compute_embedding(y, sr)

    emb_json = out / "embedding.json"
    emb_npy = out / "embedding.npy"
    save_embedding(embedding, json_path=emb_json, npy_path=emb_npy)

    png_path = out / "report.png"
    render_report(y, sr, png_path)

    result = {
        "embedding": embedding,
        "embedding_json": emb_json,
        "embedding_npy": emb_npy,
        "report_png": png_path,
    }

    if with_egemaps:
        from .egemaps import egemaps_report

        csv_path = out / "egemaps_report.csv"
        egemaps_report(path).to_csv(csv_path, index=False)
        result["egemaps_csv"] = csv_path

    return result
