"""Screen candidate vocal stems against your own voice profile.

For each stem this computes a voice-similarity score and a pitch-range overlap,
combines them into a single suitability score and gives a pass/fail verdict —
helping you pick which stem is the best candidate for a voice swap.
"""
from __future__ import annotations

from pathlib import Path

from .audio_io import load_wav
from .compare import voice_match
from .embedding import embedding_from_file
from .pitch import f0_stats, pitch_compatibility
from .profile import profile_pitch

# Suitability weighting: how much timbre similarity vs pitch-range overlap counts.
DEFAULT_WEIGHTS = (0.7, 0.3)
DEFAULT_THRESHOLD = 75.0


def screen_candidates(
    profile: dict,
    stems,
    method: str = "mfcc",
    weights=DEFAULT_WEIGHTS,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Score each candidate stem against a voice profile.

    Returns a list of result dicts (``name``, ``match``, ``pitch``,
    ``suitability``, ``verdict``) sorted by suitability, best first.
    """
    w_match, w_pitch = weights
    ref_emb = profile["embedding"]
    ref_pitch = profile_pitch(profile)

    results = []
    for stem in stems:
        stem = Path(stem)
        emb = embedding_from_file(stem, method=method)
        _dist, _sim, match = voice_match(ref_emb, emb)

        y, sr = load_wav(stem)
        pitch_pct = pitch_compatibility(ref_pitch, f0_stats(y, sr)) * 100.0

        suitability = round(w_match * match + w_pitch * pitch_pct, 2)
        results.append(
            {
                "name": stem.name,
                "match": round(match, 2),
                "pitch": round(pitch_pct, 2),
                "suitability": suitability,
                "verdict": suitability >= threshold,
            }
        )

    results.sort(key=lambda r: r["suitability"], reverse=True)
    return results


def write_screening_report(
    profile: dict,
    stems,
    output_dir,
    method: str = "mfcc",
    weights=DEFAULT_WEIGHTS,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Screen stems and write a CSV table and a green/red bar chart."""
    import pandas as pd

    from .visualize import render_screening

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = screen_candidates(
        profile, stems, method=method, weights=weights, threshold=threshold
    )

    csv_path = out / "screening.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    png_path = out / "screening.png"
    render_screening(results, png_path, threshold=threshold)

    return {"results": results, "csv": csv_path, "png": png_path}
