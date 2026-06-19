"""eGeMAPS feature extraction via openSMILE (88 functional parameters).

The extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPSv02) is a
scientific standard covering formants (resonance), jitter (frequency
perturbation), shimmer (amplitude perturbation), loudness, spectral and other
clinically relevant descriptors.
"""
from __future__ import annotations

import pandas as pd


def _smile():
    """Build an openSMILE extractor for eGeMAPSv02 functionals (lazy import)."""
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps(path):
    """Return a one-row DataFrame with the 88 eGeMAPSv02 functional features."""
    smile = _smile()
    return smile.process_file(str(path))


def egemaps_report(path):
    """Return a tidy ``(parameter, value)`` DataFrame for human-readable reports."""
    df = extract_egemaps(path)
    row = df.iloc[0]
    return pd.DataFrame({"parameter": row.index, "value": row.values})
