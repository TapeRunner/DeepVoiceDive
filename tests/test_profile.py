import numpy as np

from deepvoicedive.profile import build_profile, load_profile, save_profile


def test_profile_embedding_is_l2_normalised(tone_a, tone_a_copy):
    profile = build_profile([tone_a, tone_a_copy], method="mfcc")
    assert np.isclose(np.linalg.norm(profile["embedding"]), 1.0, atol=1e-6)
    assert profile["method"] == "mfcc"


def test_profile_save_load_roundtrip(tone_a, tone_b, tmp_path):
    profile = build_profile([tone_a, tone_b], method="mfcc")
    path = tmp_path / "prof.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert np.allclose(loaded["embedding"], profile["embedding"])
    assert loaded["method"] == "mfcc"


def test_profile_requires_a_clip():
    import pytest

    with pytest.raises(ValueError):
        build_profile([], method="mfcc")
