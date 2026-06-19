from deepvoicedive.audio_io import load_wav
from deepvoicedive.pitch import f0_stats, pitch_compatibility


def test_f0_stats_detects_pitch(tone_a):
    y, sr = load_wav(tone_a)
    stats = f0_stats(y, sr)
    # tone_a is a 220 Hz tone; the detected median should be in that ballpark.
    assert stats["median"] is not None
    assert 150 < stats["median"] < 300


def test_compatibility_identical_pitch_is_full():
    a = {"median": 220.0, "low": 200.0, "high": 240.0}
    assert pitch_compatibility(a, a) == 1.0


def test_compatibility_octave_apart_is_zero():
    a = {"median": 220.0, "low": 200.0, "high": 240.0}
    b = {"median": 440.0, "low": 400.0, "high": 480.0}  # exactly one octave up
    assert pitch_compatibility(a, b) == 0.0


def test_compatibility_close_pitch_is_high():
    a = {"median": 220.0, "low": 200.0, "high": 240.0}
    b = {"median": 226.0, "low": 206.0, "high": 246.0}  # ~0.5 semitone apart
    assert pitch_compatibility(a, b) > 0.9


def test_compatibility_handles_unknown():
    a = {"median": 220.0, "low": 200.0, "high": 240.0}
    b = {"median": None, "low": None, "high": None}
    assert pitch_compatibility(a, b) == 0.0
