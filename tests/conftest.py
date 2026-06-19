"""Shared pytest fixtures.

Test audio is generated synthetically (sine tones) so no binary fixtures need to
live in the repository.
"""
import numpy as np
import pytest
import soundfile as sf

SR = 16000


def _tone(freq, seconds=1.0, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def tone_a(tmp_path):
    path = tmp_path / "tone_a.wav"
    sf.write(str(path), _tone(220.0), SR)
    return path


@pytest.fixture
def tone_a_copy(tmp_path):
    path = tmp_path / "tone_a_copy.wav"
    sf.write(str(path), _tone(220.0), SR)
    return path


@pytest.fixture
def tone_b(tmp_path):
    """A spectrally rich, clearly different signal."""
    path = tmp_path / "tone_b.wav"
    sig = _tone(440.0) + 0.3 * _tone(1500.0) + 0.2 * _tone(3000.0)
    sf.write(str(path), sig.astype(np.float32), SR)
    return path
