import numpy as np
import pytest


def test_neural_embedding_is_normalised(tone_a):
    pytest.importorskip("speechbrain")
    pytest.importorskip("torch")
    from deepvoicedive.neural import NeuralBackendUnavailable, neural_embedding

    try:
        emb = neural_embedding(tone_a)
    except NeuralBackendUnavailable:
        pytest.skip("ECAPA model could not be loaded (no network / not cached).")

    assert emb.ndim == 1
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)
