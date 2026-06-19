import numpy as np

from deepvoicedive.embedding import EMBEDDING_DIM, embedding_from_file


def test_embedding_dimension(tone_a):
    emb = embedding_from_file(tone_a)
    assert emb.shape == (EMBEDDING_DIM,)


def test_embedding_is_l2_normalised(tone_a):
    emb = embedding_from_file(tone_a)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_embedding_is_deterministic(tone_a):
    a = embedding_from_file(tone_a)
    b = embedding_from_file(tone_a)
    assert np.allclose(a, b)
