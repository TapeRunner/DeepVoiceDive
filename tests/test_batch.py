import numpy as np
import pytest

from deepvoicedive.batch import similarity_matrix


def test_matrix_is_symmetric_with_full_diagonal(tone_a, tone_a_copy, tone_b):
    labels, m = similarity_matrix([tone_a, tone_a_copy, tone_b])
    assert m.shape == (3, 3)
    assert len(labels) == 3
    assert np.allclose(np.diag(m), 100.0)
    assert np.allclose(m, m.T)


def test_identical_pair_scores_higher_than_different(tone_a, tone_a_copy, tone_b):
    _labels, m = similarity_matrix([tone_a, tone_a_copy, tone_b])
    # index 0 and 1 are identical tones; index 2 is clearly different.
    assert m[0, 1] > m[0, 2]


def test_needs_at_least_two(tone_a):
    with pytest.raises(ValueError):
        similarity_matrix([tone_a])
