from deepvoicedive.compare import compare_files


def test_identical_recordings_match_fully(tone_a, tone_a_copy):
    dist, _sim, match = compare_files(tone_a, tone_a_copy)
    assert match >= 99.9
    assert dist < 1e-3


def test_different_recordings_match_less(tone_a, tone_a_copy, tone_b):
    _, _, same = compare_files(tone_a, tone_a_copy)
    _, _, diff = compare_files(tone_a, tone_b)
    assert diff < same
