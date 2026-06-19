import pytest


def test_egemaps_has_88_parameters(tone_a):
    pytest.importorskip("opensmile")
    from deepvoicedive.egemaps import egemaps_report

    report = egemaps_report(tone_a)
    assert len(report) == 88
    assert list(report.columns) == ["parameter", "value"]
