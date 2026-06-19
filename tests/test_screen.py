from deepvoicedive.profile import build_profile
from deepvoicedive.screen import screen_candidates, write_screening_report


def test_identical_stem_scores_higher_than_different(tone_a, tone_a_copy, tone_b):
    profile = build_profile([tone_a], method="mfcc")
    results = screen_candidates(profile, [tone_a_copy, tone_b], method="mfcc")
    # Results are sorted best-first; the identical copy must beat the different tone.
    by_name = {r["name"]: r for r in results}
    assert by_name[tone_a_copy.name]["suitability"] > by_name[tone_b.name]["suitability"]


def test_results_sorted_by_suitability(tone_a, tone_a_copy, tone_b):
    profile = build_profile([tone_a], method="mfcc")
    results = screen_candidates(profile, [tone_b, tone_a_copy], method="mfcc")
    scores = [r["suitability"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_verdict_follows_threshold(tone_a, tone_a_copy):
    profile = build_profile([tone_a], method="mfcc")
    high = screen_candidates(profile, [tone_a_copy], method="mfcc", threshold=0.0)
    low = screen_candidates(profile, [tone_a_copy], method="mfcc", threshold=101.0)
    assert high[0]["verdict"] is True
    assert low[0]["verdict"] is False


def test_screening_report_writes_artifacts(tone_a, tone_a_copy, tone_b, tmp_path):
    profile = build_profile([tone_a], method="mfcc")
    out = tmp_path / "screen"
    result = write_screening_report(
        profile, [tone_a_copy, tone_b], out, method="mfcc"
    )
    assert (out / "screening.csv").exists()
    assert (out / "screening.png").exists()
    assert len(result["results"]) == 2
