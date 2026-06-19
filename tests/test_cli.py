from deepvoicedive.cli import main


def test_analyze_cli_creates_artifacts(tone_a, tmp_path):
    out = tmp_path / "out"
    rc = main(["analyze", str(tone_a), "--output-dir", str(out), "--no-egemaps"])
    assert rc == 0
    assert (out / "embedding.json").exists()
    assert (out / "embedding.npy").exists()
    assert (out / "report.png").exists()


def test_compare_cli(tone_a, tone_a_copy, capsys):
    rc = main(["compare", str(tone_a), str(tone_a_copy)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Voice Match" in out


def test_missing_file_returns_error(tmp_path):
    rc = main(["analyze", str(tmp_path / "nope.wav"), "--no-egemaps"])
    assert rc == 1
