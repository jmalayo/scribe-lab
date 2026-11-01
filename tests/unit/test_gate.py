from shared.eval.gate import compare, read_hits, summary, write_hits


def hits(ones: int, zeros: int, start: int = 0) -> dict[str, float]:

    return {f"q{i:03d}": 1.0 if i < start + ones else 0.0 for i in range(start, start + ones + zeros)}


def test_identical_runs_pass():

    baseline = hits(89, 46)
    result = compare(baseline, dict(baseline))

    assert (result["lost"], result["gained"], result["p_value"], result["blocked"]) == ([], [], 1.0, False)


def test_significant_losses_block():

    baseline = hits(100, 35)
    current = {**baseline, **{f"q{i:03d}": 0.0 for i in range(12)}}
    result = compare(baseline, current)

    assert (len(result["lost"]), len(result["gained"])) == (12, 0)
    assert result["p_value"] < 0.05
    assert result["blocked"] is True
    assert "BLOQUEA" in summary(result)


def test_losses_that_may_be_noise_pass():

    baseline = hits(100, 35)
    current = {**baseline, "q000": 0.0, "q001": 0.0, "q120": 1.0}
    result = compare(baseline, current)

    assert (len(result["lost"]), len(result["gained"])) == (2, 1)
    assert result["blocked"] is False


def test_significant_gains_never_block():

    baseline = hits(60, 75)
    current = {**baseline, **{f"q{i:03d}": 1.0 for i in range(60, 80)}}

    assert compare(baseline, current)["blocked"] is False


def test_hits_round_trip_through_csv(tmp_path):

    path = tmp_path / "hits.csv"
    original = {"q002": 0.0, "q001": 1.0}
    write_hits(path, original)

    assert path.read_text(encoding="utf-8").splitlines() == ["question_id,hit_at_10", "q001,1", "q002,0"]
    assert read_hits(path) == original
