import numpy as np
import pytest

from sportkin.stats import (
    agreement,
    bland_altman,
    cohens_d,
    icc,
    interpret_icc,
    pearson,
    reliability,
    typical_error,
)

# Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
               [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)


@pytest.mark.parametrize("model, expected", [
    ("1,1", 0.17), ("2,1", 0.29), ("3,1", 0.71),
    ("1,k", 0.44), ("2,k", 0.62), ("3,k", 0.91),
])
def test_icc_matches_shrout_fleiss(model, expected):
    assert icc(SF, model).value == pytest.approx(expected, abs=0.005)


def test_icc_confidence_intervals_reference():
    # reference values computed with pingouin 0.5 for the same data
    r = icc(SF, "2,1")
    assert (r.ci_low, r.ci_high) == pytest.approx((0.019, 0.761), abs=0.002)
    r = icc(SF, "3,1")
    assert (r.ci_low, r.ci_high) == pytest.approx((0.342, 0.946), abs=0.002)
    assert r.f == pytest.approx(11.027, abs=1e-3)


def test_icc_against_pingouin_if_available():
    pd = pytest.importorskip("pandas")
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(7)
    true = rng.normal(35, 6, 25)
    data = np.column_stack([true + rng.normal(0, 1.5, 25), true + rng.normal(0.8, 1.5, 25)])
    long = pd.DataFrame([(i, j, data[i, j]) for i in range(25) for j in range(2)],
                        columns=["s", "r", "v"])
    ref = pg.intraclass_corr(long, "s", "r", "v").set_index("Type")
    # pingouin renamed the ICC types between versions: "ICC2" -> "ICC(A,1)"
    for model, keys in [("2,1", ("ICC(A,1)", "ICC2")), ("3,1", ("ICC(C,1)", "ICC3"))]:
        key = next(k for k in keys if k in ref.index)
        assert icc(data, model).value == pytest.approx(ref.loc[key, "ICC"], abs=1e-9)


def test_icc_drops_incomplete_rows_and_validates():
    data = np.vstack([SF, [np.nan, 1, 2, 3]])
    assert icc(data, "3,1").n == 6
    with pytest.raises(ValueError):
        icc(SF, "4,1")


def test_interpret_icc_labels():
    assert interpret_icc(0.4) == "poor"
    assert interpret_icc(0.8, "tr") == "iyi"
    assert interpret_icc(0.95) == "excellent"


def test_bland_altman_manual():
    a = np.array([10.0, 12, 14, 16, 18])
    b = np.array([9.0, 12.5, 13, 15, 17.5])
    r = bland_altman(a, b)
    d = a - b
    assert r.bias == pytest.approx(d.mean())
    assert r.loa_high == pytest.approx(d.mean() + 1.96 * d.std(ddof=1), rel=1e-3)
    assert r.bias_ci[0] < r.bias < r.bias_ci[1]
    assert r.loa_low_ci[0] < r.loa_low < r.loa_low_ci[1]


def test_bland_altman_detects_proportional_bias():
    x = np.linspace(20, 60, 40)
    r = bland_altman(x * 1.1, x)
    assert r.prop_bias_slope > 0 and r.prop_bias_p < 0.001


def test_typical_error():
    t1 = np.array([30.0, 32, 35, 28, 40, 33])
    t2 = t1 + np.array([1.0, -1, 1, -1, 1, -1])
    te = typical_error(t1, t2)
    assert te["typical_error"] == pytest.approx(np.std(t2 - t1, ddof=1) / np.sqrt(2))
    assert te["mdc95"] == pytest.approx(1.96 * np.sqrt(2) * te["typical_error"])


def test_cohens_d_paired_and_independent():
    a = np.array([5.0, 6, 7, 8, 9])
    b = a - np.array([1.0, 2, 1, 2, 1])
    d = cohens_d(a, b)
    diff = a - b
    assert d["d_z"] == pytest.approx(diff.mean() / diff.std(ddof=1))
    ind = cohens_d([1.0, 2, 3, 4], [3.0, 4, 5, 6], paired=False)
    assert ind["d"] == pytest.approx(-2 / np.std([1, 2, 3, 4], ddof=1))
    assert abs(ind["hedges_g"]) < abs(ind["d"])


def test_pearson_ci_contains_r():
    x = np.arange(20.0)
    r = pearson(x, x + np.sin(x))
    assert r["ci_low"] < r["r"] < r["ci_high"]


def test_summaries_run_end_to_end():
    rng = np.random.default_rng(1)
    crit = rng.normal(35, 5, 20)
    video = crit + rng.normal(0.3, 1.0, 20)
    out = agreement(video, crit)
    assert out["icc"]["value"] > 0.9
    rel = reliability(crit, crit + rng.normal(0, 0.5, 20))
    assert rel["typical_error"]["typical_error"] < 1


def test_too_few_pairs():
    with pytest.raises(ValueError):
        bland_altman([1.0, 2.0], [1.0, 2.0])
