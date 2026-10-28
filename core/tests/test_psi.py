"""PSICalculator: bucketing, PSI values and checkpointing (G.5, Table 9)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import PSIConfig
from bayesian_credit.exceptions import ConfigurationError
from bayesian_credit.monitoring import MISSING_BUCKET, OTHER_BUCKET, PSICalculator


def frame(rng: np.random.Generator, n: int, shift: float = 0.0, grades=("A", "B", "C", "D")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fico_range_low": rng.normal(700.0 + shift * 30.0, 30.0, n).round(0),
            "dti": rng.gamma(4.0, 4.0, n) + shift * 5.0,
            "inq_last_6mths": rng.poisson(0.8, n).astype(float),
            "grade": pd.Series(rng.choice(list(grades), n), dtype="string"),
        }
    )


FEATURES = ["fico_range_low", "dti", "inq_last_6mths", "grade"]


@pytest.fixture
def reference() -> pd.DataFrame:
    return frame(np.random.default_rng(10), 6000)


def test_identical_distributions_give_zero_psi(reference):
    calc = PSICalculator(FEATURES).fit(reference)
    same = calc.psi(reference)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in same.per_feature.values())
    assert same.aggregate_max == pytest.approx(0.0, abs=1e-12)
    fresh = calc.psi(frame(np.random.default_rng(11), 6000))
    assert fresh.aggregate_max < 0.05  # sampling noise only
    assert fresh.n_reference == 6000 and fresh.n_current == 6000


def test_shifted_normal_exceeds_breaker_threshold():
    rng = np.random.default_rng(12)
    ref = pd.DataFrame({"x": rng.normal(0.0, 1.0, 5000)})
    cur = pd.DataFrame({"x": rng.normal(1.0, 1.0, 5000)})
    res = PSICalculator(["x"]).fit(ref).psi(cur)
    assert res.per_feature["x"] > 0.25
    t = res.tables["x"]
    assert list(t.columns) == ["bucket", "expected_share", "actual_share", "contribution"]
    assert (t["contribution"] >= 0).all()
    assert t["contribution"].sum() == pytest.approx(res.per_feature["x"])
    assert t["expected_share"].sum() == pytest.approx(1.0) and t["actual_share"].sum() == pytest.approx(1.0)


def test_numeric_bins_are_reference_deciles_with_open_ends(reference):
    calc = PSICalculator(["dti"], n_bins=10).fit(reference)
    edges = calc.bins["dti"].edges
    assert len(edges) == 9  # 10 buckets from deciles (+ the missing bucket)
    labels = calc.bucket_labels("dti")
    assert labels[0].startswith("(-inf") and labels[-2].endswith("inf)") and labels[-1] == MISSING_BUCKET
    extreme = pd.DataFrame({"dti": [-1e9, 1e9]})
    t = calc.psi(extreme).tables["dti"]
    actual = t["actual_share"].to_numpy()
    assert actual[0] > 0.4 and actual[9] > 0.4  # out-of-range values land in the outer bins
    shares = np.asarray(calc.bins["dti"].expected_counts[:-1]) / 6000
    assert shares == pytest.approx(np.full(10, 0.1), abs=0.01)


def test_categorical_new_level_goes_to_other(reference):
    calc = PSICalculator(["grade"]).fit(reference)
    rng = np.random.default_rng(13)
    cur = frame(rng, 4000, grades=("A", "B", "C", "D", "Z"))
    res = calc.psi(cur)
    t = res.tables["grade"].set_index("bucket")
    assert list(t.index) == ["A", "B", "C", "D", OTHER_BUCKET, MISSING_BUCKET]
    assert t.loc[OTHER_BUCKET, "actual_share"] == pytest.approx(0.2, abs=0.03)
    assert t.loc[OTHER_BUCKET, "expected_share"] < 1e-3
    assert res.per_feature["grade"] > 0.25


def test_missing_feature_is_skipped_and_listed(reference):
    calc = PSICalculator([*FEATURES, "revol_util"]).fit(reference)  # revol_util not in reference
    assert calc.unavailable == ["revol_util"]
    res = calc.psi(reference.drop(columns=["dti"]))
    assert set(res.missing_features) == {"dti", "revol_util"}
    assert "dti" not in res.per_feature and "dti" not in res.tables
    assert set(res.per_feature) == {"fico_range_low", "inq_last_6mths", "grade"}


def test_low_cardinality_numeric_is_categorical_and_dtype_robust():
    rng = np.random.default_rng(14)
    ref = pd.DataFrame({"flag": rng.integers(0, 2, 3000).astype(float)})
    calc = PSICalculator(["flag"]).fit(ref)
    assert calc.bins["flag"].kind == "discrete"
    as_int = pd.DataFrame({"flag": ref["flag"].astype(int)})
    assert calc.psi(as_int).per_feature["flag"] == pytest.approx(0.0, abs=1e-12)
    unseen = pd.DataFrame({"flag": np.r_[np.zeros(500), np.full(500, 2.0)]})
    t = calc.psi(unseen).tables["flag"].set_index("bucket")
    assert t.loc[OTHER_BUCKET, "actual_share"] == pytest.approx(0.5, abs=1e-3)


def test_point_mass_feature_still_gets_two_buckets():
    rng = np.random.default_rng(15)
    x = np.where(rng.random(4000) < 0.95, 0.0, rng.uniform(1, 500, 4000))
    calc = PSICalculator(["delinq_amnt"]).fit(pd.DataFrame({"delinq_amnt": x}))
    assert calc.bins["delinq_amnt"].kind == "numeric"
    assert len(calc.bins["delinq_amnt"].edges) == 1  # {0} vs (0, inf)
    shifted = np.where(rng.random(4000) < 0.70, 0.0, rng.uniform(1, 500, 4000))
    assert calc.psi(pd.DataFrame({"delinq_amnt": shifted})).per_feature["delinq_amnt"] > 0.25


def test_missingness_shift_is_detected(reference):
    calc = PSICalculator(["dti"]).fit(reference)
    cur = reference.copy()
    cur.loc[cur.index[:1800], "dti"] = np.nan
    res = calc.psi(cur)
    t = res.tables["dti"].set_index("bucket")
    assert t.loc[MISSING_BUCKET, "actual_share"] == pytest.approx(0.3, abs=1e-3)
    assert res.per_feature["dti"] > 0.25


def test_psi_matches_hand_computation():
    ref = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})
    cur = pd.DataFrame({"g": ["a"] * 30 + ["b"] * 30 + ["c"] * 40})
    eps = 1e-4
    res = PSICalculator(["g"], eps=eps).fit(ref).psi(cur)
    e = np.array([0.5, 0.3, 0.2, 0.0, 0.0]) + eps
    a = np.array([0.3, 0.3, 0.4, 0.0, 0.0]) + eps
    e, a = e / e.sum(), a / a.sum()
    assert res.per_feature["g"] == pytest.approx(float(np.sum((a - e) * np.log(a / e))))


def test_empty_window_gives_nan(reference):
    res = PSICalculator(FEATURES).fit(reference).psi(reference.iloc[:0])
    assert res.n_current == 0
    assert all(math.isnan(v) for v in res.per_feature.values())
    assert math.isnan(res.aggregate_max) and math.isnan(res.aggregate_mean)
    json.dumps(res.to_dict(), allow_nan=False)


def test_calculator_checkpoint_roundtrip(reference):
    calc = PSICalculator.from_config(PSIConfig(features=tuple(FEATURES), n_bins=8, eps=1e-3)).fit(reference)
    payload = json.dumps(calc.to_dict(), allow_nan=False)
    clone = PSICalculator.from_dict(json.loads(payload))
    cur = frame(np.random.default_rng(16), 3000, shift=0.7)
    a, b = calc.psi(cur), clone.psi(cur)
    assert a.per_feature == b.per_feature
    assert a.aggregate_max == b.aggregate_max and a.aggregate_mean == b.aggregate_mean
    for f in FEATURES:
        pd.testing.assert_frame_equal(a.tables[f], b.tables[f])
    assert a.aggregate("mean") == a.aggregate_mean
    with pytest.raises(ConfigurationError):
        a.aggregate("median")


def test_fit_and_usage_errors(reference):
    with pytest.raises(ConfigurationError):
        PSICalculator(FEATURES).psi(reference)
    with pytest.raises(ConfigurationError):
        PSICalculator(FEATURES).fit(reference.iloc[:0])
    with pytest.raises(ConfigurationError):
        PSICalculator(["nope"]).fit(reference)
    with pytest.raises(ConfigurationError):
        PSICalculator([])
