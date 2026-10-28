"""Outcome derivation, temporal split and summary (SPEC §1.2, §2.12)."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import DataConfig
from bayesian_credit.data.loader import STRING_DTYPE
from bayesian_credit.data.outcomes import (
    add_months,
    derive_outcomes,
    months_between,
    summarize,
    temporal_split,
)
from bayesian_credit.data.schema import OUTCOME_COLUMNS
from bayesian_credit.data.segmentation import Segmenter
from bayesian_credit.data.synthetic import generate
from bayesian_credit.exceptions import DataSourceError

T = pd.Timestamp
NaT = pd.NaT

# status, issue, last_pymnt_d, term, funded, rec_prncp, rec_int, late_fee, recoveries, coll_fee
ROWS = [
    ("Fully Paid", "2016-01-01", "2017-07-01", 36, 10_000, 10_000, 1_500, 0, 0, 0),  # 0
    ("Charged Off", "2016-01-01", "2016-07-01", 36, 10_000, 1_500, 700, 15, 1_000, 160),  # 1
    ("Charged Off", "2018-06-01", None, 36, 5_000, 0, 0, 0, 0, 0),  # 2 never paid
    ("Default", "2018-01-01", "2018-12-01", 60, 20_000, 3_000, 2_000, 0, 0, 0),  # 3 resolution past snapshot
    ("Current", "2018-03-01", "2019-03-01", 36, 8_000, 2_000, 900, 0, 0, 0),  # 4
    ("Late (31-120 days)", "2017-01-01", "2018-12-01", 60, 12_000, 3_000, 2_500, 30, 0, 0),  # 5
    ("Current", "2018-12-01", None, 36, 6_000, 0, 0, 0, 0, 0),  # 6 no payment yet
    ("Does not meet the credit policy. Status:Fully Paid", "2010-01-01", "2013-01-01", 36, 4_000, 4_000, 800, 0, 0, 0),  # 7
    ("Does not meet the credit policy. Status:Charged Off", "2010-01-01", "2011-01-01", 36, 4_000, 1_000, 400, 0, 200, 30),  # 8
    ("Fully Paid", "2015-01-01", "2018-06-01", 36, 9_000, 9_000, 2_000, 0, 0, 0),  # 9 paid after term
    ("Charged Off", "2016-01-01", "2018-01-01", 36, 10_000, 9_000, 1_000, 0, 3_000, 480),  # 10 over-recovery
    ("In Grace Period", "2018-01-01", "2019-02-01", 36, 7_000, 2_000, 600, 0, 0, 0),  # 11
]


@pytest.fixture
def loans():
    cols = ["loan_status", "issue_d", "last_pymnt_d", "term", "funded_amnt", "total_rec_prncp", "total_rec_int",
            "total_rec_late_fee", "recoveries", "collection_recovery_fee"]
    df = pd.DataFrame(ROWS, columns=cols)
    df["issue_d"] = pd.to_datetime(df["issue_d"]).astype("datetime64[ns]")
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"]).astype("datetime64[ns]")
    df["loan_status"] = df["loan_status"].astype(STRING_DTYPE)
    df["loan_amnt"] = df["funded_amnt"].astype(float)
    df["int_rate"] = np.linspace(0.06, 0.29, len(df))
    df["grade"] = pd.array(list("ABCDEFGABCDE"), dtype=STRING_DTYPE)
    df["purpose"] = pd.array(["credit_card", "debt_consolidation", "car"] * 4, dtype=STRING_DTYPE)
    df["home_ownership"] = pd.array(["RENT", "OWN", "MORTGAGE", "ANY"] * 3, dtype=STRING_DTYPE)
    df.insert(0, "loan_id", np.arange(1, len(df) + 1))
    return df


@pytest.fixture
def out(loans):
    return derive_outcomes(loans)


class TestColumns:
    def test_all_outcome_columns_with_dtypes(self, out):
        for col in OUTCOME_COLUMNS:
            assert col in out.columns
        assert out["y_resolved"].dtype == bool
        for col in ("y_default", "y_recovery_rate", "y_realized_revenue", "y_realized_loss", "y_months_on_book"):
            assert out[col].dtype == np.float64, col
        assert out["y_resolution_date"].dtype == np.dtype("datetime64[ns]")
        assert out["y_historical_arm"].dtype == np.int64
        assert out["y_group"].dtype == STRING_DTYPE

    def test_input_untouched_and_idempotent(self, loans, out):
        assert not any(c.startswith("y_") for c in loans.columns)
        again = derive_outcomes(out)
        pd.testing.assert_frame_equal(again, out)

    def test_missing_columns_raise(self, loans):
        with pytest.raises(DataSourceError, match="recoveries"):
            derive_outcomes(loans.drop(columns=["recoveries"]))


class TestDefinitions:
    def test_resolved_and_default_flags(self, out):
        assert out["y_resolved"].tolist() == [True, True, True, True, False, False, False, True, True, True, True, False]
        expected = [0, 1, 1, 1, np.nan, np.nan, np.nan, 0, 1, 0, 1, np.nan]
        np.testing.assert_array_equal(out["y_default"].to_numpy(), np.array(expected, dtype=float))

    def test_recovery_rate_principal_basis(self, out):
        rr = out["y_recovery_rate"]
        assert rr[1] == pytest.approx((1_500 + 1_000 - 160) / 10_000)
        assert rr[2] == 0.0
        assert rr[8] == pytest.approx((1_000 + 200 - 30) / 4_000)
        assert rr[10] == 1.0  # clipped
        assert rr[out["y_default"] != 1].isna().all()
        assert rr.dropna().between(0, 1).all()

    def test_recovery_rate_post_chargeoff_basis(self, loans):
        out = derive_outcomes(loans, cfg=DataConfig(recovery_basis="post_chargeoff"))
        rr = out["y_recovery_rate"]
        assert rr[1] == pytest.approx((1_000 - 160) / (10_000 - 1_500))
        assert rr[2] == 0.0
        assert rr[10] == 1.0  # (3000 - 480) / 1000 clipped
        assert rr[0] != rr[0]  # NaN for non-defaults

    def test_revenue_and_loss(self, out):
        assert out.loc[1, "y_realized_revenue"] == 715.0
        assert out.loc[0, "y_realized_revenue"] == 1_500.0
        assert out.loc[1, "y_realized_loss"] == pytest.approx(10_000 - 1_500 - (1_000 - 160))
        assert out.loc[2, "y_realized_loss"] == 5_000.0
        assert out.loc[10, "y_realized_loss"] == 0.0  # max(., 0)
        assert out.loc[0, "y_realized_loss"] == 0.0 and out.loc[7, "y_realized_loss"] == 0.0
        assert np.isnan(out.loc[4, "y_realized_loss"]) and np.isnan(out.loc[5, "y_realized_loss"])

    def test_months_on_book(self, out):
        mob = out["y_months_on_book"].tolist()
        assert mob[0] == 18 and mob[1] == 6
        assert mob[2] == 1  # resolved, never paid -> floor
        assert mob[4] == 12 and mob[5] == 23
        assert mob[6] == 3  # unresolved, never paid -> issue..snapshot
        assert mob[9] == 36  # capped at term
        assert min(mob) >= 1

    def test_resolution_date(self, out):
        res = out["y_resolution_date"]
        assert res[0] == T("2017-07-01")
        assert res[1] == T("2016-11-01")  # last payment + 4 months
        assert res[2] == T("2018-10-01")  # issue + 4 when no payment
        assert res[3] == T("2019-03-31")  # 2019-04-01 clipped to the snapshot
        assert res[7] == T("2013-01-01") and res[8] == T("2011-05-01")
        assert res[[4, 5, 6, 11]].isna().all()

    def test_lag_and_snapshot_configurable(self, loans):
        out = derive_outcomes(loans, cfg=DataConfig(chargeoff_lag_months=6), snapshot_date="2025-01-01")
        assert out.loc[1, "y_resolution_date"] == T("2017-01-01")
        assert out.loc[3, "y_resolution_date"] == T("2019-06-01")
        assert out.loc[6, "y_months_on_book"] == 36  # far snapshot -> capped at term

    def test_historical_arm_and_group(self, loans, out, arm_space):
        for i, row in loans.iterrows():
            assert out.loc[i, "y_historical_arm"] == arm_space.nearest(row.loan_amnt, row.int_rate, row.term)
        assert out["y_group"].tolist() == Segmenter(["grade", "purpose_group", "home_ownership"]).assign(loans).tolist()
        assert out.loc[3, "y_group"] == "D|credit_card|OTHER"

    def test_custom_arm_space_and_segmenter(self, loans, small_arm_space):
        out = derive_outcomes(loans, small_arm_space, Segmenter(["grade"]))
        assert set(out["y_historical_arm"]) <= set(small_arm_space.lending_arms)
        assert out["y_group"].tolist() == list("ABCDEFGABCDE")

    def test_unknown_status_warns_and_is_unresolved(self, loans):
        loans.loc[0, "loan_status"] = "Issued"
        with pytest.warns(UserWarning, match="Issued"):
            out = derive_outcomes(loans)
        assert not out.loc[0, "y_resolved"] and np.isnan(out.loc[0, "y_default"])


@pytest.fixture(scope="module")
def synth():
    return derive_outcomes(generate(20_000, seed=13))


class TestSplitAndSummary:
    def test_temporal_split(self, synth):
        train, stream = temporal_split(synth, "2015-01-01")
        assert len(train) + len(stream) == len(synth)
        assert train["issue_d"].max() < T("2015-01-01") <= stream["issue_d"].min()
        assert train.index.equals(pd.RangeIndex(len(train)))
        _, capped = temporal_split(synth, "2015-01-01", stream_end="2016-12-31")
        assert capped["issue_d"].max() <= T("2016-12-31") and len(capped) < len(stream)

    def test_summarize(self, synth):
        s = summarize(synth)
        json.dumps(s)
        assert s["n"] == len(synth) == s["n_unique_loans"]
        assert s["issue_min"] == "2012-01-01" and s["issue_max"] == "2018-12-01"
        assert sum(s["status_counts"].values()) == len(synth)
        assert s["n_resolved"] == int(synth["y_resolved"].sum())
        assert s["resolved_share"] == pytest.approx(synth["y_resolved"].mean())
        assert s["default_rate"] == pytest.approx(synth["y_default"].mean())
        assert s["mean_recovery"] == pytest.approx(synth["y_recovery_rate"].mean())
        assert set(s["default_rate_by_grade"]) == set("ABCDEFG")
        assert sum(s["term_mix"].values()) == pytest.approx(1.0)

    def test_summarize_without_outcome_columns_matches(self, synth):
        raw = synth.drop(columns=list(OUTCOME_COLUMNS))
        a, b = summarize(raw), summarize(synth)
        for key in ("default_rate", "resolved_share", "mean_recovery", "n_defaults"):
            assert a[key] == pytest.approx(b[key])

    def test_summarize_empty(self, synth):
        s = summarize(synth.iloc[:0])
        assert s["n"] == 0 and s["default_rate"] is None
        json.dumps(s)


class TestMonthHelpers:
    def test_months_between(self):
        a = pd.Series([T("2015-12-01"), T("2016-01-01"), NaT])
        b = pd.Series([T("2016-03-01"), NaT, T("2016-01-01")])
        np.testing.assert_array_equal(months_between(a, b), [3, np.nan, np.nan])
        np.testing.assert_array_equal(months_between(a, T("2019-03-31")), [39, 38, np.nan])

    def test_add_months(self):
        a = pd.Series([T("2015-12-01"), T("2016-01-15"), NaT])
        out = add_months(a, [1, 13, 2])
        assert out.tolist()[:2] == [T("2016-01-01"), T("2017-02-01")] and pd.isna(out[2])
        assert add_months(a, 4).dtype == np.dtype("datetime64[ns]")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            add_months(a, np.array([np.nan, 1, 1]))
