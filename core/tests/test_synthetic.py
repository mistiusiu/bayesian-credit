"""SyntheticLendingClub: schema, internal consistency, calibration, shift (SPEC §1.3)."""

import time

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.data.arms import annuity_installment
from bayesian_credit.data.loader import CANONICAL_COLUMNS, STRING_DTYPE, validate_loan_table
from bayesian_credit.data.outcomes import derive_outcomes, months_between, summarize
from bayesian_credit.data.schema import (
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    STATUS_DEFAULTED,
    STATUS_DELINQUENT,
    STATUS_FULLY_PAID,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED,
)
from bayesian_credit.data.synthetic import (
    DEFAULT_SHIFT,
    GRADE_SHARES,
    GRADES,
    SUB_GRADES,
    SyntheticLendingClub,
    generate,
    generate_table,
)
from bayesian_credit.exceptions import DataSourceError

SNAPSHOT = pd.Timestamp("2019-03-31")
N = 60_000


@pytest.fixture(scope="module")
def table():
    return SyntheticLendingClub(seed=7).generate(N)


@pytest.fixture(scope="module")
def derived(table):
    return derive_outcomes(table)


class TestSchema:
    def test_canonical_columns_and_dtypes(self, table):
        assert list(table.columns) == list(CANONICAL_COLUMNS)
        validate_loan_table(table)
        for col in CATEGORICAL_COLUMNS:
            assert table[col].dtype == STRING_DTYPE
        for col in DATE_COLUMNS:
            assert table[col].dtype == np.dtype("datetime64[ns]")
        assert table["term"].dtype == np.int64 and table["loan_id"].dtype == np.int64

    def test_ids_unique_and_increase_with_issue_date(self, table):
        assert table["loan_id"].is_unique and table["loan_id"].is_monotonic_increasing
        assert table["issue_d"].is_monotonic_increasing

    def test_issue_window_and_growth(self, table):
        assert table["issue_d"].min() == pd.Timestamp("2012-01-01")
        assert table["issue_d"].max() == pd.Timestamp("2018-12-01")
        assert (table["issue_d"].dt.day == 1).all()
        per_year = table["issue_d"].dt.year.value_counts().sort_index()
        assert per_year.is_monotonic_increasing and per_year.iloc[-1] > 3 * per_year.iloc[0]

    def test_statuses_are_lc_vocabulary(self, table):
        assert set(table["loan_status"]) <= STATUS_RESOLVED | STATUS_UNRESOLVED
        for status in ("Fully Paid", "Charged Off", "Current", "Late (31-120 days)", "In Grace Period",
                       "Late (16-30 days)", "Default"):
            assert (table["loan_status"] == status).any(), status

    def test_deterministic_per_seed(self):
        a, b = generate(2_000, seed=3), generate(2_000, seed=3)
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(generate(2_000, seed=4))
        pd.testing.assert_frame_equal(generate_table(500, seed=1), SyntheticLendingClub(1).generate_table(500))

    def test_metadata(self, table):
        meta = table.attrs["synthetic"]
        assert meta["n"] == N and meta["seed"] == 7 and meta["shift"] is None

    def test_generation_speed_200k(self):
        t0 = time.perf_counter()
        df = SyntheticLendingClub(seed=0).generate(200_000)
        assert len(df) == 200_000
        assert time.perf_counter() - t0 < 10.0


class TestMarginals:
    def test_fico_bands(self, table):
        low, high = table["fico_range_low"], table["fico_range_high"]
        assert (low % 5 == 0).all() and (high == low + 4).all()
        assert low.min() >= 660 and low.max() <= 845
        assert 690 < low.mean() < 706 and 28 < low.std() < 38

    def test_amounts_terms_rates(self, table):
        amt = table["loan_amnt"]
        assert amt.between(1_000, 40_000).all() and (amt % 25 == 0).all()
        assert (table["funded_amnt"] == amt).all()
        assert 0.22 < (table["term"] == 60).mean() < 0.28
        assert table.loc[table.term == 60, "loan_amnt"].mean() > table.loc[table.term == 36, "loan_amnt"].mean()
        assert table["int_rate"].between(0.05, 0.31).all()
        expected = [annuity_installment(a, r, t) for a, r, t in table[["loan_amnt", "int_rate", "term"]].head(200).to_numpy()]
        np.testing.assert_allclose(table["installment"].head(200), expected, atol=0.006)

    def test_grade_mix(self, table):
        share = table["grade"].value_counts(normalize=True)
        for g in GRADES:
            assert share[g] == pytest.approx(GRADE_SHARES[g], abs=0.015), g
        assert set(table["sub_grade"]) <= set(SUB_GRADES)
        assert (table["sub_grade"].str[0] == table["grade"]).all()

    def test_grade_to_rate_is_monotone(self, table):
        by_grade = table.groupby("grade")["int_rate"].mean().loc[list(GRADES)]
        assert by_grade.is_monotonic_increasing
        by_sub = table.groupby("sub_grade")["int_rate"].mean().loc[[s for s in SUB_GRADES if s in set(table.sub_grade)]]
        assert by_sub.is_monotonic_increasing
        assert by_sub.iloc[0] == pytest.approx(0.053, abs=0.004) and by_sub.iloc[-1] == pytest.approx(0.309, abs=0.006)

    def test_categorical_mixes(self, table):
        home = table["home_ownership"].value_counts(normalize=True)
        assert home["MORTGAGE"] == pytest.approx(0.49, abs=0.03) and home["RENT"] == pytest.approx(0.40, abs=0.03)
        assert set(home.index) <= {"MORTGAGE", "RENT", "OWN", "OTHER"}
        purpose = table["purpose"].value_counts(normalize=True)
        assert purpose["debt_consolidation"] == pytest.approx(0.566, abs=0.02)
        assert purpose["credit_card"] == pytest.approx(0.229, abs=0.02)
        assert set(table["verification_status"]) == {"Not Verified", "Source Verified", "Verified"}
        assert table["emp_length"].isna().mean() == pytest.approx(0.065, abs=0.01)
        assert set(table["emp_length"].dropna()) <= set(map(float, range(11)))

    def test_joint_fields_only_for_joint_apps(self, table):
        joint = table["application_type"] == "Joint App"
        assert joint.mean() == pytest.approx(0.05, abs=0.01)
        for col in ("annual_inc_joint", "dti_joint", "verified_status_joint"):
            assert table.loc[joint, col].notna().all() and table.loc[~joint, col].isna().all(), col
        assert (table.loc[joint, "annual_inc_joint"] > table.loc[joint, "annual_inc"]).all()

    def test_bureau_fields_added_late_2015_are_missing_before(self, table):
        early = table["issue_d"] < "2015-12-01"
        for col in ("inq_last_12m", "all_util"):
            assert table.loc[early, col].isna().all() and table.loc[~early, col].notna().all()

    def test_income_dti_util(self, table):
        assert table["annual_inc"].median() == pytest.approx(65_000, rel=0.05)
        assert 16 < table["dti"].mean() < 21 and table["dti"].between(0, 45).all()
        assert table["revol_util"].mean() == pytest.approx(0.50, abs=0.03)
        assert table["mths_since_last_delinq"].isna().mean() == pytest.approx(0.5, abs=0.08)


class TestConsistency:
    def test_dates(self, table):
        assert (table["last_pymnt_d"].dropna() <= SNAPSHOT).all()
        has = table["last_pymnt_d"].notna()
        assert (table.loc[has, "last_pymnt_d"] > table.loc[has, "issue_d"]).all()
        resolved = table["loan_status"].isin(STATUS_RESOLVED)
        assert (table.loc[resolved, "last_pymnt_d"].dropna() <= SNAPSHOT).all()
        assert table.loc[resolved, "next_pymnt_d"].isna().all()
        assert (table.loc[~resolved, "next_pymnt_d"] == pd.Timestamp("2019-04-01")).all()

    def test_current_loans_have_outstanding_principal(self, table):
        cur = table["loan_status"] == "Current"
        assert (table.loc[cur, "out_prncp"] > 0).all()
        assert (table.loc[cur, "last_pymnt_d"] >= pd.Timestamp("2019-03-01")).all()
        assert (table["out_prncp"] == table["out_prncp_inv"]).all()

    def test_prepayment_is_front_loaded(self, table):
        fp = table["loan_status"].isin(STATUS_FULLY_PAID)
        months = months_between(table.loc[fp, "issue_d"], table.loc[fp, "last_pymnt_d"])
        term = table.loc[fp, "term"].to_numpy()
        early = months < term
        assert (months[early] >= 6).all() and (months[early] <= term[early] - 1).all()
        # fully observed cohorts (36m issued before 2015) avoid the censoring bias toward early payoffs
        observed = (table.loc[fp, "issue_d"] < "2015-01-01").to_numpy() & (term == 36) & early
        pos = (months[observed] - 6) / (term[observed] - 7)  # position within [6, term - 1]
        assert observed.sum() > 1_000
        assert np.median(pos) < 0.35  # Beta(1, 2.5): median ~0.24; uniform timing would give ~0.5
        assert 0.40 < observed.sum() / ((table.loc[fp, "issue_d"] < "2015-01-01").to_numpy() & (term == 36)).sum() < 0.50

    def test_fully_paid(self, table):
        fp = table["loan_status"].isin(STATUS_FULLY_PAID)
        np.testing.assert_allclose(table.loc[fp, "total_rec_prncp"], table.loc[fp, "funded_amnt"])
        assert (table.loc[fp, "out_prncp"] == 0).all() and (table.loc[fp, "recoveries"] == 0).all()
        months = months_between(table.loc[fp, "issue_d"], table.loc[fp, "last_pymnt_d"])
        assert (months >= 6).all() and (months <= table.loc[fp, "term"].to_numpy()).all()

    def test_charged_off(self, table):
        co = table["loan_status"].isin(STATUS_DEFAULTED)
        assert (table.loc[co, "recoveries"] >= 0).all()
        assert (table.loc[co, "total_rec_prncp"] < table.loc[co, "funded_amnt"]).all()
        assert (table.loc[co, "out_prncp"] == 0).all()
        np.testing.assert_allclose(
            table.loc[co, "collection_recovery_fee"], 0.16 * table.loc[co, "recoveries"], atol=0.006
        )
        assert table.loc[co, "last_pymnt_d"].isna().mean() < 0.03  # few defaults never paid
        months_since = months_between(table.loc[co, "last_pymnt_d"].dropna(), SNAPSHOT)
        assert (months_since > 4).all()

    def test_late_statuses(self, table):
        late = table["loan_status"].isin(STATUS_DELINQUENT)
        assert (table.loc[late, "num_tl_30dpd"] == 1).all() and (table.loc[~late, "num_tl_30dpd"] == 0).all()
        assert (table.loc[late, "out_prncp"] > 0).all()
        paid = late & table["last_pymnt_d"].notna()
        assert ((m := months_between(table.loc[paid, "last_pymnt_d"], SNAPSHOT)) >= 1).all() and (m <= 4).all()
        never_paid = late & table["last_pymnt_d"].isna()  # defaulted on the first installment
        assert (months_between(table.loc[never_paid, "issue_d"], SNAPSHOT) <= 4).all()

    def test_cash_flow_identities(self, table):
        total = table["total_rec_prncp"] + table["total_rec_int"] + table["total_rec_late_fee"] + table["recoveries"]
        np.testing.assert_allclose(table["total_pymnt"], total, atol=0.02)
        assert (table["total_pymnt_inv"] == table["total_pymnt"]).all()
        assert (table["total_rec_int"] >= 0).all()
        fees = table["total_rec_late_fee"]
        assert fees[fees > 0].between(15, 45).all()
        assert (table["last_fico_range_high"] == table["last_fico_range_low"] + 4).all()

    def test_defaulters_last_fico_drops(self, table):
        bad = table["loan_status"].isin(STATUS_DEFAULTED)
        drop = (table["fico_range_low"] - table["last_fico_range_low"])
        assert drop[bad].mean() > 60 and abs(drop[table["loan_status"] == "Fully Paid"].mean()) < 10


class TestCalibration:
    def test_default_rate_among_resolved(self, derived):
        s = summarize(derived)
        assert 0.12 <= s["default_rate"] <= 0.28
        assert 0.155 <= s["default_rate"] <= 0.20  # calibrated: ~17.5% with front-loaded prepayment

    def test_cohort_default_rates_track_lending_club(self):
        # real LC resolved default rate by issue year: 2012-15 16/16/18/20%, 2016-18 23/23/16%
        by_year = summarize(derive_outcomes(SyntheticLendingClub(seed=3).generate(150_000)))["default_rate_by_issue_year"]
        for year in ("2012", "2013", "2014", "2015"):
            assert 0.12 <= by_year[year] <= 0.19, year
        assert 0.19 <= by_year["2016"] <= 0.26
        assert 0.18 <= by_year["2017"] <= 0.26
        assert 0.11 <= by_year["2018"] <= 0.19
        assert by_year["2018"] < by_year["2017"]  # censoring: few 2018 defaults charged off yet

    def test_default_rate_increases_with_grade(self, derived):
        by_grade = summarize(derived)["default_rate_by_grade"]
        rates = [by_grade[g] for g in GRADES]
        assert rates == sorted(rates)
        assert rates[0] < 0.10 and rates[-1] > 0.35

    def test_recovery_and_late_fee_shares(self, derived):
        s = summarize(derived)
        assert 0.25 < s["mean_recovery"] < 0.45
        fp = derived["loan_status"] == "Fully Paid"
        assert 0.01 < (derived.loc[fp, "total_rec_late_fee"] > 0).mean() < 0.05

    def test_fico_predicts_default(self, derived):
        r = derived[derived["y_resolved"]]
        low = r["fico_range_low"] < r["fico_range_low"].median()
        assert r.loc[low, "y_default"].mean() > 1.3 * r.loc[~low, "y_default"].mean()


class TestShift:
    def test_shift_lowers_fico_and_raises_defaults(self):
        gen = SyntheticLendingClub(seed=21)
        base = gen.generate(80_000)
        base_meta = dict(gen.last_meta)
        shifted = gen.generate(80_000, shift_at="2016-06", shift=DEFAULT_SHIFT)
        shift_meta = dict(gen.last_meta)
        post = (base["issue_d"] >= "2016-06-01").to_numpy()
        assert shift_meta["n_shifted"] == int(post.sum())
        assert shifted.loc[post, "fico_range_low"].mean() == pytest.approx(base.loc[post, "fico_range_low"].mean() - 30, abs=1)
        assert shifted.loc[post, "dti"].mean() > base.loc[post, "dti"].mean() + 5
        assert shifted.loc[post, "annual_inc"].median() < base.loc[post, "annual_inc"].median()
        assert shifted.loc[post, "int_rate"].mean() > base.loc[post, "int_rate"].mean()
        assert shift_meta["lifetime_default_share"] > base_meta["lifetime_default_share"] + 0.02
        b, s = derive_outcomes(base), derive_outcomes(shifted)
        window = (b["issue_d"] >= "2016-06-01") & (b["issue_d"] <= "2017-06-01")
        assert s.loc[window & s.y_resolved, "y_default"].mean() > b.loc[window & b.y_resolved, "y_default"].mean()

    def test_pre_shift_cohort_is_unchanged(self):
        gen = SyntheticLendingClub(seed=4)
        base = gen.generate(20_000)
        pre = (base["issue_d"] < "2016-06-01").to_numpy()
        for shift in (DEFAULT_SHIFT, {"latent": -0.3}, {"grade_mix": "worse", "loan_amnt": 2_000, "open_acc": 2}):
            other = gen.generate(20_000, shift_at="2016-06", shift=shift)
            pd.testing.assert_frame_equal(base[pre], other[pre])

    def test_grade_mix_worse(self):
        df = SyntheticLendingClub(seed=2).generate(40_000, shift_at="2016-01", shift={"grade_mix": "worse"})
        post = df["issue_d"] >= "2016-01-01"
        share_a = df.groupby(post)["grade"].apply(lambda s: (s == "A").mean())
        share_e_plus = df.groupby(post)["grade"].apply(lambda s: s.isin(["E", "F", "G"]).mean())
        assert share_a[True] < share_a[False] - 0.05 and share_e_plus[True] > share_e_plus[False] + 0.05

    def test_generic_numeric_shift(self):
        gen = SyntheticLendingClub(seed=9)
        base, shifted = gen.generate(5_000), gen.generate(5_000, shift_at="2017-01", shift={"open_acc": 3})
        post = base["issue_d"] >= "2017-01-01"
        assert (shifted.loc[post, "open_acc"] == base.loc[post, "open_acc"] + 3).all()

    def test_default_shift_when_only_shift_at_given(self):
        gen = SyntheticLendingClub(seed=1)
        gen.generate(1_000, shift_at="2017-01")
        assert gen.last_meta["shift"] == DEFAULT_SHIFT and gen.last_meta["shift_at"] == "2017-01-01"


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(n=0),
            dict(n=10, start="2018-01", end="2017-01"),
            dict(n=10, end="2019-06", snapshot="2019-03-31"),
            dict(n=10, shift_at="2016-01", shift={"zip_code": 1}),
            dict(n=10, shift_at="2016-01", shift={"installment": 10}),
            dict(n=10, shift_at="2016-01", shift={"grade_mix": "sideways"}),
            dict(n=10, shift_at="2016-01", shift={"dti": float("nan")}),
        ],
    )
    def test_bad_arguments(self, kwargs):
        with pytest.raises(DataSourceError):
            SyntheticLendingClub(seed=0).generate(**kwargs)

    def test_custom_window_and_snapshot(self):
        df = generate(3_000, seed=5, start="2014-01", end="2014-12", snapshot="2016-12-31")
        assert df["issue_d"].between("2014-01-01", "2014-12-01").all()
        assert (df["last_pymnt_d"].dropna() <= pd.Timestamp("2016-12-31")).all()
        assert (df.loc[df.loan_status == "Current", "next_pymnt_d"] == pd.Timestamp("2017-01-01")).all()
