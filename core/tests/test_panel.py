"""LoanStatePanel - reconstructed monthly credit-state paths (SPEC §2.6, Appendix G.2)."""

import json

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import MCTMConfig
from bayesian_credit.data.loader import STRING_DTYPE
from bayesian_credit.data.panel import (
    LoanStatePanel,
    interim_state,
    paths_to_counts,
    row_segments,
)
from bayesian_credit.data.schema import STATUS_FULLY_PAID
from bayesian_credit.data.synthetic import generate
from bayesian_credit.exceptions import DataSourceError
from bayesian_credit.types import State, TransitionStructure

C, D, R, L = (int(s) for s in State)
SNAP = "2019-03-31"
PROPOSAL = MCTMConfig(structure=TransitionStructure.PROPOSAL)
EXTENDED = MCTMConfig(structure=TransitionStructure.EXTENDED)


def make_table(rows):
    df = pd.DataFrame(rows, columns=["loan_id", "loan_status", "issue_d", "last_pymnt_d", "term", "total_rec_late_fee"])
    df["loan_status"] = df["loan_status"].astype(STRING_DTYPE)
    df["issue_d"] = pd.to_datetime(df["issue_d"]).astype("datetime64[ns]")
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"]).astype("datetime64[ns]")
    return df


# loan_id, status, issue, last payment, term, late fees
HAND = make_table(
    [
        (1, "Fully Paid", "2016-01-01", "2016-07-01", 36, 0.0),  # never late: [C6] (+R under EXTENDED)
        (2, "Charged Off", "2016-01-01", "2016-04-01", 36, 15.0),  # [C3][D4][L1] (fees of the default)
        (3, "Current", "2018-01-01", "2019-03-01", 36, 15.0),  # 14 months + fee -> cured [C7][D1][C7]
        (4, "Late (31-120 days)", "2017-06-01", "2018-12-01", 60, 0.0),  # [C18][D3]
        (5, "Charged Off", "2018-06-01", None, 36, 0.0),  # never paid -> [D4][L1]
        (6, "Fully Paid", "2017-01-01", "2018-01-01", 36, 25.0),  # paid late fees -> [C11][D1][R1] (both)
    ]
)
HAND_PATHS = {
    1: [(C, 6)],
    2: [(C, 3), (D, 4), (L, 1)],
    3: [(C, 7), (D, 1), (C, 7)],
    4: [(C, 18), (D, 3)],
    5: [(D, 4), (L, 1)],
    6: [(C, 11), (D, 1), (R, 1)],
}


def expected_hand(structure):
    m = np.zeros((4, 4))
    m[C, C] = 5 + 2 + 12 + 17 + 10
    m[C, D] = 4  # loans 2, 3, 4, 6
    m[D, C] = 1  # loan 3 (cured while on book)
    m[D, D] = 3 + 2 + 3
    m[D, L] = 2
    m[D, R] = 1  # loan 6 (repayment following delinquency) - under BOTH structures
    if structure == TransitionStructure.EXTENDED:
        m[C, R] = 1  # loan 1 (never-late payoff)
    return m


def expand(segments):
    return [s for s, run in segments for _ in range(run)]


@pytest.fixture(scope="module")
def synth():
    return generate(30_000, seed=17)


class TestHandBuiltPanel:
    @pytest.mark.parametrize("cfg", [PROPOSAL, EXTENDED], ids=["proposal", "extended"])
    def test_exact_counts(self, cfg):
        panel = LoanStatePanel.from_table(HAND, cfg, SNAP)
        np.testing.assert_array_equal(panel.counts, expected_hand(cfg.structure))
        extra = 1 if cfg.structure == TransitionStructure.EXTENDED else 0
        assert panel.n_loans == 6
        assert panel.n_loan_months == 68 + extra
        assert panel.counts.sum() == panel.n_loan_months - panel.n_loans
        assert panel.n_censored == (2 if extra else 3)  # loan 6 is absorbed in R under both
        assert panel.n_skipped == 0

    def test_paths(self):
        panel = LoanStatePanel.from_table(HAND, PROPOSAL, SNAP, keep_paths=True)
        assert list(panel.paths.columns) == ["loan_id", "segments"]
        assert dict(zip(panel.paths.loan_id, panel.paths.segments, strict=True)) == HAND_PATHS
        ext = LoanStatePanel.from_table(HAND, EXTENDED, SNAP, keep_paths=True)
        assert ext.paths.segments.iloc[0] == [(C, 6), (R, 1)]
        assert ext.paths.segments.iloc[5] == [(C, 11), (D, 1), (R, 1)]
        for row in HAND.to_dict("records"):
            assert row_segments(row, 4, TransitionStructure.PROPOSAL, SNAP) == HAND_PATHS[row["loan_id"]]
        assert LoanStatePanel.from_table(HAND, PROPOSAL, SNAP).paths is None

    def test_paths_to_counts(self):
        assert np.array_equal(paths_to_counts(list(HAND_PATHS.values())), expected_hand(TransitionStructure.PROPOSAL))
        single = paths_to_counts([(C, 3), (D, 4), (L, 1)])
        assert single[C, C] == 2 and single[C, D] == 1 and single[D, D] == 3 and single[D, L] == 1
        assert paths_to_counts([]).sum() == 0
        assert paths_to_counts([[(C, 2), (C, 3)]])[C, C] == 4  # adjacent runs of one state merge

    def test_cohort_counts(self):
        panel = LoanStatePanel.from_table(HAND, PROPOSAL, SNAP)
        assert set(panel.counts_by_cohort) == {2016, 2017, 2018}
        np.testing.assert_array_equal(sum(panel.counts_by_cohort.values()), panel.counts)
        np.testing.assert_array_equal(
            panel.counts_by_cohort[2017], paths_to_counts([HAND_PATHS[4], HAND_PATHS[6]])
        )

    def test_chargeoff_lag_is_configurable(self):
        panel = LoanStatePanel.from_table(HAND.iloc[[1]], MCTMConfig(chargeoff_lag_months=2), SNAP, keep_paths=True)
        assert panel.paths.segments.iloc[0] == [(C, 3), (D, 2), (L, 1)]

    def test_interim_state_matches_segments(self):
        panel = LoanStatePanel.from_table(HAND, PROPOSAL, SNAP)
        for row in HAND.to_dict("records"):
            path = expand(HAND_PATHS[row["loan_id"]])
            for month in range(1, len(path) + 1):
                state, paid = panel.interim_state(row, month)
                assert state == path[month - 1], (row["loan_id"], month)
                bonus = 1 if path[month - 1] == R and path[month - 2] == D else 0
                assert paid == path[:month].count(C) + bonus

    def test_repaid_after_delinquency_interim_path(self):
        row = HAND.to_dict("records")[5]  # 12 payment months, late fees -> [C11][D1][R1]
        assert interim_state(row, 11, snapshot_date=SNAP) == (State.CURRENT, 11)
        assert interim_state(row, 12, snapshot_date=SNAP) == (State.DELINQUENT, 11)  # late payoff month
        assert interim_state(row, 13, snapshot_date=SNAP) == (State.RECOVERED, 12)  # all 12 honoured
        assert interim_state(row, 40, EXTENDED, SNAP) == (State.RECOVERED, 12)

    def test_interim_state_after_the_path(self):
        rows = HAND.to_dict("records")
        assert interim_state(rows[0], 7, snapshot_date=SNAP) == (State.RECOVERED, 6)  # paid off
        assert interim_state(rows[0], 50, EXTENDED, SNAP) == (State.RECOVERED, 6)
        assert interim_state(rows[1], 20, snapshot_date=SNAP) == (State.LOSS, 3)
        assert interim_state(rows[2], 30, snapshot_date=SNAP) == (State.CURRENT, 14)  # censored: carry forward
        assert interim_state(rows[3], 40, snapshot_date=SNAP) == (State.DELINQUENT, 18)
        assert interim_state(rows[4], 1, snapshot_date=SNAP) == (State.DELINQUENT, 0)

    def test_interim_state_accepts_series_and_namedtuples(self):
        series = HAND.iloc[3]
        tup = next(HAND.iloc[[3]].itertuples(index=False))
        assert interim_state(series, 19, snapshot_date=SNAP) == interim_state(tup, 19, snapshot_date=SNAP)
        assert interim_state(series, 19, snapshot_date=SNAP) == (State.DELINQUENT, 18)

    def test_interim_states_vectorised_on_hand_table(self):
        panel = LoanStatePanel.from_table(HAND, PROPOSAL, SNAP)
        for month in (1, 6, 7, 11, 12, 13, 25):
            states, paid = panel.interim_states(HAND, month)
            expected = [tuple(map(int, panel.interim_state(r, month))) for r in HAND.to_dict("records")]
            assert list(zip(states.tolist(), paid.tolist(), strict=True)) == expected, month

    def test_month_must_be_positive(self):
        panel = LoanStatePanel.from_table(HAND, PROPOSAL, SNAP)
        with pytest.raises(ValueError):
            panel.interim_state(HAND.iloc[0], 0)
        with pytest.raises(ValueError):
            panel.interim_states(HAND, 0)


class TestStatusRules:
    def test_other_statuses(self):
        table = make_table(
            [
                (10, "In Grace Period", "2018-09-01", "2019-02-01", 36, 0.0),
                (11, "Late (16-30 days)", "2018-09-01", "2019-02-01", 36, 25.0),
                (12, "Default", "2017-01-01", "2018-01-01", 36, 0.0),
                (13, "Does not meet the credit policy. Status:Fully Paid", "2010-01-01", "2013-01-01", 36, 0.0),
                (14, "Does not meet the credit policy. Status:Charged Off", "2010-01-01", "2010-03-01", 36, 0.0),
                (15, "Fully Paid", "2016-01-01", "2016-02-01", 36, 30.0),  # 1 month + fee: D -> R
                (16, "Late (31-120 days)", "2018-11-01", None, 36, 0.0),  # never paid
                (17, "Late (31-120 days)", "2016-01-01", "2018-09-01", 36, 0.0),  # 6 months behind -> capped
                (18, "Fully Paid", "2015-01-01", "2018-06-01", 36, 0.0),  # paid after term -> capped
                (19, "Current", "2018-12-01", None, 36, 0.0),  # no payment yet
                (20, "Issued", "2019-03-01", None, 36, 0.0),  # unknown status -> skipped
            ]
        )
        panel = LoanStatePanel.from_table(table, PROPOSAL, SNAP, keep_paths=True)
        got = dict(zip(panel.paths.loan_id, panel.paths.segments, strict=True))
        assert got[10] == [(C, 5), (D, 1)]
        assert got[11] == [(C, 5), (D, 1)]  # late fee of the current episode: no cure
        assert got[12] == [(C, 12), (D, 4), (L, 1)]
        assert got[13] == [(C, 36)]
        assert got[14] == [(C, 2), (D, 4), (L, 1)]
        assert got[15] == [(D, 1), (R, 1)]
        assert got[16] == [(D, 4)]
        assert got[17] == [(C, 32), (D, 4)]
        assert got[18] == [(C, 36)]
        assert got[19] == [(C, 3)]
        assert got[20] == []
        assert panel.n_skipped == 1 and panel.n_loans == 10
        for row in table.to_dict("records"):
            assert row_segments(row, 4, TransitionStructure.PROPOSAL, SNAP) == got[row["loan_id"]]

    def test_missing_columns(self):
        with pytest.raises(DataSourceError, match="total_rec_late_fee"):
            LoanStatePanel.from_table(HAND.drop(columns=["total_rec_late_fee"]))


@pytest.fixture(scope="module")
def panels(synth):
    return (
        LoanStatePanel.from_table(synth, PROPOSAL, keep_paths=True),
        LoanStatePanel.from_table(synth, EXTENDED, keep_paths=True),
    )


class TestSyntheticPanel:
    def test_identity_and_absorbing_rows(self, panels):
        for panel in panels:
            assert panel.counts.sum() == panel.n_loan_months - panel.n_loans
            assert panel.counts[R].sum() == 0 and panel.counts[L].sum() == 0
            assert panel.counts[C, L] == 0
            assert sum(len(s) > 0 for s in panel.paths.segments) == panel.n_loans
            assert sum(run for segs in panel.paths.segments for _, run in segs) == panel.n_loan_months

    def test_d_to_r_is_estimated_from_data(self, panels, synth):
        paid = synth["loan_status"].isin(STATUS_FULLY_PAID)
        fee = synth["total_rec_late_fee"] > 0
        cur_fee = (synth["loan_status"] == "Current") & fee
        for panel in panels:
            assert panel.counts[D, R] > 0
            assert panel.counts[D, R] == int((paid & fee).sum())  # one D -> R per repaid-after-late loan
            assert 0 < panel.counts[D, C] <= int(cur_fee.sum())  # cures come from loans still on book
            p = panel.empirical_matrix()
            assert 0 < p[D, R] < p[D, L]

    def test_structures_differ_only_in_c_to_r(self, panels, synth):
        prop, ext = panels
        diff = ext.counts - prop.counts
        clean_paid = synth["loan_status"].isin(STATUS_FULLY_PAID) & (synth["total_rec_late_fee"] <= 0)
        n_clean = int(clean_paid.sum())
        assert diff[C, R] == n_clean and prop.counts[C, R] == 0
        diff[C, R] = 0
        assert not diff.any()
        assert ext.n_loan_months - prop.n_loan_months == n_clean
        assert prop.n_censored - ext.n_censored == n_clean

    def test_paths_reproduce_counts(self, panels):
        for panel in panels:
            np.testing.assert_array_equal(paths_to_counts(panel.paths.segments.tolist()), panel.counts)

    def test_cohorts_sum_to_total(self, panels):
        prop, _ = panels
        np.testing.assert_array_equal(sum(prop.counts_by_cohort.values()), prop.counts)
        assert set(prop.counts_by_cohort) == set(range(2012, 2019))

    def test_counts_are_additive_over_subsets(self, panels, synth):
        prop, _ = panels
        half = synth["loan_id"] % 2 == 0
        total = prop.transition_counts_for(synth[half]) + prop.transition_counts_for(synth[~half])
        np.testing.assert_array_equal(total, prop.counts)

    def test_scalar_paths_match_vectorised(self, panels, synth):
        prop, _ = panels
        sample = synth.sample(1_500, random_state=3)
        by_id = prop.paths.set_index("loan_id")["segments"]
        for row in sample.to_dict("records"):
            assert row_segments(row, 4, TransitionStructure.PROPOSAL, SNAP) == by_id[row["loan_id"]]

    @pytest.mark.parametrize("month", [1, 5, 13, 37, 70])
    def test_interim_states_vectorised_equals_scalar(self, panels, synth, month):
        prop, _ = panels
        sample = synth.sample(800, random_state=month).reset_index(drop=True)
        states, paid = prop.interim_states(sample, month)
        for i, row in enumerate(sample.to_dict("records")):
            assert (int(states[i]), int(paid[i])) == tuple(map(int, prop.interim_state(row, month)))

    def test_interim_states_per_row_months(self, panels, synth):
        prop, _ = panels
        sample = synth.head(200)
        months = np.arange(1, 201)
        states, paid = prop.interim_states(sample, months)
        for i, row in enumerate(sample.to_dict("records")):
            assert (states[i], paid[i]) == tuple(map(int, prop.interim_state(row, int(months[i]))))

    def test_empirical_matrix_and_serialisation(self, panels):
        prop, ext = panels
        for panel in (prop, ext):
            p = panel.empirical_matrix()
            np.testing.assert_allclose(p.sum(axis=1), 1.0)
            np.testing.assert_array_equal(p[R], [0, 0, 1, 0])
            np.testing.assert_array_equal(p[L], [0, 0, 0, 1])
            d = panel.to_dict()
            json.dumps(d)
            assert d["structure"] == panel.structure.value and d["n_loans"] == panel.n_loans

    def test_validate_catches_corruption(self, panels):
        prop, _ = panels
        bad = LoanStatePanel(
            counts=prop.counts.copy(), n_loans=prop.n_loans, n_loan_months=prop.n_loan_months,
            n_censored=prop.n_censored, counts_by_cohort={}, structure=prop.structure,
        )
        bad.counts[R, C] = 1
        with pytest.raises(DataSourceError):
            bad.validate()
        bad.counts[R, C] = 0
        bad.n_loan_months += 1
        with pytest.raises(DataSourceError):
            bad.validate()
        bad.n_loan_months -= 1
        bad.counts[C, R] = 5
        with pytest.raises(DataSourceError):
            bad.validate()
