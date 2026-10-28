"""Prepayment: InterestRealization / RealizationLookup and the reward's ``realization``
factor (docs/MATH.md §2.8)."""

import json

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import ArmSpaceConfig, RewardConfig
from bayesian_credit.data.arms import ArmSpace
from bayesian_credit.data.segmentation import Segmenter
from bayesian_credit.exceptions import ConfigurationError
from bayesian_credit.rewards.amortization import total_interest
from bayesian_credit.rewards.realization import InterestRealization, RealizationLookup, group_grade

TRUE = {("A", 36): 0.85, ("A", 60): 0.70, ("C", 36): 0.80, ("C", 60): 0.65, ("E", 36): 0.75}


def make_paid(n_per_cell=400, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    rows = []
    for (g, t), rho in {**TRUE, **(extra or {})}.items():
        L = rng.uniform(5_000, 30_000, n_per_cell).round(-2)
        r = rng.uniform(0.06, 0.25, n_per_cell)
        sched = total_interest(L, r, np.full(n_per_cell, t))
        noise = rng.normal(0.0, 0.05, n_per_cell)
        rows.append(pd.DataFrame({
            "grade": g, "term": t, "funded_amnt": L, "loan_amnt": L, "int_rate": r,
            "total_rec_int": sched * np.clip(rho + noise - noise.mean(), 0.0, 1.0),
            "y_resolved": True, "y_default": 0.0,
            "issue_d": pd.Timestamp("2012-01-01"),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["grade"] = df["grade"].astype("string")
    return df


def test_fit_recovers_the_cell_means_and_falls_back():
    df = make_paid(extra={("G", 60): 0.1})  # G/60: a pathological cell clipped to 0.3
    tab = InterestRealization.fit(df)
    for (g, t), rho in TRUE.items():
        assert tab.factor(g, t) == pytest.approx(rho, abs=0.01)
        assert tab.source(g, t) == "grade_term"
    assert tab.factor("G", 60) == 0.3  # every entry is clipped to [0.3, 1.0]
    assert tab.factor("E", 60) == tab.by_term[60] and tab.source("E", 60) == "term"  # no E/60 cell
    assert tab.factor(None, 36) == tab.by_term[36]
    assert tab.factor("A", 84) == tab.global_rho and tab.source("A", 84) == "global"  # unknown term
    assert tab.factor("A", None) == tab.global_rho
    assert tab.counts["A|36"] == 400 and tab.counts["term:60"] == 1200 and tab.n_fit == len(df)
    assert InterestRealization().factor("A", 36) == 1.0 and InterestRealization().is_empty
    frame = tab.table()
    assert set(frame["level"]) == {"grade_term", "term", "global"} and len(frame) == len(TRUE) + 1 + 2 + 1


def test_per_loan_ratios_are_capped_and_defaults_unresolved_rows_are_ignored():
    df = make_paid(n_per_cell=100)
    over = df.assign(total_rec_int=df["total_rec_int"] * 3.0)  # late interest / data errors: ratio > 1
    assert InterestRealization.fit(over).factor("A", 36) == pytest.approx(1.0)
    mixed = pd.concat([df, df.assign(y_default=1.0, total_rec_int=0.0), df.assign(y_resolved=False, total_rec_int=0.0)])
    assert InterestRealization.fit(mixed).to_dict() == InterestRealization.fit(df).to_dict()
    small = InterestRealization.fit(df, min_count=1_000)  # every cell too small -> global only
    assert not small.by_grade_term and not small.by_term and small.factor("A", 36) == small.global_rho


def test_elapsed_term_loans_are_preferred():
    old = make_paid(n_per_cell=300)  # issued 2012-01: elapsed by 2019-03 for both terms
    recent = make_paid(n_per_cell=300, seed=1).assign(issue_d=pd.Timestamp("2017-06-01"), total_rec_int=lambda d: d["total_rec_int"] * 0.4)
    both = pd.concat([old, recent], ignore_index=True)
    tab = InterestRealization.fit(both, snapshot_date="2019-03-31")
    assert tab.sample == "fully_paid_elapsed" and tab.n_fit == len(old)
    assert tab.factor("A", 36) == pytest.approx(InterestRealization.fit(old).factor("A", 36))
    biased = InterestRealization.fit(both)  # no snapshot: early payers of recent vintages pull ρ down
    assert biased.sample == "fully_paid" and biased.factor("A", 36) < tab.factor("A", 36) - 0.1


def test_round_trip_and_summary():
    tab = InterestRealization.fit(make_paid(n_per_cell=60), snapshot_date="2019-03-31")
    back = InterestRealization.from_dict(json.loads(json.dumps(tab.to_dict(), allow_nan=False)))
    assert back == tab and back.factor("C", 60) == tab.factor("C", 60)
    s = tab.summary()
    assert set(s) >= {"global", "by_term", "by_grade_term", "n_fit", "sample"} and s["by_term"]["36"] == tab.by_term[36]


def test_group_grade_follows_the_segmentation():
    assert group_grade("B|credit_card|RENT") == "B"
    assert group_grade("B|credit_card|RENT", Segmenter(("grade", "purpose_group", "home_ownership"))) == "B"
    assert group_grade("RENT|C", Segmenter(("home_ownership", "grade"))) == "C"
    assert group_grade("C4|OWN", Segmenter(("sub_grade", "home_ownership"))) == "C"
    assert group_grade("OWN|700-739", Segmenter(("home_ownership", "fico_band"))) is None
    assert group_grade("debt|RENT") is None and group_grade(None) is None


def test_lookup_maps_group_and_arm_to_the_grade_term_factor(arm_space):
    tab = InterestRealization.fit(make_paid(n_per_cell=60))
    seg = Segmenter(("grade", "purpose_group", "home_ownership"))
    look = RealizationLookup.from_config(RewardConfig(), tab, arm_space, seg)
    a36, a60 = arm_space.nearest(10_000, 0.13, 36), arm_space.nearest(10_000, 0.13, 60)
    assert look.enabled
    assert look("A|other|OWN", a36) == tab.factor("A", 36) and look("A|other|OWN", a60) == tab.factor("A", 60)
    assert look("A|other|OWN", 0) == 1.0 and look("A|other|OWN", 10_000) == 1.0  # decline / outside the space
    assert look("Z|other|OWN", a60) == tab.by_term[60]
    assert look.for_term("C|other|OWN", 60) == tab.factor("C", 60)
    finer = look.with_segmenter(seg.refine())
    assert finer("C|other|OWN|700-739", a36) == tab.factor("C", 36)
    off = RealizationLookup.from_config(RewardConfig(prepayment_adjustment=None), tab, arm_space, seg)
    assert not off.enabled and off("A|other|OWN", a60) == 1.0 and off.for_term("A|x|y", 60) == 1.0
    assert not RealizationLookup(None, arm_space).enabled
    with pytest.raises(ConfigurationError):
        RewardConfig(prepayment_adjustment="magic").validate()
    assert ArmSpace.from_config(ArmSpaceConfig()).n_arms == arm_space.n_arms
