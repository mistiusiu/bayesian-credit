"""Compound risk-adjusted reward (SPEC §2.7, proposal §3.3)."""

import math

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import RewardConfig
from bayesian_credit.data.arms import NULL_POLICY, LendingPolicy
from bayesian_credit.models.beta import BetaPosterior
from bayesian_credit.models.mctm import MarkovChainTransitionMatrix
from bayesian_credit.models.two_stage import RiskEstimate
from bayesian_credit.rewards.amortization import schedule_arrays, weighted_average_life_years
from bayesian_credit.rewards.compound import (
    COMPONENT_KEYS,
    CompoundRewardFunction,
    RewardDistribution,
    reward_moments,
)
from bayesian_credit.types import State

POLICY = LendingPolicy(10_000.0, 0.16, 36)
COUNTS = np.array(
    [[9700.0, 300.0, 0.0, 0.0], [120.0, 150.0, 20.0, 110.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
)


def make_risk(pa, pb, ra, rb):
    return RiskEstimate("g", 1, BetaPosterior(pa, pb), BetaPosterior(ra, rb))


RISK = make_risk(6.0, 34.0, 3.0, 5.0)  # E[p] = 0.15, E[R] = 0.375


@pytest.fixture
def mctm():
    return MarkovChainTransitionMatrix().fit_counts(COUNTS)


def test_zero_default_risk_gives_gross_interest_minus_funding():
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04))
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    gross, funding = s.interest.sum(), 0.04 * s.opening_balance.sum() / 12
    np.testing.assert_allclose(f.sample(POLICY, np.zeros(5), np.full(5, 0.3)), gross - funding, rtol=1e-12)
    d = f.expected(POLICY, make_risk(1e-12, 1e6, 2.0, 5.0))
    assert d.mean == pytest.approx(gross - funding, rel=1e-9)
    assert d.std == pytest.approx(0.0, abs=1e-6)
    assert d.components["expected_loss"] == pytest.approx(0.0, abs=1e-9)
    assert d.components["expected_interest"] == pytest.approx(gross, rel=1e-12)


def test_certain_default_without_recovery_under_uniform_timing():
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.0))
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    F = np.arange(1, 37) / 36
    received = float(s.interest @ (1 - F))
    reward = f.sample(POLICY, np.array([1.0]), np.array([0.0]))[0]
    assert reward == pytest.approx(received - POLICY.amount, rel=1e-12)  # loss = L
    assert 0.45 < received / s.total_interest < 0.70  # ≈ half of the gross interest
    d = f.expected(POLICY, make_risk(1e9, 1e-3, 1e-3, 1e9))  # p ≈ 1, R ≈ 0
    assert d.components["expected_loss"] == pytest.approx(POLICY.amount, rel=1e-6)
    assert d.components["expected_interest"] == pytest.approx(received, rel=1e-6)
    assert d.mean == pytest.approx(received - POLICY.amount, rel=1e-6)


@pytest.mark.parametrize("basis", ["full", "outstanding"])
@pytest.mark.parametrize("use_mctm", [False, True])
def test_closed_form_mean_and_variance_match_monte_carlo(basis, use_mctm, mctm):
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04, exposure_basis=basis))
    chain = mctm if use_mctm else None
    rng = np.random.default_rng(2024)
    p = RISK.p_default.sample(rng, 200_000)
    R = RISK.recovery.sample(rng, 200_000)
    mc = f.sample(POLICY, p, R, chain)
    d = f.expected(POLICY, RISK, chain)
    assert d.mean > 100  # keep the relative tolerance meaningful
    assert mc.mean() == pytest.approx(d.mean, rel=0.01)
    assert mc.var() == pytest.approx(d.var, rel=0.01)


def test_components_add_up_and_reward_moments_formula():
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.05))
    d = f.expected(POLICY, RISK)
    c = d.components
    assert set(COMPONENT_KEYS) <= set(c)
    assert d.mean == pytest.approx(c["expected_interest"] - c["expected_loss"] - c["funding_cost"], rel=1e-12)
    t = f.terms(POLICY)
    ep, er = RISK.p_default.mean, RISK.recovery.mean
    k = t.interest_at_risk + t.exposure
    assert d.mean == pytest.approx(t.fixed - k * ep + t.exposure * ep * er, rel=1e-12)
    e_h = k - t.exposure * er
    e_h2 = k * k - 2 * k * t.exposure * er + t.exposure**2 * RISK.recovery.second_moment
    assert d.var == pytest.approx(RISK.p_default.second_moment * e_h2 - (ep * e_h) ** 2, rel=1e-9)
    mean, var = reward_moments(
        np.array([t.fixed]), t.interest_at_risk, t.exposure, ep, RISK.p_default.second_moment,
        er, RISK.recovery.second_moment,
    )
    assert mean[0] == pytest.approx(d.mean) and var[0] == pytest.approx(d.var)


@pytest.mark.parametrize("policy", [POLICY, LendingPolicy(35_000.0, 0.28, 60), LendingPolicy(5_000.0, 0.07, 36)])
def test_funding_cost_equals_rate_times_amount_times_wal(policy):
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.05))
    wal = weighted_average_life_years(policy.amount, policy.rate, policy.term)
    t = f.terms(policy)
    assert t.funding_cost == pytest.approx(0.05 * policy.amount * wal, rel=1e-12)
    assert f.expected(policy, RISK).components["funding_cost"] == t.funding_cost


def test_terminal_reward_of_fully_paid_loan():
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04))
    L, r, T = 12_000.0, 0.1299, 36
    s = schedule_arrays(L, r, T)
    late_fee = 15.0
    row = pd.Series(
        {
            "funded_amnt": L, "loan_amnt": L, "int_rate": r, "term": T,
            "y_realized_revenue": s.total_interest + late_fee, "y_realized_loss": 0.0,
            "y_months_on_book": T,
        }
    )
    expected = s.total_interest + late_fee - 0.04 * L * weighted_average_life_years(L, r, T)
    assert f.terminal(row) == pytest.approx(expected, rel=1e-12)
    assert f.terminal(row.to_dict()) == pytest.approx(expected, rel=1e-12)
    capped = row.copy()
    capped["y_months_on_book"] = 45  # months on book beyond the term are capped
    assert f.terminal(capped) == pytest.approx(expected, rel=1e-12)


def test_terminal_reward_of_charged_off_loan_charges_funding_while_on_book():
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04))
    L, r, T, k = 10_000.0, 0.2, 60, 10
    s = schedule_arrays(L, r, T)
    rev = s.cum_interest[k]
    loss = s.balances[k] * 0.8
    row = {"funded_amnt": L, "int_rate": r, "term": T, "y_realized_revenue": rev,
           "y_realized_loss": loss, "y_months_on_book": k}
    assert f.terminal(row) == pytest.approx(rev - loss - 0.04 * s.opening_balance[:k].sum() / 12, rel=1e-12)
    assert f.terminal(row) < 0


def test_terminal_many_matches_terminal_and_handles_missing():
    f = CompoundRewardFunction()
    rows = [
        {"funded_amnt": 12_000.0, "int_rate": 0.1299, "term": 36, "y_realized_revenue": 2600.0,
         "y_realized_loss": 0.0, "y_months_on_book": 36},
        {"funded_amnt": np.nan, "loan_amnt": 8_000.0, "int_rate": 0.25, "term": 60,
         "y_realized_revenue": 900.0, "y_realized_loss": 5000.0, "y_months_on_book": 7},
        {"funded_amnt": 5_000.0, "int_rate": 0.1, "term": 36, "y_realized_revenue": np.nan,
         "y_realized_loss": 0.0, "y_months_on_book": 36},
    ]
    table = pd.DataFrame(rows)
    out = f.terminal_many(table)
    singles = [f.terminal(row) for _, row in table.iterrows()]
    np.testing.assert_allclose(out[:2], singles[:2], rtol=1e-9)
    assert np.isnan(out[2]) and math.isnan(singles[2])
    assert math.isnan(f.terminal({"int_rate": 0.1}))


def test_null_policy_has_zero_reward():
    f = CompoundRewardFunction()
    d = f.expected(NULL_POLICY, RISK)
    assert d.mean == 0.0 and d.std == 0.0
    assert all(v == 0.0 for v in d.components.values())
    assert np.all(f.sample(NULL_POLICY, np.full(4, 0.3), np.full(4, 0.5)) == 0.0)
    assert f.interim(NULL_POLICY, State.DELINQUENT, 3, 1, RISK).mean == 0.0
    assert f.score(d) == 0.0


def test_outstanding_exposure_basis_reduces_expected_loss(mctm):
    full = CompoundRewardFunction(RewardConfig(exposure_basis="full"))
    outstanding = CompoundRewardFunction(RewardConfig(exposure_basis="outstanding"))
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    t = outstanding.terms(POLICY)
    assert t.exposure_factor == pytest.approx(s.opening_balance.mean() / POLICY.amount)  # uniform pmf
    assert 0.0 < t.exposure_factor < 1.0
    assert full.terms(POLICY).exposure_factor == 1.0
    assert (
        outstanding.expected(POLICY, RISK).components["expected_loss"]
        < full.expected(POLICY, RISK).components["expected_loss"]
    )
    f_m = mctm.default_timing_pmf(36)
    assert outstanding.terms(POLICY, mctm).exposure_factor == pytest.approx(
        float(f_m @ s.opening_balance) / POLICY.amount
    )


def test_mctm_default_timing_drives_interest_at_risk(mctm):
    f = CompoundRewardFunction()
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    t = f.terms(POLICY, mctm)
    assert t.interest_at_risk == pytest.approx(float(s.interest @ mctm.default_timing_cdf(36)))
    assert f.terms(POLICY).interest_at_risk == pytest.approx(float(s.interest @ (np.arange(1, 37) / 36)))
    np.testing.assert_allclose(f.timing_cdf(36), np.arange(1, 37) / 36)
    other = MarkovChainTransitionMatrix().fit_counts(COUNTS * np.array([[1, 3, 0, 0]] * 2 + [[0] * 4] * 2))
    assert f.terms(POLICY, other).interest_at_risk != t.interest_at_risk  # cache keyed on the matrix


def test_risk_adjusted_score():
    d = RewardDistribution(100.0, 20.0, {})
    assert d.score(0.0) == 100.0 and d.score(1.5) == 70.0
    assert CompoundRewardFunction(RewardConfig(risk_aversion=0.5)).score(d) == 90.0
    assert float(d) == 100.0 and d.var == 400.0
    assert d.to_dict() == {"mean": 100.0, "std": 20.0, "components": {}}


def test_interim_at_origination_equals_expected_without_mctm():
    f = CompoundRewardFunction()
    a = f.interim(POLICY, State.CURRENT, 0, 0, RISK)
    b = f.expected(POLICY, RISK)
    assert a.mean == pytest.approx(b.mean, rel=1e-12)
    assert a.std == pytest.approx(b.std, rel=1e-9)
    assert a.components["realized_interest"] == 0.0


def test_interim_absorbing_states():
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04))
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    funding = f.terms(POLICY).funding_cost
    rec = f.interim(POLICY, State.RECOVERED, 20, 20, RISK)
    assert rec.mean == pytest.approx(s.cum_interest[20] - funding)
    assert rec.std == 0.0
    loss = f.interim(POLICY, State.LOSS, 14, 10, RISK)
    assert loss.mean == pytest.approx(s.cum_interest[10] - s.balances[10] * (1 - RISK.recovery.mean) - funding)
    assert loss.std == pytest.approx(s.balances[10] * RISK.recovery.std)


def test_interim_with_mctm_uses_state_loss_probability_and_remaining_timing(mctm):
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04))
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    res = f.interim(POLICY, State.DELINQUENT, 12, 11, RISK, mctm)
    q = mctm.p_loss_given_state(State.DELINQUENT, 24)
    F_rem = mctm.default_timing_cdf(24, State.DELINQUENT)
    future = s.interest[12:].sum() - q * float(s.interest[12:] @ F_rem)
    loss = s.balances[11] * q * (1 - RISK.recovery.mean)
    expected = s.cum_interest[11] + future - loss - f.terms(POLICY, mctm).funding_cost
    assert res.mean == pytest.approx(expected, rel=1e-12)
    assert res.components["p_loss"] == q
    assert res.std > 0


@pytest.mark.parametrize("use_mctm", [False, True])
def test_interim_delinquent_is_worse_than_current(use_mctm, mctm):
    f = CompoundRewardFunction()
    chain = mctm if use_mctm else None
    cur = f.interim(POLICY, State.CURRENT, 12, 12, RISK, chain)
    delq = f.interim(POLICY, State.DELINQUENT, 12, 11, RISK, chain)
    assert delq.mean < cur.mean
    assert 0.0 <= delq.components["p_loss"] <= 1.0
    if not use_mctm:
        assert delq.components["p_loss"] == 0.5  # max(E[p], 0.5)
        assert cur.components["p_loss"] == pytest.approx(RISK.p_default.mean)


def test_interim_at_maturity_in_current_state_is_fully_realised(mctm):
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.04))
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    d = f.interim(POLICY, State.CURRENT, 36, 36, RISK, mctm)
    assert d.mean == pytest.approx(s.total_interest - f.terms(POLICY).funding_cost, rel=1e-12)
    assert d.std == pytest.approx(0.0, abs=1e-9)
    clipped = f.interim(POLICY, State.CURRENT, 99, 99, RISK, mctm)
    assert clipped.mean == pytest.approx(d.mean)


# ----- prepayment: the realization factor (docs/MATH.md §2.8) ---------------------------------------
@pytest.mark.parametrize("with_mctm", [False, True])
def test_realization_one_is_bit_identical_to_the_default(mctm, with_mctm):
    f = CompoundRewardFunction()
    chain = mctm if with_mctm else None
    a, b = f.expected(POLICY, RISK, chain), f.expected(POLICY, RISK, chain, realization=1.0)
    assert (a.mean, a.std) == (b.mean, b.std)
    p, R = np.linspace(0.01, 0.9, 7), np.linspace(0.0, 0.8, 7)
    np.testing.assert_array_equal(f.sample(POLICY, p, R, chain), f.sample(POLICY, p, R, chain, realization=1.0))
    for state, month, paid in ((State.CURRENT, 5, 5), (State.DELINQUENT, 9, 7), (State.LOSS, 12, 10), (State.RECOVERED, 20, 20)):
        x, y = f.interim(POLICY, state, month, paid, RISK, chain), f.interim(POLICY, state, month, paid, RISK, chain, realization=1.0)
        assert (x.mean, x.std) == (y.mean, y.std)


def test_realization_scales_expected_interest_and_funding(mctm):
    f = CompoundRewardFunction(RewardConfig(funding_rate=0.05))
    rho = 0.78
    t = f.terms(POLICY, mctm)
    d = f.expected(POLICY, RISK, mctm, realization=rho)
    c = d.components
    ep, er = RISK.p_default.mean, RISK.recovery.mean
    assert c["realization"] == rho and c["gross_interest"] == t.gross_interest  # the contract is unchanged
    assert c["expected_interest"] == pytest.approx(rho * (t.gross_interest - t.interest_at_risk * ep), rel=1e-12)
    assert c["funding_cost"] == pytest.approx(rho * t.funding_cost, rel=1e-12)
    assert c["expected_loss"] == pytest.approx(t.exposure * ep * (1 - er), rel=1e-12)  # the loss is not
    assert d.mean == pytest.approx(c["expected_interest"] - c["expected_loss"] - c["funding_cost"], rel=1e-12)
    assert d.mean == pytest.approx(rho * t.fixed - (rho * t.interest_at_risk + t.exposure) * ep + t.exposure * ep * er, rel=1e-12)
    assert d.mean < f.expected(POLICY, RISK, mctm).mean
    rng = np.random.default_rng(3)
    p = RISK.p_default.sample(rng, 300_000)
    R = RISK.recovery.sample(rng, 300_000)
    mc = f.sample(POLICY, p, R, mctm, realization=rho)
    assert mc.mean() == pytest.approx(d.mean, rel=0.01) and mc.var() == pytest.approx(d.var, rel=0.01)
    with pytest.raises(ValueError):
        f.expected(POLICY, RISK, mctm, realization=0.0)
    with pytest.raises(ValueError):
        f.sample(POLICY, p[:3], R[:3], realization=float("nan"))
    assert f.expected(NULL_POLICY, RISK, realization=0.5).mean == 0.0


def test_interim_with_realization_keeps_the_origination_identity_and_scales_only_the_future():
    f = CompoundRewardFunction()
    rho = 0.7
    a = f.interim(POLICY, State.CURRENT, 0, 0, RISK, realization=rho)
    b = f.expected(POLICY, RISK, realization=rho)
    assert a.mean == pytest.approx(b.mean, rel=1e-12) and a.std == pytest.approx(b.std, rel=1e-9)
    s = schedule_arrays(POLICY.amount, POLICY.rate, POLICY.term)
    funding = f.terms(POLICY).funding_cost
    rec = f.interim(POLICY, State.RECOVERED, 20, 20, RISK, realization=rho)
    assert rec.mean == pytest.approx(s.cum_interest[20] - rho * funding)  # interest received is not scaled
    cur1 = f.interim(POLICY, State.CURRENT, 10, 10, RISK)
    cur = f.interim(POLICY, State.CURRENT, 10, 10, RISK, realization=rho)
    future1 = cur1.components["expected_interest"] - s.cum_interest[10]
    assert cur.components["expected_interest"] - s.cum_interest[10] == pytest.approx(rho * future1, rel=1e-12)
    assert cur.mean == pytest.approx(cur1.mean - (1 - rho) * future1 + (1 - rho) * funding, rel=1e-12)
    assert cur.components["realization"] == rho
