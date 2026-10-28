"""Amortisation helpers (SPEC §2.7)."""

import numpy as np
import pytest

from bayesian_credit.data.arms import LendingPolicy, annuity_installment
from bayesian_credit.rewards.amortization import (
    cumulative_opening_balance,
    installment,
    outstanding_balances,
    schedule,
    schedule_arrays,
    total_interest,
    weighted_average_life_years,
)

LOANS = [(10_000.0, 0.13, 36), (35_000.0, 0.28, 60), (5_000.0, 0.07, 36), (1_000.0, 0.0, 12)]


def test_installment_matches_annuity_formula_and_arm_space():
    L, r, T = 10_000.0, 0.13, 36
    i = r / 12
    expected = L * i * (1 + i) ** T / ((1 + i) ** T - 1)
    assert installment(L, r, T) == pytest.approx(expected, rel=1e-12)
    assert installment(L, r, T) == annuity_installment(L, r, T)
    assert LendingPolicy(L, r, T).installment == pytest.approx(installment(L, r, T), rel=1e-15)
    assert installment(1200.0, 0.0, 12) == pytest.approx(100.0)


@pytest.mark.parametrize(("L", "r", "T"), LOANS)
def test_schedule_identities(L, r, T):
    s = schedule(L, r, T)
    assert list(s.columns) == ["month", "payment", "interest", "principal", "opening_balance", "balance"]
    assert len(s) == T
    assert s["month"].tolist() == list(range(1, T + 1))
    assert s["principal"].sum() == pytest.approx(L, rel=1e-12)
    assert s["balance"].iloc[-1] == 0.0
    assert s["opening_balance"].iloc[0] == L
    np.testing.assert_allclose(s["opening_balance"].to_numpy()[1:], s["balance"].to_numpy()[:-1])
    np.testing.assert_allclose(s["interest"], r / 12 * s["opening_balance"], rtol=1e-12)
    np.testing.assert_allclose(s["payment"], installment(L, r, T), rtol=1e-9)
    assert np.all(np.diff(s["opening_balance"].to_numpy()) < 0)
    assert total_interest(L, r, T) == pytest.approx(s["interest"].sum(), rel=1e-12)
    assert total_interest(L, r, T) == pytest.approx(T * installment(L, r, T) - L, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("L", "r", "T"), LOANS)
def test_weighted_average_life_identities(L, r, T):
    s = schedule(L, r, T)
    wal = weighted_average_life_years(L, r, T)
    assert wal == pytest.approx((s["month"] * s["principal"]).sum() / L / 12, rel=1e-12)
    assert wal == pytest.approx(s["opening_balance"].sum() / (12 * L), rel=1e-12)
    if r > 0:
        assert wal == pytest.approx(total_interest(L, r, T) / (r * L), rel=1e-9)
    else:
        assert wal == pytest.approx((T + 1) / 24, rel=1e-12)
    assert 0 < wal < T / 12


def test_outstanding_balances_are_read_only_and_amortise_to_zero():
    B = outstanding_balances(10_000.0, 0.13, 36)
    assert B.shape == (37,)
    assert B[0] == 10_000.0 and B[-1] == 0.0
    assert np.all(np.diff(B) < 0)
    with pytest.raises(ValueError):
        B[0] = 1.0
    s = schedule_arrays(10_000.0, 0.13, 36)
    np.testing.assert_array_equal(s.balances[:-1], s.opening_balance)
    assert s.cum_interest[-1] == pytest.approx(s.total_interest)
    assert s.cum_opening_balance[12] == pytest.approx(s.opening_balance[:12].sum())


@pytest.mark.parametrize(("L", "r", "T"), LOANS)
def test_cumulative_opening_balance_matches_schedule(L, r, T):
    s = schedule_arrays(L, r, T)
    for k in (0, 1, 7, T - 1, T, T + 10):
        expected = s.opening_balance[: min(k, T)].sum()
        assert cumulative_opening_balance(L, r, T, k) == pytest.approx(expected, rel=1e-12, abs=1e-9)
        vec = cumulative_opening_balance(np.array([L]), np.array([r]), np.array([T]), np.array([k]))
        assert vec[0] == pytest.approx(expected, rel=1e-9, abs=1e-7)
    assert cumulative_opening_balance(L, r, T, -3) == 0.0
    assert cumulative_opening_balance(L, r, T, T) == pytest.approx(12 * L * weighted_average_life_years(L, r, T))


def test_vectorised_inputs_match_scalar_results():
    L = np.array([amount for amount, _, _ in LOANS])
    r = np.array([x for _, x, _ in LOANS])
    T = np.array([t for _, _, t in LOANS])
    np.testing.assert_allclose(installment(L, r, T), [installment(*x) for x in LOANS], rtol=1e-12)
    np.testing.assert_allclose(total_interest(L, r, T), [total_interest(*x) for x in LOANS], rtol=1e-9)
    np.testing.assert_allclose(
        weighted_average_life_years(L, r, T), [weighted_average_life_years(*x) for x in LOANS], rtol=1e-9
    )
    assert installment(10_000.0, np.array([0.1, 0.2]), 36).shape == (2,)
    np.testing.assert_array_equal(installment(np.array([0.0, -5.0]), 0.1, 36), [0.0, 0.0])


def test_null_policy_is_all_zero():
    assert installment(0, 0, 0) == 0.0
    assert total_interest(0, 0, 0) == 0.0
    assert weighted_average_life_years(0, 0, 0) == 0.0
    assert schedule(0, 0, 0).empty
    np.testing.assert_array_equal(outstanding_balances(0, 0, 0), [0.0])
    assert cumulative_opening_balance(0, 0, 0, 10) == 0.0


def test_scalar_calls_are_cached():
    a = schedule_arrays(12_345.0, 0.11, 36)
    b = schedule_arrays(12_345, 0.11, 36.0)
    assert a is b


@pytest.mark.parametrize(("L", "r", "T"), [(1000, -0.1, 36), (1000, 0.1, 36.5), (np.nan, 0.1, 36)])
def test_invalid_inputs_raise(L, r, T):
    with pytest.raises(ValueError):
        schedule(L, r, T)
