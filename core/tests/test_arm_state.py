"""GaussianArmPosterior / GaussianArmSet: the proposal's incremental mean + online variance."""

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bayesian_credit.bandits.arm_state import (
    GaussianArmPosterior,
    GaussianArmSet,
    posterior_variance,
    ucb_index,
    welford_update,
)
from bayesian_credit.exceptions import (
    ConfigurationError,
    InvalidPosteriorError,
    NumericalInstabilityError,
)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("k", [2, 3, 10, 257, 2000])
def test_unit_weight_updates_equal_sample_mean_and_ddof1_variance(k):
    rng = np.random.default_rng(100 + k)
    rs = rng.normal(250.0, 80.0, size=k)
    post = GaussianArmPosterior(mu=-3.0, sigma2=1e4)  # the prior carries no pseudo-count
    for r in rs:
        post.update(r)
    assert post.n == k
    assert close(post.mu, float(np.mean(rs)))
    assert close(post.sigma2, float(np.var(rs, ddof=1)))
    assert close(post.m2, float(np.sum((rs - rs.mean()) ** 2)))


@settings(max_examples=150, deadline=None)
@given(st.lists(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False), min_size=2, max_size=60))
def test_welford_equivalence_property(rs):
    post = GaussianArmPosterior(mu=0.0, sigma2=1.0)
    for r in rs:
        post.update(r)
    arr = np.asarray(rs, dtype=float)
    assert math.isclose(post.mu, float(arr.mean()), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(post.sigma2, float(arr.var(ddof=1)), rel_tol=1e-7, abs_tol=1e-6)


def test_proposal_recursion_is_the_same_update():
    """σ² ← ((n−1)σ²_old + (r−μ_old)(r−μ_new))/n with n the pre-update count (§3.3)."""
    rng = np.random.default_rng(7)
    rs = rng.normal(0.0, 3.0, size=200)
    post = GaussianArmPosterior(mu=0.0, sigma2=99.0)
    mu, var, n = 0.0, 99.0, 0
    for r in rs:
        post.update(r)
        mu_old = mu
        mu = mu + (r - mu) / (n + 1)
        if n >= 1:
            var = ((n - 1) * var + (r - mu_old) * (r - mu)) / n
        n += 1
        assert close(post.mu, mu)
        if n >= 2:
            assert close(post.sigma2, var)


def test_variance_is_prior_until_two_observations():
    post = GaussianArmPosterior(mu=5.0, sigma2=400.0)
    assert post.sigma2 == 400.0 and post.prior_mu == 5.0 and post.prior_sigma2 == 400.0
    post.update(12.0)
    assert post.mu == 12.0  # first observation replaces the prior mean
    assert post.n == 1 and post.sigma2 == 400.0  # variance undefined at n = 1
    post.update(14.0)
    assert post.sigma2 == pytest.approx(2.0)


def test_weighted_updates_match_weighted_moments():
    rng = np.random.default_rng(3)
    rs = rng.normal(10.0, 4.0, size=40)
    ws = rng.uniform(0.1, 1.0, size=40)
    post = GaussianArmPosterior(mu=0.0, sigma2=1.0)
    for r, w in zip(rs, ws, strict=True):
        post.update(r, weight=w)
    mean = float(np.average(rs, weights=ws))
    m2 = float(np.sum(ws * (rs - mean) ** 2))
    assert close(post.n, float(ws.sum()))
    assert close(post.mu, mean)
    assert close(post.m2, m2)
    assert close(post.sigma2, m2 / (ws.sum() - 1.0))


def test_zero_weight_is_a_noop_and_fractional_n_keeps_prior_variance():
    post = GaussianArmPosterior(mu=1.0, sigma2=9.0)
    post.update(100.0, weight=0.0)
    assert post.to_dict() == GaussianArmPosterior(mu=1.0, sigma2=9.0).to_dict()
    post.update(4.0, weight=0.5)
    post.update(6.0, weight=0.5)
    assert post.n == 1.0 and post.sigma2 == 9.0  # n <= 1: still undefined


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_reward_raises(bad):
    post = GaussianArmPosterior()
    with pytest.raises(NumericalInstabilityError):
        post.update(bad)
    with pytest.raises(NumericalInstabilityError):
        post.update(1.0, weight=-0.5)
    with pytest.raises(NumericalInstabilityError):
        post.update(1.0, weight=float("nan"))


def test_overflow_raises_and_leaves_state_untouched():
    post = GaussianArmPosterior()
    post.update(1e200)
    before = post.to_dict()
    with pytest.raises(NumericalInstabilityError):
        post.update(-1e200)  # M2 would be 2e400
    assert post.to_dict() == before


def test_sampling_scales_and_floor():
    post = GaussianArmPosterior(mu=0.0, sigma2=1.0)
    for r in (10.0, 12.0, 14.0, 16.0):
        post.update(r)
    sigma = math.sqrt(post.sigma2)
    assert post.std("proposal") == pytest.approx(sigma)
    assert post.std("standard_error") == pytest.approx(sigma / 2.0)
    draws = np.array([post.sample(np.random.default_rng(i)) for i in range(4000)])
    assert draws.mean() == pytest.approx(13.0, abs=0.2)
    assert draws.std() == pytest.approx(sigma, rel=0.1)
    flat = GaussianArmPosterior(mu=2.0, sigma2=0.0)
    assert flat.std(min_sigma=1e-3) == 1e-3
    with pytest.raises(ConfigurationError):
        post.std("bogus")


def test_ucb_helper():
    post = GaussianArmPosterior(mu=3.0, sigma2=4.0)
    assert post.ucb(10) == math.inf  # untried
    post.update(3.0)
    post.update(5.0)
    expected = post.mu + 2.0 * math.sqrt(2 * math.log(50) / 2.0)
    assert post.ucb(50, c=2.0) == pytest.approx(expected)
    scaled = post.mu + 1.5 * math.sqrt(post.sigma2) * math.sqrt(2 * math.log(50) / 2.0)
    assert post.ucb(50, c=1.5, scale=None) == pytest.approx(scaled)
    idx = ucb_index(np.array([1.0, 2.0]), np.array([0.0, 4.0]), 9, 1.0)
    assert idx[0] == math.inf and idx[1] == pytest.approx(2.0 + math.sqrt(2 * math.log(9) / 4))


def test_posterior_serialisation_roundtrip():
    post = GaussianArmPosterior(mu=1.0, sigma2=2.0)
    for r in (3.0, -1.0, 7.5):
        post.update(r, weight=0.7)
    again = GaussianArmPosterior.from_dict(json.loads(json.dumps(post.to_dict(), allow_nan=False)))
    assert again == post


def test_arm_set_is_bit_identical_to_scalar_posterior():
    rng = np.random.default_rng(11)
    prior_mu = rng.normal(0, 5, size=6)
    prior_sigma2 = rng.uniform(1, 9, size=6)
    arms = GaussianArmSet(prior_mu, prior_sigma2)
    posts = [GaussianArmPosterior(mu=m, sigma2=s) for m, s in zip(prior_mu, prior_sigma2, strict=True)]
    for _ in range(300):
        k = int(rng.integers(6))
        r = float(rng.normal(k, 2))
        w = float(rng.choice([1.0, 0.5]))
        arms.update(k, r, w)
        posts[k].update(r, w)
    for k in range(6):
        assert arms.posterior(k) == posts[k]
    restored = GaussianArmSet.from_dict(json.loads(json.dumps(arms.to_dict(), allow_nan=False)))
    for name in ("mu", "sigma2", "n", "m2", "prior_mu", "prior_sigma2", "selections"):
        assert np.array_equal(getattr(restored, name), getattr(arms, name))
    assert restored.n_updates == arms.n_updates == 300
    assert posterior_variance(2.0, 1.0, 5.0) == 5.0


def test_arm_set_sampling_uses_one_vectorised_draw():
    arms = GaussianArmSet(np.array([0.0, 10.0, 20.0]), np.array([1.0, 1.0, 1.0]))
    idx = np.array([0, 2])
    a = arms.sample(np.random.default_rng(5), idx)
    b = np.random.default_rng(5).normal(arms.mu[idx], arms.std(idx))
    assert np.array_equal(a, b)


def test_invalid_posteriors_raise():
    with pytest.raises(InvalidPosteriorError):
        GaussianArmPosterior(mu=float("nan"))
    with pytest.raises(InvalidPosteriorError):
        GaussianArmPosterior(sigma2=-1.0)
    with pytest.raises(InvalidPosteriorError):
        GaussianArmSet(np.array([0.0, 1.0]), np.array([1.0]))
    assert welford_update(1.0, 0.0, 0.0, 5.0, 0.0) == (1.0, 0.0, 0.0)
