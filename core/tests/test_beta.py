"""BetaPosterior (SPEC §2.1): moments, conjugate update, HDI, method of moments, guards."""

import numpy as np
import pytest
from scipy import stats

from bayesian_credit.exceptions import InvalidPosteriorError, NumericalInstabilityError
from bayesian_credit.models.beta import BetaPosterior


@pytest.mark.parametrize(("a", "b"), [(1.0, 1.0), (2.0, 20.0), (0.5, 3.0), (40.0, 7.0)])
def test_moments_match_scipy(a, b):
    post = BetaPosterior(a, b)
    ref = stats.beta(a, b)
    assert post.mean == pytest.approx(ref.mean(), rel=1e-12)
    assert post.var == pytest.approx(ref.var(), rel=1e-12)
    assert post.std == pytest.approx(ref.std(), rel=1e-12)
    assert post.second_moment == pytest.approx(ref.moment(2), rel=1e-12)
    assert post.concentration == pytest.approx(a + b)


def test_mode():
    assert BetaPosterior(3.0, 5.0).mode == pytest.approx(2.0 / 6.0)
    assert BetaPosterior(0.5, 5.0).mode is None


def test_conjugate_update_matches_proposal_posterior_mean():
    # p | D ~ Beta(α + d, β + n),  E[p] = (α + d) / (α + β + d + n)
    alpha, beta, d, n = 2.5, 17.5, 7, 93
    post = BetaPosterior(alpha, beta).update(d, n)
    assert (post.alpha, post.beta) == (alpha + d, beta + n)
    assert post.mean == pytest.approx((alpha + d) / (alpha + beta + d + n), rel=1e-15)


def test_sequential_updates_equal_batch_update():
    y = np.random.default_rng(0).random(200) < 0.2
    post = BetaPosterior(1.0, 1.0)
    for v in y:
        post = post.update(int(v), int(not v))
    batch = BetaPosterior(1.0, 1.0).update(int(y.sum()), int((~y).sum()))
    assert (post.alpha, post.beta) == (batch.alpha, batch.beta)


def test_hdi_contains_mean_and_is_narrower_than_equal_tailed_for_skewed_beta():
    post = BetaPosterior(2.0, 20.0)  # right-skewed
    lo, hi = post.hdi(0.95)
    assert lo < post.mean < hi
    assert post.cdf(hi) - post.cdf(lo) == pytest.approx(0.95, abs=1e-6)
    et_lo, et_hi = post.quantile(0.025), post.quantile(0.975)
    assert hi - lo < et_hi - et_lo
    pdf = stats.beta(2.0, 20.0).pdf
    assert pdf(lo) == pytest.approx(pdf(hi), rel=0.02)  # equal density at HDI endpoints


def test_hdi_of_symmetric_beta_is_equal_tailed():
    post = BetaPosterior(5.0, 5.0)
    lo, hi = post.hdi(0.9)
    assert lo == pytest.approx(post.quantile(0.05), abs=1e-4)
    assert hi == pytest.approx(post.quantile(0.95), abs=1e-4)


@pytest.mark.parametrize(("a", "b"), [(2.0, 8.0), (30.0, 70.0), (0.8, 2.5), (5.0, 1.5)])
def test_method_of_moments_round_trips_mean_and_variance(a, b):
    post = BetaPosterior(a, b)
    fitted = BetaPosterior.from_moments(post.mean, post.var)
    assert fitted.alpha == pytest.approx(a, rel=1e-9)
    assert fitted.beta == pytest.approx(b, rel=1e-9)
    assert fitted.mean == pytest.approx(post.mean, rel=1e-12)
    assert fitted.var == pytest.approx(post.var, rel=1e-9)


def test_method_of_moments_formula():
    # α = μ(μ(1−μ)/σ² − 1),  β = (1−μ)(μ(1−μ)/σ² − 1)
    mu, var = 0.3, 0.01
    k = mu * (1 - mu) / var - 1
    post = BetaPosterior.from_moments(mu, var)
    assert post.alpha == pytest.approx(mu * k, rel=1e-12)
    assert post.beta == pytest.approx((1 - mu) * k, rel=1e-12)


def test_from_mean_concentration():
    post = BetaPosterior.from_mean_concentration(0.2, 50.0)
    assert (post.alpha, post.beta) == pytest.approx((10.0, 40.0))


def test_near_zero_variance_is_capped_at_max_concentration():
    post = BetaPosterior.from_moments(0.4, 0.0, max_concentration=1000.0)
    assert post.concentration == pytest.approx(1000.0)
    assert post.mean == pytest.approx(0.4)


@pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (1.0, -2.0), (np.inf, 1.0), (np.nan, 1.0)])
def test_invalid_parameters_raise(a, b):
    with pytest.raises(InvalidPosteriorError):
        BetaPosterior(a, b)


@pytest.mark.parametrize(("mu", "var"), [(0.5, 0.25), (0.2, 0.5), (0.2, -1e-3), (0.2, np.nan)])
def test_method_of_moments_guards_raise_numerical_instability(mu, var):
    with pytest.raises(NumericalInstabilityError):
        BetaPosterior.from_moments(mu, var)


def test_other_guards():
    with pytest.raises(InvalidPosteriorError):
        BetaPosterior(1.0, 1.0).update(-1, 0)
    with pytest.raises(InvalidPosteriorError):
        BetaPosterior(1.0, 1.0).hdi(1.0)
    with pytest.raises(InvalidPosteriorError):
        BetaPosterior.from_mean_concentration(0.3, 0.0)


def test_sampling_is_reproducible_and_inside_unit_interval(rng):
    post = BetaPosterior(3.0, 9.0)
    x = post.sample(rng, 20_000)
    assert x.shape == (20_000,)
    assert np.all((x > 0) & (x < 1))
    assert x.mean() == pytest.approx(post.mean, abs=0.005)
    a = post.sample(np.random.default_rng(7), 5)
    b = post.sample(np.random.default_rng(7), 5)
    assert np.array_equal(a, b)


def test_pdf_grid_and_to_dict():
    x, y = BetaPosterior(0.5, 0.5).pdf_grid(11)
    assert x.shape == y.shape == (11,)
    assert np.all(np.isfinite(y))
    d = BetaPosterior(2.0, 5.0).to_dict()
    assert {"alpha", "beta", "mean", "std", "hdi_low", "hdi_high", "concentration"} <= set(d)
