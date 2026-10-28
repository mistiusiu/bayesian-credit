"""Stage 1 — HierarchicalBetaBernoulli (SPEC §2.2, proposal §3.3)."""

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from bayesian_credit.config import HierarchyConfig
from bayesian_credit.models.beta import BetaPosterior
from bayesian_credit.models.hierarchical import (
    AUTO_KAPPA_FALLBACK,
    HierarchicalBetaBernoulli,
    beta_binomial_kappa,
    beta_hdi,
)

SUMMARY_COLUMNS = [
    "level", "group", "arm", "alpha", "beta", "n_obs", "defaults",
    "mean", "hdi_low", "hdi_high", "shrinkage_weight",
]


def toy_data():
    """A/1: 3 of 20 default; A/2: 1 of 10; B/1: 6 of 20."""
    rows = (
        [("A", 1, 1)] * 3 + [("A", 1, 0)] * 17
        + [("A", 2, 1)] * 1 + [("A", 2, 0)] * 9
        + [("B", 1, 1)] * 6 + [("B", 1, 0)] * 14
    )
    g, a, d = zip(*rows, strict=True)
    return list(g), list(a), list(d)


@pytest.fixture
def toy():
    cfg = HierarchyConfig(alpha0=1.0, beta0=1.0, kappa_group=20.0, kappa_leaf=10.0)
    return HierarchicalBetaBernoulli(cfg).fit(*toy_data())


def heterogeneous_data(seed=3, n=200):
    rng = np.random.default_rng(seed)
    groups, arms, defaults = [], [], []
    for gi, rate in enumerate([0.02, 0.05, 0.10, 0.20, 0.35]):
        for arm in range(4):
            groups += [f"g{gi}"] * n
            arms += [arm] * n
            defaults += list((rng.random(n) < min(rate * (1 + 0.3 * arm), 0.95)).astype(int))
    return groups, arms, defaults


def test_counts_propagate_to_all_levels(toy):
    assert toy.counts("A", 1) == (3, 17)
    assert toy.counts("A", 2) == (1, 9)
    assert toy.counts("B", 1) == (6, 14)
    assert toy.group_counts("A") == (4, 26)
    assert toy.group_counts("B") == (6, 14)
    assert toy.global_counts() == (10, 40)
    toy.update("A", 2, True)
    toy.update("C", 5, 0)
    assert toy.counts("A", 2) == (2, 9)
    assert toy.group_counts("A") == (5, 26)
    assert toy.counts("C", 5) == (0, 1)
    assert toy.group_counts("C") == (0, 1)
    assert toy.global_counts() == (11, 41)
    assert toy.n_obs("A", 2) == 11


def test_priors_are_built_top_down_and_posterior_matches_proposal(toy):
    g0 = toy.global_posterior()
    assert (g0.alpha, g0.beta) == (1.0 + 10, 1.0 + 40)
    gp = toy.group_prior("A")
    assert (gp.alpha, gp.beta) == pytest.approx((20 * g0.mean, 20 * (1 - g0.mean)))
    ga = toy.group_posterior("A")
    assert (ga.alpha, ga.beta) == pytest.approx((gp.alpha + 4, gp.beta + 26))
    lp = toy.leaf_prior("A", 1)
    assert (lp.alpha, lp.beta) == pytest.approx((10 * ga.mean, 10 * (1 - ga.mean)))
    post = toy.posterior("A", 1)
    # Beta(α_{g,a} + d, β_{g,a} + n) with mean (α + d)/(α + β + d + n)
    assert (post.alpha, post.beta) == pytest.approx((lp.alpha + 3, lp.beta + 17))
    assert toy.p_default("A", 1) == pytest.approx((lp.alpha + 3) / (lp.alpha + lp.beta + 3 + 17))


def test_unseen_leaf_equals_its_shrunk_group_posterior(toy):
    expected = BetaPosterior.from_mean_concentration(toy.group_posterior("A").mean, 10.0)
    post = toy.posterior("A", 99)
    assert (post.alpha, post.beta) == pytest.approx((expected.alpha, expected.beta))
    assert post == toy.leaf_prior("A", 99)
    assert toy.counts("A", 99) == (0, 0)


def test_unseen_group_returns_group_prior(toy):
    expected = BetaPosterior.from_mean_concentration(toy.global_posterior().mean, 20.0)
    post = toy.posterior("Z|never|seen", 1)
    assert (post.alpha, post.beta) == pytest.approx((expected.alpha, expected.beta))
    assert post == toy.group_prior()


def test_kappa_controls_shrinkage_between_group_mean_and_empirical_rate():
    data = toy_data()
    big = HierarchicalBetaBernoulli(HierarchyConfig(kappa_leaf=1e8)).fit(*data)
    assert big.p_default("A", 1) == pytest.approx(big.group_posterior("A").mean, rel=1e-5)
    assert big.shrinkage_weight("A", 1) == pytest.approx(0.0, abs=1e-6)
    small = HierarchicalBetaBernoulli(HierarchyConfig(kappa_leaf=1e-7)).fit(*data)
    assert small.p_default("A", 1) == pytest.approx(3 / 20, rel=1e-6)
    assert small.shrinkage_weight("A", 1) == pytest.approx(1.0, abs=1e-6)
    # same at the group level
    big_g = HierarchicalBetaBernoulli(HierarchyConfig(kappa_group=1e8)).fit(*data)
    assert big_g.group_posterior("B").mean == pytest.approx(big_g.global_posterior().mean, rel=1e-5)
    small_g = HierarchicalBetaBernoulli(HierarchyConfig(kappa_group=1e-7)).fit(*data)
    assert small_g.group_posterior("B").mean == pytest.approx(6 / 20, rel=1e-6)


def test_posterior_mean_is_shrinkage_weighted_average(toy):
    s = toy.summary()
    rows = s[s["level"] != "global"]
    w = rows["shrinkage_weight"].to_numpy()
    blend = w * rows["empirical_rate"].to_numpy() + (1 - w) * rows["prior_mean"].to_numpy()
    np.testing.assert_allclose(rows["mean"].to_numpy(), blend, rtol=1e-12)


def test_online_updates_keep_priors_frozen_until_refresh(toy):
    prior_before = toy.leaf_prior("A", 1)
    group_prior_before = toy.group_prior()
    for _ in range(50):
        toy.update("A", 2, True)
    assert toy.leaf_prior("A", 1) == prior_before
    assert toy.group_prior() == group_prior_before
    assert toy.counts("A", 2) == (51, 9)
    assert toy.group_counts("A") == (54, 26)
    assert toy.global_counts() == (60, 40)
    post = toy.posterior("A", 2)  # posterior still reflects all of the new data
    assert (post.alpha, post.beta) == pytest.approx((prior_before.alpha + 51, prior_before.beta + 9))
    assert toy.posterior("A", 1) == prior_before.update(3, 17)
    toy.refresh_priors()
    assert toy.leaf_prior("A", 1).mean > prior_before.mean  # group A became riskier


def test_update_ignores_unmapped_arm_and_missing_flag(toy):
    before = toy.to_dict()
    toy.update("A", -1, True)
    toy.update("A", 1, None)
    toy.update("A", 1, float("nan"))
    assert toy.to_dict() == before
    with pytest.raises(ValueError):
        toy.update("A", 1, 0.5)


def test_fit_equals_sequential_updates_after_refresh():
    g, a, d = toy_data()
    batch = HierarchicalBetaBernoulli().fit(g, a, d)
    online = HierarchicalBetaBernoulli()
    for gi, ai, di in zip(g, a, d, strict=True):
        online.update(gi, ai, di)
    online.refresh_priors()
    pd.testing.assert_frame_equal(batch.summary(), online.summary())


def test_rows_with_unmapped_arm_or_missing_values_are_skipped():
    m = HierarchicalBetaBernoulli().fit(
        ["A", "A", "A", None, "B"], [1, -1, 2, 1, np.nan], [1, 0, np.nan, 1, 0]
    )
    assert m.global_counts() == (1, 0)
    assert m.leaves == [("A", 1)]
    assert m.diagnostics["skipped"] == 4


def test_fit_accepts_pandas_and_numpy_inputs():
    g, a, d = toy_data()
    frame = pd.DataFrame({"g": pd.Series(g, dtype="string"), "a": a, "d": pd.Series(d, dtype="Float64")})
    from_pandas = HierarchicalBetaBernoulli().fit(frame["g"], frame["a"], frame["d"])
    from_numpy = HierarchicalBetaBernoulli().fit(np.array(g), np.array(a), np.array(d, dtype=float))
    pd.testing.assert_frame_equal(from_pandas.summary(), from_numpy.summary())


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        HierarchicalBetaBernoulli().fit(["A"], [1], [0.5])
    with pytest.raises(ValueError):
        HierarchicalBetaBernoulli().fit(["A", "B"], [1], [0, 1])
    with pytest.raises(ValueError):
        HierarchicalBetaBernoulli().fit(["A"], [1.5], [1])


def test_summary_shape_columns_and_values(toy):
    s = toy.summary()
    assert set(SUMMARY_COLUMNS) <= set(s.columns)
    assert len(s) == 1 + 2 + 3
    assert s["level"].tolist() == ["global", "group", "group", "leaf", "leaf", "leaf"]
    assert s["arm"].isna().tolist() == [True, True, True, False, False, False]
    assert s["group"].isna().tolist() == [True] + [False] * 5
    assert s.loc[0, "shrinkage_weight"] == 0.0
    leaf = s[(s["level"] == "leaf") & (s["group"] == "A") & (s["arm"] == 1)].iloc[0]
    assert leaf["n_obs"] == 20 and leaf["defaults"] == 3
    assert leaf["shrinkage_weight"] == pytest.approx(20 / (20 + 10))
    assert leaf["key"] == "A#1"
    grp = s[(s["level"] == "group") & (s["group"] == "B")].iloc[0]
    assert grp["shrinkage_weight"] == pytest.approx(20 / (20 + 20))
    assert np.all(s["hdi_low"] < s["mean"]) and np.all(s["mean"] < s["hdi_high"])
    post = toy.posterior("A", 1)
    ref_lo, ref_hi = post.hdi(0.95)
    assert leaf["hdi_low"] == pytest.approx(ref_lo, abs=1e-4)
    assert leaf["hdi_high"] == pytest.approx(ref_hi, abs=1e-4)
    assert leaf["mean"] == pytest.approx(post.mean)


def test_summary_of_unfitted_model_has_only_global_row():
    s = HierarchicalBetaBernoulli().summary()
    assert len(s) == 1 and s.loc[0, "level"] == "global" and s.loc[0, "n_obs"] == 0


def test_to_dict_round_trip_is_exact(toy):
    toy.update("A", 7, True)  # online-created leaf ...
    toy.update("Q", 3, False)  # ... and group, with priors still frozen
    state = toy.to_dict()
    restored = HierarchicalBetaBernoulli.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state
    pd.testing.assert_frame_equal(restored.summary(), toy.summary())
    for g, a in [("A", 1), ("A", 7), ("Q", 3), ("Q", 4), ("Z", 1)]:
        assert restored.posterior(g, a) == toy.posterior(g, a)


def test_auto_kappa_is_finite_and_within_bounds():
    cfg = HierarchyConfig(kappa_group="auto", kappa_leaf="auto", kappa_min=1.0, kappa_max=1000.0)
    m = HierarchicalBetaBernoulli(cfg).fit(*heterogeneous_data())
    assert np.isfinite(m.kappa_group)
    assert 1.0 <= m.kappa_group <= 1000.0
    assert m.kappa_group < 100  # strongly heterogeneous groups -> little pooling
    for g in m.groups:
        assert np.isfinite(m.kappa_leaf(g))
        assert 1.0 <= m.kappa_leaf(g) <= 1000.0
    assert m.diagnostics["kappa_group_source"] == "auto"
    assert m.diagnostics["kappa_leaf_estimated_groups"] == 5
    s = m.summary()
    assert np.all(np.isfinite(s["kappa"])) and np.all((s["mean"] > 0) & (s["mean"] < 1))


def test_auto_kappa_respects_custom_bounds():
    cfg = HierarchyConfig(kappa_group="auto", kappa_leaf="auto", kappa_min=50.0, kappa_max=60.0)
    m = HierarchicalBetaBernoulli(cfg).fit(*heterogeneous_data())
    assert 50.0 <= m.kappa_group <= 60.0
    assert all(50.0 <= m.kappa_leaf(g) <= 60.0 for g in m.groups)


def test_auto_kappa_falls_back_when_not_identifiable():
    cfg = HierarchyConfig(kappa_group="auto", kappa_leaf="auto")
    m = HierarchicalBetaBernoulli(cfg).fit(["A"] * 10, [1] * 10, [0] * 9 + [1])
    assert m.kappa_group == AUTO_KAPPA_FALLBACK["group"]
    assert m.kappa_leaf("A") == AUTO_KAPPA_FALLBACK["leaf"]
    assert m.diagnostics["kappa_group_source"] == "auto_fallback"


def test_beta_binomial_kappa_formula_and_edge_cases():
    d, n = np.array([5.0, 20.0, 40.0]), np.array([100.0, 100.0, 100.0])
    r = d / n
    m = r.mean()
    v = ((r - m) ** 2).mean() - m * (1 - m) / 100.0
    assert beta_binomial_kappa(d, n) == pytest.approx(m * (1 - m) / v - 1)
    assert beta_binomial_kappa([10] * 5, [100] * 5, 1.0, 500.0) == 500.0  # homogeneous -> pool
    assert beta_binomial_kappa([0, 0, 0], [10, 20, 30], 1.0, 300.0) == 300.0  # m = 0
    assert beta_binomial_kappa([1], [10]) is None
    assert beta_binomial_kappa([1, 2], [3, 4]) is None  # children below min_obs


@pytest.mark.parametrize(("a", "b"), [(3.0, 40.0), (2.0, 2.0), (50.0, 500.0), (0.4, 9.6), (9.6, 0.4), (1.0, 30.0)])
def test_vectorised_hdi_matches_beta_posterior_hdi(a, b):
    lo, hi = beta_hdi([a], [b], 0.95)
    ref_lo, ref_hi = BetaPosterior(a, b).hdi(0.95)
    assert lo[0] == pytest.approx(ref_lo, abs=2e-4)
    assert hi[0] == pytest.approx(ref_hi, abs=2e-4)
    assert stats.beta.cdf(hi[0], a, b) - stats.beta.cdf(lo[0], a, b) == pytest.approx(0.95, abs=1e-7)


def test_vectorised_hdi_broadcasts():
    alphas, betas = np.array([2.0, 5.0, 30.0]), np.array([10.0, 5.0, 3.0])
    lo, hi = beta_hdi(alphas, betas, 0.9)
    for k in range(3):
        lk, hk = beta_hdi(alphas[k], betas[k], 0.9)
        assert (lo[k], hi[k]) == pytest.approx((lk[0], hk[0]), abs=1e-12)
    assert beta_hdi([], [])[0].size == 0


@settings(max_examples=60, deadline=None)
@given(
    data=st.lists(
        st.tuples(st.sampled_from(["A", "B", "C"]), st.integers(0, 4), st.booleans()), max_size=60
    ),
    alpha0=st.floats(0.05, 50.0),
    beta0=st.floats(0.05, 50.0),
    kappa_group=st.one_of(st.floats(0.01, 1e4), st.just("auto")),
    kappa_leaf=st.one_of(st.floats(0.01, 1e4), st.just("auto")),
    query=st.tuples(st.sampled_from(["A", "B", "C", "unseen"]), st.integers(0, 6)),
)
def test_posterior_mean_always_in_open_unit_interval(data, alpha0, beta0, kappa_group, kappa_leaf, query):
    m = HierarchicalBetaBernoulli(HierarchyConfig(alpha0, beta0, kappa_group, kappa_leaf))
    if data:
        g, a, d = zip(*data, strict=True)
        m.fit(g, a, [int(x) for x in d])
    post = m.posterior(*query)
    assert 0.0 < post.mean < 1.0
    assert np.isfinite(post.alpha) and np.isfinite(post.beta)
    m.update(query[0], query[1], True)
    assert 0.0 < m.p_default(*query) < 1.0
    assert 0.0 < m.group_posterior(query[0]).mean < 1.0


# ----------------------------------------------------------------------------------------
# structured leaf prior (leaf-prior tilt; docs/MATH.md "Structured leaf prior")
# ----------------------------------------------------------------------------------------
def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    return np.log(p) - np.log1p(-p)


def hand_policy_model():
    from bayesian_credit.models.policy_effect import PolicyEffectModel

    m = PolicyEffectModel()
    m.beta = np.array([8.0, 0.5, 4.0, -0.2])
    m.group_means = {"A": np.array([0.08, 0.2, 0.08, 9.3]), "B": np.array([0.15, 0.3, 0.09, 9.4])}
    m.group_median_income = {"A": 70_000.0, "B": 60_000.0}
    m.global_means = np.array([0.12, 0.25, 0.085, 9.35])
    m.global_median_income = 65_000.0
    m.n_fit, m.fitted = 1234, True
    m.diagnostics = {"status": "fitted", "converged": True, "n_fit": 1234, "sample": "elapsed"}
    return m


def test_tilt_moves_the_leaf_prior_mean_on_the_logit_scale(toy):
    def tilt(g, a):
        return 0.5 * a if g == "A" else -1.0

    m_a = toy.group_posterior("A").mean  # the anchor frozen at fit
    m_0 = toy.global_posterior().mean
    untilted_post = toy.posterior("A", 2)
    toy.set_leaf_prior_tilt(tilt)
    assert toy.leaf_prior_tilt is tilt and toy.prior_tilt("A", 2) == 1.0 and toy.prior_tilt("A", None) == 0.0
    lp = toy.leaf_prior("A", 2)
    assert lp.mean == pytest.approx(_expit(_logit(m_a) + 1.0), rel=1e-12)
    assert lp.concentration == pytest.approx(10.0)  # κ_leaf unchanged
    assert toy.posterior("A", 2) == lp.update(1, 9)  # conjugate update unchanged
    assert toy.posterior("A", 2).mean > untilted_post.mean
    unseen_leaf = toy.posterior("A", 7)
    assert unseen_leaf == toy.leaf_prior("A", 7) and unseen_leaf.mean == pytest.approx(_expit(_logit(m_a) + 3.5))
    unseen_group = toy.posterior("Z", 3)  # unseen group: σ(logit(m_0) + t) with κ_group
    assert unseen_group.mean == pytest.approx(_expit(_logit(m_0) - 1.0)) and unseen_group.concentration == pytest.approx(20.0)
    assert toy.group_posterior("A").mean == pytest.approx(m_a)  # group / global levels untouched
    s = toy.summary()
    leaves = s[s["level"] == "leaf"]
    for _, row in leaves.iterrows():
        assert row["prior_tilt_logit"] == tilt(row["group"], int(row["arm"]))
        assert row["prior_mean"] == pytest.approx(toy.leaf_prior(row["group"], int(row["arm"])).mean)
    assert (s.loc[s["level"] != "leaf", "prior_tilt_logit"] == 0.0).all()


def test_zero_tilt_is_exactly_the_untilted_model():
    data = toy_data()
    plain = HierarchicalBetaBernoulli().fit(*data)
    zero = HierarchicalBetaBernoulli(leaf_prior_tilt=lambda g, a: 0.0).fit(*data)
    for g, a in [("A", 1), ("A", 2), ("A", 9), ("B", 1), ("Q", 4)]:
        assert zero.posterior(g, a) == plain.posterior(g, a)
    pd.testing.assert_frame_equal(zero.summary().drop(columns="prior_tilt_logit"), plain.summary().drop(columns="prior_tilt_logit"))


def test_tilt_survives_fit_and_refresh_reapplies_it_to_the_new_anchor():
    m = HierarchicalBetaBernoulli(leaf_prior_tilt=lambda g, a: 1.0)
    m.fit(*toy_data())
    assert m.leaf_prior_tilt is not None  # fit() replaces data, not the tilt
    before = m.leaf_prior("A", 1).mean
    for _ in range(40):
        m.update("A", 2, True)
    assert m.leaf_prior("A", 1).mean == before  # priors frozen between refreshes (H2)
    m.refresh_priors()
    after = m.leaf_prior("A", 1).mean
    assert after > before
    assert after == pytest.approx(_expit(_logit(m.group_posterior("A").mean) + 1.0), rel=1e-12)
    m.set_leaf_prior_tilt(None)
    assert m.leaf_prior("A", 1).mean == pytest.approx(m.group_posterior("A").mean)


def test_leaf_data_dominate_the_tilted_prior_as_they_accumulate():
    gaps = []
    for n in (10, 100, 1_000, 10_000):
        groups = ["A"] * (n + 200)
        arms = [1] * n + [2] * 200
        defaults = [1] * (n // 10) + [0] * (n - n // 10) + [1] * 20 + [0] * 180
        m = HierarchicalBetaBernoulli(leaf_prior_tilt=lambda g, a: 2.5 if a == 1 else 0.0).fit(groups, arms, defaults)
        assert m.leaf_prior("A", 1).mean > 0.4  # the prior says "much riskier" ...
        gaps.append(abs(m.p_default("A", 1) - 0.1))  # ... the data say 10 %
        assert m.shrinkage_weight("A", 1) == pytest.approx(n / (n + 10.0))
    assert all(a > b for a, b in zip(gaps, gaps[1:], strict=False)) and gaps[-1] < 0.002


def test_round_trip_with_a_policy_effect_tilt(arm_space):
    from bayesian_credit.models.policy_effect import PolicyEffectTilt, TiltTable

    model = hand_policy_model()
    m = HierarchicalBetaBernoulli(leaf_prior_tilt=PolicyEffectTilt(model, arm_space)).fit(*toy_data())
    m.update("Q", 3, True)  # online-created group
    state = m.to_dict()
    assert state["leaf_prior_tilt"]["type"] == "PolicyEffectTilt"
    assert len(state["leaf_tilt_table"]) == 3 * arm_space.n_arms  # seen groups x every arm of the space
    restored = HierarchicalBetaBernoulli.from_dict(json.loads(json.dumps(state)))
    assert isinstance(restored.leaf_prior_tilt, PolicyEffectTilt)
    assert restored.to_dict() == state
    pd.testing.assert_frame_equal(restored.summary(), m.summary())
    for g in ("A", "B", "Q", "Z|unseen"):
        for a in (0, 1, 2, 64, 128):
            assert restored.posterior(g, a) == m.posterior(g, a)
    assert m.prior_tilt("A", 128) == model.effect("A", arm_space[128]) != 0.0
    assert m.prior_tilt("A", 0) == 0.0  # the decline arm
    d = m.diagnostics["policy_effect"]
    assert d["installed"] and d["beta"] == model.coefficients and d["n_fit"] == 1234 and d["type"] == "PolicyEffectTilt"
    assert HierarchicalBetaBernoulli().diagnostics["policy_effect"] == {"installed": False}
    # a tilt that cannot serialise itself is restored from the table of values it produced
    lam = HierarchicalBetaBernoulli(leaf_prior_tilt=lambda g, a: 0.3 * a).fit(*toy_data())
    st = lam.to_dict()
    assert st["leaf_prior_tilt"] is None and sorted(map(tuple, st["leaf_tilt_table"])) == [
        ("A", 1, 0.3), ("A", 2, 0.6), ("B", 1, 0.3), ("B", 2, 0.6)
    ]
    back = HierarchicalBetaBernoulli.from_dict(json.loads(json.dumps(st)))
    assert isinstance(back.leaf_prior_tilt, TiltTable)
    for g, a in [("A", 1), ("A", 2), ("B", 1), ("B", 2)]:
        assert back.posterior(g, a) == lam.posterior(g, a)


def test_invalid_tilts_are_rejected():
    from bayesian_credit.exceptions import NumericalInstabilityError

    m = HierarchicalBetaBernoulli(leaf_prior_tilt=lambda g, a: float("nan")).fit(*toy_data())
    with pytest.raises(NumericalInstabilityError):
        m.posterior("A", 1)
    with pytest.raises(TypeError):
        m.set_leaf_prior_tilt(3.0)


@settings(max_examples=40, deadline=None)
@given(
    tilt=st.floats(-3.0, 3.0),
    query=st.tuples(st.sampled_from(["A", "B", "unseen"]), st.integers(0, 6)),
    kappa_leaf=st.one_of(st.floats(0.01, 1e4), st.just("auto")),
)
def test_tilted_posterior_mean_stays_in_the_open_unit_interval(tilt, query, kappa_leaf):
    m = HierarchicalBetaBernoulli(HierarchyConfig(kappa_leaf=kappa_leaf), leaf_prior_tilt=lambda g, a: tilt)
    m.fit(*toy_data())
    post = m.posterior(*query)
    assert 0.0 < post.mean < 1.0 and np.isfinite(post.alpha) and np.isfinite(post.beta)
