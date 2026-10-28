"""Stage 2 — BetaMoMRecovery (SPEC §2.3, proposal §3.3)."""

import json

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import RecoveryConfig
from bayesian_credit.exceptions import NumericalInstabilityError
from bayesian_credit.models.beta import BetaPosterior
from bayesian_credit.models.recovery import FALLBACK_PRIOR, BetaMoMRecovery, RunningMoments


@pytest.fixture
def chain_data():
    """A/1: 30 obs (own fit); A/2: 3 obs (too few -> group A); B/1: 2 obs (B too thin -> global)."""
    x = list(np.random.default_rng(0).beta(2.0, 5.0, 30)) + [0.2, 0.3, 0.4] + [0.5, 0.6]
    g = ["A"] * 33 + ["B"] * 2
    a = [1] * 30 + [2] * 3 + [1] * 2
    return g, a, x


def test_fallback_chain_leaf_group_global_prior(chain_data):
    m = BetaMoMRecovery(RecoveryConfig(min_obs=5)).fit(*chain_data)
    assert m.resolve("A", 1)[1] == "leaf"
    assert m.resolve("A", 2)[1] == "group"
    assert m.resolve("A", 99)[1] == "group"  # unseen leaf of a known group
    assert m.resolve("B", 1)[1] == "global"
    assert m.resolve("Z", 1)[1] == "global"  # unseen group
    assert m.posterior("A", 2) == m.group_posterior("A")
    assert m.posterior("B", 1) == m.global_posterior()
    post, src = BetaMoMRecovery().resolve("A", 1)  # nothing at all -> weakly informative prior
    assert src == "prior"
    assert (post.alpha, post.beta) == FALLBACK_PRIOR
    d = m.diagnostics
    assert d["source_leaf"] == 1 and d["source_group"] == 1 and d["source_global"] == 1
    assert d["insufficient_leaf"] == 2 and d["insufficient_group"] == 1
    assert d["global_source"] == "global"


def test_leaf_beta_is_the_method_of_moments_of_its_data(chain_data):
    g, a, x = chain_data
    m = BetaMoMRecovery().fit(g, a, x)
    s = m.stats("A", 1)
    assert s.n == 30
    assert s.mean == pytest.approx(np.mean(x[:30]), abs=1e-12)
    assert s.var == pytest.approx(np.var(x[:30], ddof=1), abs=1e-12)
    mu, var = s.mean, s.var
    k = mu * (1 - mu) / var - 1
    post = m.posterior("A", 1)
    assert (post.alpha, post.beta) == pytest.approx((mu * k, (1 - mu) * k), rel=1e-12)
    assert m.expected_recovery("A", 1) == pytest.approx(post.mean)


def test_invalid_variance_leaf_falls_back_and_is_counted():
    # leaf A/1 recoveries alternate 0/1: sample variance 0.278 > μ(1−μ) = 0.25
    g = ["A"] * 30
    a = [1] * 10 + [2] * 20
    x = [0.0, 1.0] * 5 + list(np.linspace(0.1, 0.6, 20))
    m = BetaMoMRecovery().fit(g, a, x)
    s = m.stats("A", 1)
    with pytest.raises(NumericalInstabilityError):
        BetaPosterior.from_moments(s.mean, s.var)
    post, src = m.resolve("A", 1)
    assert src == "group"
    assert post == m.group_posterior("A")
    d = m.diagnostics
    assert d["invalid_leaf"] == 1
    assert d["source_group"] == 1 and d["source_leaf"] == 1


def test_degenerate_global_falls_back_to_prior():
    m = BetaMoMRecovery(RecoveryConfig(min_obs=2)).fit(["A", "A"], [1, 1], [0.0, 1.0])
    post, src = m.resolve("A", 1)
    assert src == "prior" and (post.alpha, post.beta) == FALLBACK_PRIOR
    assert m.diagnostics["invalid_global"] == 1 and m.diagnostics["global_source"] == "prior"


def test_zero_variance_leaf_is_capped_not_rejected():
    m = BetaMoMRecovery(RecoveryConfig(max_concentration=500.0)).fit(["A"] * 6, [1] * 6, [0.3] * 6)
    post, src = m.resolve("A", 1)
    assert src == "leaf"
    assert post.concentration == pytest.approx(500.0)
    assert post.mean == pytest.approx(0.3)


def test_online_update_matches_batch_fit():
    rng = np.random.default_rng(1)
    n = 500
    g = rng.choice(["A", "B", "C"], n)
    a = rng.integers(0, 4, n)
    x = rng.beta(1.2, 3.0, n)
    batch = BetaMoMRecovery().fit(g, a, x)
    warm = BetaMoMRecovery().fit(g[:200], a[:200], x[:200])
    cold = BetaMoMRecovery()
    for gi, ai, xi in zip(g[200:], a[200:], x[200:], strict=True):
        warm.update(gi, ai, xi)
    for gi, ai, xi in zip(g, a, x, strict=True):
        cold.update(gi, ai, xi)
    keys = [(None, None)] + [(k, None) for k in "ABC"] + [(k, j) for k in "ABC" for j in range(4)]
    for online in (warm, cold):
        for key in keys:
            sb, so = batch.stats(*key), online.stats(*key)
            assert so.n == sb.n
            assert so.mean == pytest.approx(sb.mean, abs=1e-9)
            assert so.var == pytest.approx(sb.var, abs=1e-9)
        for k in "ABC":
            for j in range(4):
                pb, po = batch.posterior(k, j), online.posterior(k, j)
                assert (po.alpha, po.beta) == pytest.approx((pb.alpha, pb.beta), rel=1e-7)


def test_running_moments_welford():
    x = np.random.default_rng(2).normal(5.0, 2.0, 1000)
    rm = RunningMoments()
    for v in x:
        rm.push(float(v))
    assert rm.n == 1000
    assert rm.mean == pytest.approx(x.mean(), rel=1e-12)
    assert rm.var == pytest.approx(x.var(ddof=1), rel=1e-10)
    assert np.isnan(RunningMoments(1, 0.4, 0.0).var)
    assert RunningMoments.from_list(rm.to_list()) == rm


def test_out_of_range_rates_are_clipped_and_missing_rows_skipped():
    m = BetaMoMRecovery().fit(["A", "A", "A", "A"], [1, 1, -1, 1], [1.2, -0.1, 0.5, np.nan])
    assert m.stats().n == 2
    assert m.stats().mean == pytest.approx(0.5)
    d = m.diagnostics
    assert d["clipped"] == 2 and d["skipped"] == 2
    m.update("A", 1, 1.5)
    m.update("A", -1, 0.5)
    m.update("A", 1, None)
    assert m.stats().n == 3 and m.diagnostics["clipped"] == 3


def test_summary_columns_rows_and_sources(chain_data):
    m = BetaMoMRecovery().fit(*chain_data)
    s = m.summary()
    assert {"level", "group", "arm", "n_obs", "mean", "var", "alpha", "beta", "source"} <= set(s.columns)
    assert s["level"].tolist() == ["global", "group", "group", "leaf", "leaf", "leaf"]
    assert set(s["source"]) <= {"leaf", "group", "global", "prior"}
    leaf = s[(s["level"] == "leaf") & (s["arm"] == 2)].iloc[0]
    assert leaf["source"] == "group" and leaf["n_obs"] == 3
    assert leaf["mean"] == pytest.approx(0.3)
    assert leaf["expected_recovery"] == pytest.approx(m.expected_recovery("A", 2))
    empty = BetaMoMRecovery().summary()
    assert len(empty) == 1 and empty.loc[0, "source"] == "prior" and np.isnan(empty.loc[0, "mean"])


def test_sample_shape_and_range(chain_data, rng):
    m = BetaMoMRecovery().fit(*chain_data)
    draws = m.sample("A", 1, 5000, rng)
    assert draws.shape == (5000,) and np.all((draws > 0) & (draws < 1))
    assert draws.mean() == pytest.approx(m.expected_recovery("A", 1), abs=0.02)


def test_to_dict_round_trip(chain_data):
    m = BetaMoMRecovery(RecoveryConfig(min_obs=4)).fit(*chain_data)
    m.update("C", 7, 0.25)
    state = m.to_dict()
    restored = BetaMoMRecovery.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state
    assert restored.cfg.min_obs == 4
    pd.testing.assert_frame_equal(restored.summary(), m.summary())
    assert restored.diagnostics == m.diagnostics


# ----- recovery by term: leaf -> (group, term) -> group -> global (docs/MATH.md §2.9) -----------------
TERMS = {0: 0, 1: 36, 2: 36, 3: 60, 4: 60}


def term_data():
    """Group A: 36m arms recover ~0.5, 60m arms ~0.3; every leaf below min_obs except A/1."""
    rng = np.random.default_rng(7)
    g = ["A"] * 40 + ["A"] * 4 + ["A"] * 12 + ["A"] * 3 + ["B"] * 6
    a = [1] * 40 + [2] * 4 + [3] * 12 + [4] * 3 + [3] * 6
    x = list(rng.beta(5, 5, 40)) + list(rng.beta(5, 5, 4)) + list(rng.beta(3, 7, 12)) + list(rng.beta(3, 7, 3)) + list(rng.beta(3, 7, 6))
    return g, a, x


def test_group_term_level_sits_between_leaf_and_group():
    g, a, x = term_data()
    m = BetaMoMRecovery(RecoveryConfig(min_obs=5), term_of_arm=TERMS).fit(g, a, x)
    assert m.has_terms and m.term_of(3) == 60 and m.term_of(0) is None and m.term_of(99) is None
    assert m.resolve("A", 1)[1] == "leaf"
    assert m.resolve("A", 2)[1] == "group_term"  # 4 obs < 5 -> A/36 (44 obs)
    assert m.resolve("A", 4)[1] == "group_term"  # A/60 has 15 obs
    assert m.resolve("A", 0)[1] == "group"  # decline arm: no term
    assert m.resolve("B", 3)[1] == "leaf" and m.resolve("B", 4)[1] == "group_term"  # unseen B leaf -> B/60 (6 obs)
    s36, s60 = m.stats("A", term=36), m.stats("A", term=60)
    assert s36.n == 44 and s60.n == 15
    assert s36.mean == pytest.approx(np.mean(x[:44])) and s60.mean == pytest.approx(np.mean(x[44:59]))
    assert m.posterior("A", 4).mean < m.posterior("A", 2).mean  # 60m borrows the 60m mean, not the group's
    post, src = m.group_term_posterior("A", 60)
    assert src == "group_term" and post == m.posterior("A", 4)
    m.update("A", 3, 0.25)
    assert m.stats("A", term=60).n == 16 and m.stats("A").n == 60
    d = m.diagnostics
    assert d["term_level"] and d["n_group_terms"] == 3 and d["source_group_term"] == 2  # A/2 and A/4
    assert {"insufficient_group_term", "invalid_group_term"} <= set(d)
    s = m.summary()
    gt = s[s["level"] == "group_term"]
    assert len(gt) == 3 and set(gt["term"].astype(int)) == {36, 60}
    assert (s.loc[s["level"] == "leaf", "term"].notna()).all()


def test_without_terms_the_chain_is_unchanged_and_round_trips(chain_data):
    plain = BetaMoMRecovery(RecoveryConfig(min_obs=5)).fit(*chain_data)
    assert not plain.has_terms and not plain.group_terms
    assert "group_term" not in set(plain.summary()["level"])
    g, a, x = term_data()
    m = BetaMoMRecovery(RecoveryConfig(min_obs=5), term_of_arm=TERMS).fit(g, a, x)
    state = m.to_dict()
    restored = BetaMoMRecovery.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state and restored.diagnostics == m.diagnostics
    for arm in range(5):
        assert restored.resolve("A", arm) == m.resolve("A", arm)
    old = {k: v for k, v in state.items() if k not in ("group_terms", "arm_terms")}  # a v1 dict
    v1 = BetaMoMRecovery.from_dict(old)
    assert not v1.has_terms and v1.resolve("A", 2)[1] == "group"
    by_fn = BetaMoMRecovery(RecoveryConfig(min_obs=5), term_of_arm=lambda arm: TERMS.get(arm)).fit(g, a, x)
    assert by_fn.resolve("A", 4) == m.resolve("A", 4)
    from bayesian_credit.data.arms import ArmSpace

    space = ArmSpace.from_grid([5_000], [0.1, 0.2], [36, 60])  # arms 1-2: 36m, 3-4: 60m
    assert BetaMoMRecovery(term_of_arm=space).term_of(4) == 60
