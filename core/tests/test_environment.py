"""BayesianEnvironment: posterior-predictive rewards, online learning, exact fast paths."""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig
from bayesian_credit.pipeline import fit_models, prepare_dataset
from bayesian_credit.simulation import BayesianEnvironment
from bayesian_credit.types import State


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, _ = prepare_dataset("synthetic", None, cfg=cfg, n=6000, seed=21)
    fitted = fit_models(df, cfg, "2015-01-01", evaluate=False)
    return cfg, df, fitted


def make_env() -> BayesianEnvironment:
    cfg, df, fitted = _data()
    train = df.loc[df["issue_d"] < "2015-01-01"]
    return BayesianEnvironment.fit(train, cfg, arm_space=fitted.arm_space, segmenter=fitted.segmenter,
                                   reward_fn=fitted.reward_fn, mctm=fitted.mctm)


def busiest_group() -> str:
    _, df, _ = _data()
    return str(df["y_group"].value_counts().index[0])


def test_terminal_draws_average_to_the_closed_form_mean():
    env = make_env()
    g = busiest_group()
    rng = np.random.default_rng(0)
    for arm in (5, 40, 100):
        draws = np.array([env.draw_terminal_reward(g, arm, rng) for _ in range(20_000)])
        mean = env.expected(g, arm).mean
        assert abs(draws.mean() - mean) <= 0.02 * abs(mean), (arm, draws.mean(), mean)
        batch = env.draw_terminal_rewards([g] * 20_000, [arm] * 20_000, np.random.default_rng(1))
        assert abs(batch.mean() - mean) <= 0.02 * abs(mean)
        assert batch.std() == pytest.approx(env.expected(g, arm).std, rel=0.05)


def test_vectorised_draws_are_deterministic_and_skip_the_null_arm():
    env = make_env()
    g = busiest_group()
    arms = np.array([0, 3, 0, 77, 128])
    a = env.draw_terminal_rewards([g] * 5, arms, np.random.default_rng(9))
    b = env.draw_terminal_rewards([g] * 5, arms, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    assert a[0] == 0.0 and a[2] == 0.0 and np.all(a[[1, 3, 4]] != 0.0)


def test_null_arm_is_worth_exactly_zero():
    env = make_env()
    g = busiest_group()
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert env.draw_terminal_reward(g, 0, rng) == 0.0
    assert rng.bit_generator.state == state  # no random numbers consumed
    assert env.expected(g, 0).mean == 0.0 and env.expected(g, 0).std == 0.0
    assert env.interim_reward(g, 0, State.CURRENT, 3, 3) == 0.0
    assert env.interim_rewards([g], [0], [1], [4], [2]).tolist() == [0.0]
    mu, sigma = env.prior_provider(g)
    assert mu[0] == 0.0 and sigma[0] == pytest.approx(1e-6) and np.all(sigma >= 1e-6)
    assert mu.shape == sigma.shape == (env.arm_space.n_arms,)


def test_observe_outcome_moves_the_posterior():
    env = make_env()
    g, arm = busiest_group(), 9
    before = env.risk(g, arm)
    exp_before = env.expected(g, arm).mean
    row = pd.Series({"y_group": g, "y_historical_arm": arm, "y_default": 1.0, "y_recovery_rate": 0.05})
    for _ in range(25):
        assert env.observe_outcome(row)
    after = env.risk(g, arm)
    assert after.p_default.mean > before.p_default.mean
    assert after.n_default_obs == before.n_default_obs + 25
    assert env.expected(g, arm).mean < exp_before
    assert env.n_observed == 25
    assert not env.observe_outcome(pd.Series({"y_group": g, "y_historical_arm": arm, "y_default": np.nan}))


def test_fast_risk_path_equals_the_two_stage_model_exactly():
    env = make_env()
    _, df, _ = _data()
    groups = sorted(df["y_group"].unique())[:12] + ["Z|never|SEEN"]
    k = env.arm_space.n_arms
    for g in groups:
        for a in range(k):
            assert env.risk(g, a) == env.two_stage.risk(g, a)
    stream = df.loc[df["issue_d"] >= "2015-01-01"].head(2000)
    assert env.observe_table(stream) > 0
    env.observe_outcomes(["Q|new|OWN"] * 3, [5, 5, 7], [1.0, 0.0, 1.0], [0.3, np.nan, 0.5])
    env.refresh()
    for g in groups + ["Q|new|OWN"]:
        for a in range(k):
            assert env.risk(g, a) == env.two_stage.risk(g, a)


def test_vectorised_interim_rewards_match_the_reward_function_bit_for_bit():
    env = make_env()
    _, df, fitted = _data()
    rng = np.random.default_rng(3)
    groups = list(df["y_group"].unique()[:10]) + ["Z|never|SEEN"]
    n = 4000
    g = rng.choice(groups, n)
    arms = rng.integers(0, env.arm_space.n_arms, n)
    states = rng.integers(0, 4, n)
    months = rng.integers(0, 61, n)
    paid = np.minimum(months, rng.integers(0, 61, n))
    fast = env.interim_rewards(g, arms, states, months, paid)
    ref = np.array(
        [
            fitted.reward_fn.interim(env.arm_space[int(a)], int(s), int(m), int(p), env.two_stage.risk(str(gg), int(a)), env.mctm).mean
            for gg, a, s, m, p in zip(g, arms, states, months, paid, strict=True)
        ]
    )
    np.testing.assert_array_equal(fast, ref)
    scalar = np.array([env.interim_reward(gg, a, s, m, p) for gg, a, s, m, p in zip(g, arms, states, months, paid, strict=True)])
    np.testing.assert_array_equal(scalar, ref)


def test_snapshot_restore_round_trip_and_resegment():
    env = make_env()
    _, df, fitted = _data()
    env.observe_table(df.loc[df["issue_d"] >= "2015-01-01"].head(500))
    snap = env.snapshot()
    copy = BayesianEnvironment.from_snapshot(snap, fitted.reward_fn, fitted.arm_space)
    g = busiest_group()
    for a in (0, 1, 50, 128):
        assert copy.risk(g, a) == env.risk(g, a)
        assert copy.expected(g, a) == env.expected(g, a)
    assert copy.n_observed == env.n_observed and copy.segmenter == env.segmenter
    finer = env.segmenter.refine()
    from bayesian_credit.models import TwoStageBetaBeta

    sub = df.loc[df["y_resolved"]]
    ts = TwoStageBetaBeta.from_config(FrameworkConfig()).fit(sub.assign(y_group=finer.assign(sub).astype(str)))
    env.resegment(finer, ts)
    assert env.segmenter == finer and len(env.segmenter.dimensions) == 4
    key = finer.assign(sub.head(1)).iloc[0]
    assert env.risk(key, 5) == ts.risk(key, 5)


def test_expected_table_and_score_use_the_risk_aversion():
    cfg, df, fitted = _data()
    from bayesian_credit.rewards import CompoundRewardFunction

    cautious = FrameworkConfig()
    cautious.reward.risk_aversion = 2.0
    train = df.loc[df["issue_d"] < "2015-01-01"]
    env = BayesianEnvironment.fit(train, cautious, arm_space=fitted.arm_space, segmenter=fitted.segmenter,
                                  reward_fn=CompoundRewardFunction(cautious.reward), mctm=fitted.mctm)
    g = busiest_group()
    mean, std, score = env.expected_table([g])
    np.testing.assert_allclose(score, mean - 2.0 * std)
    assert env.score(g, 7) == pytest.approx(score[0, 7])


def make_tilted_env() -> BayesianEnvironment:
    """The same environment with the structured leaf prior active (small data: lower min_rows)."""
    cfg, df, fitted = _data()
    tilted = FrameworkConfig.from_dict(cfg.to_dict())
    tilted.hierarchy.policy_effect_cfg.min_rows = 300
    train = df.loc[df["issue_d"] < "2015-01-01"]
    return BayesianEnvironment.fit(train, tilted, arm_space=fitted.arm_space, segmenter=fitted.segmenter,
                                   reward_fn=fitted.reward_fn, mctm=fitted.mctm)


def test_fast_paths_stay_exact_with_a_policy_effect_tilt():
    env = make_tilted_env()
    _, df, fitted = _data()
    assert env.two_stage.policy_effect_info["status"] == "fitted"
    groups = sorted(df["y_group"].unique())[:8] + ["Z|never|SEEN"]
    k = env.arm_space.n_arms
    tilts = [env.two_stage.default_model.prior_tilt(g, a) for g in groups for a in range(k)]
    assert max(abs(t) for t in tilts) > 0.5
    for g in groups:
        for a in range(k):
            assert env.risk(g, a) == env.two_stage.risk(g, a)
    rng = np.random.default_rng(4)
    n = 1500
    g = rng.choice(groups, n)
    arms = rng.integers(0, k, n)
    states = rng.integers(0, 4, n)
    months = rng.integers(0, 61, n)
    paid = np.minimum(months, rng.integers(0, 61, n))
    ref = np.array(
        [
            fitted.reward_fn.interim(env.arm_space[int(a)], int(s), int(m), int(p), env.two_stage.risk(str(gg), int(a)), env.mctm).mean
            for gg, a, s, m, p in zip(g, arms, states, months, paid, strict=True)
        ]
    )
    np.testing.assert_array_equal(env.interim_rewards(g, arms, states, months, paid), ref)
    env.observe_table(df.loc[df["issue_d"] >= "2015-01-01"].head(1500))
    env.refresh()
    for gg in groups:
        for a in range(0, k, 7):
            assert env.risk(gg, a) == env.two_stage.risk(gg, a)
    copy = BayesianEnvironment.from_snapshot(env.snapshot(), fitted.reward_fn, fitted.arm_space)
    for a in (0, 1, 50, 128):
        assert copy.risk(groups[0], a) == env.risk(groups[0], a)
        assert copy.expected(groups[0], a) == env.expected(groups[0], a)


def make_prepay_env(cfg_override=None) -> BayesianEnvironment:
    """Environment with the prepayment table of the fit (ρ varies with grade and term)."""
    cfg, df, fitted = _data()
    train = df.loc[df["issue_d"] < "2015-01-01"]
    use = cfg if cfg_override is None else cfg_override
    from bayesian_credit.rewards import CompoundRewardFunction

    return BayesianEnvironment.fit(
        train, use, arm_space=fitted.arm_space, segmenter=fitted.segmenter, reward_fn=CompoundRewardFunction(use.reward),
        mctm=fitted.mctm, interest_realization=fitted.interest_realization,
    )


def test_prepayment_factor_keeps_every_fast_path_exact():
    env = make_prepay_env()
    cfg, df, fitted = _data()
    assert fitted.interest_realization is not None
    groups = sorted(df["y_group"].unique())[:8] + ["Z|never|SEEN"]
    k = env.arm_space.n_arms
    rhos = {(g, a): env.realization(g, a) for g in groups for a in range(k)}
    assert len({round(v, 12) for v in rhos.values()}) > 2 and all(0.3 <= v <= 1.0 for v in rhos.values())
    for g in groups:
        for a in range(k):
            pol = env.arm_space[a]
            assert env.realization(g, a) == fitted.realization(g, a)  # same lookup as the fit / shadow engine
            if pol.is_null:
                continue
            ref = env.reward_fn.expected(pol, env.two_stage.risk(g, a), env.mctm, realization=rhos[(g, a)])
            assert env.expected(g, a) == ref
    # vectorised terminal draws == reward_fn.sample(..., realization=ρ) on the same random numbers
    rng = np.random.default_rng(5)
    n = 600
    g_arr = rng.choice(groups, n)
    arms = rng.integers(0, k, n)
    fast = env.draw_terminal_rewards(g_arr, arms, np.random.default_rng(11))
    lend = np.flatnonzero(arms != 0)
    est = [env.risk(g_arr[i], int(arms[i])) for i in lend]
    replay = np.random.default_rng(11)
    p = replay.beta([e.p_default.alpha for e in est], [e.p_default.beta for e in est])
    r = replay.beta([e.recovery.alpha for e in est], [e.recovery.beta for e in est])
    ref = np.zeros(n)
    for j, i in enumerate(lend):
        ref[i] = env.reward_fn.sample(env.arm_space[int(arms[i])], p[j], r[j], env.mctm, realization=env.realization(g_arr[i], int(arms[i])))
    np.testing.assert_array_equal(fast, ref)
    # vectorised interim rewards == reward_fn.interim(..., realization=ρ), bit for bit
    states = rng.integers(0, 4, n)
    months = rng.integers(0, 61, n)
    paid = np.minimum(months, rng.integers(0, 61, n))
    want = np.array(
        [
            env.reward_fn.interim(env.arm_space[int(a)], int(s), int(m), int(q), env.two_stage.risk(str(gg), int(a)), env.mctm,
                                  realization=env.realization(str(gg), int(a))).mean
            for gg, a, s, m, q in zip(g_arr, arms, states, months, paid, strict=True)
        ]
    )
    np.testing.assert_array_equal(env.interim_rewards(g_arr, arms, states, months, paid), want)
    scalar = np.array([env.interim_reward(gg, a, s, m, q) for gg, a, s, m, q in zip(g_arr, arms, states, months, paid, strict=True)])
    np.testing.assert_array_equal(scalar, want)
    # the yardstick and the warm start use the same formula
    mean, _std, _score = env.expected_table([groups[0]])
    mu, _sigma = env.prior_provider(groups[0])
    np.testing.assert_array_equal(mean[0], mu)
    # the table round-trips through a checkpoint snapshot
    copy = BayesianEnvironment.from_snapshot(env.snapshot(), env.reward_fn, env.arm_space)
    assert copy.interest_realization == env.interest_realization
    for a in (1, 70, 128):
        assert copy.expected(groups[0], a) == env.expected(groups[0], a)


def test_prepayment_switch_and_resegmentation():
    cfg, df, fitted = _data()
    off_cfg = FrameworkConfig.from_dict(cfg.to_dict())
    off_cfg.reward.prepayment_adjustment = None
    off = make_prepay_env(off_cfg)
    on = make_prepay_env()
    g = busiest_group()
    assert off.realization(g, 70) == 1.0 and on.realization(g, 70) < 1.0
    assert on.expected(g, 70).mean < off.expected(g, 70).mean  # less interest (net of funding) when prepaying
    finer = on.segmenter.refine()
    from bayesian_credit.models import TwoStageBetaBeta

    sub = df.loc[df["y_resolved"]]
    ts = TwoStageBetaBeta.from_config(FrameworkConfig(), on.arm_space).fit(sub.assign(y_group=finer.assign(sub).astype(str)))
    on.resegment(finer, ts)
    key = finer.assign(sub.head(1)).iloc[0]
    grade = str(sub.head(1)["grade"].iloc[0])
    assert on.realization(key, 70) == fitted.interest_realization.factor(grade, 60)
