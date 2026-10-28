"""Flat Gaussian Thompson Sampling and the shared Policy protocol behaviour."""

import json

import numpy as np
import pytest

from bayesian_credit.bandits import Decision, GaussianThompsonSampling, policy_from_state
from bayesian_credit.config import BanditConfig
from bayesian_credit.data.arms import ArmSpace, LendingPolicy
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.exceptions import ConfigurationError, NumericalInstabilityError
from bayesian_credit.types import FeedbackKind

CTX = BorrowerContext(loan_id=0, features={"grade": "B", "annual_inc": 60_000.0})


def three_arm_space() -> ArmSpace:
    return ArmSpace(
        [LendingPolicy(1_000, 0.10, 36), LendingPolicy(2_000, 0.10, 36), LendingPolicy(3_000, 0.10, 36)]
    )


def train(policy, steps: int, means: np.ndarray, seed: int = 0, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    env = np.random.default_rng(seed + 1)
    arms = []
    for _ in range(steps):
        d = policy.select(CTX, None, rng)
        policy.update(d, float(env.normal(means[d.arm], noise)))
        arms.append(d.arm)
    return np.asarray(arms)


@pytest.mark.parametrize("scale", ["proposal", "standard_error"])
def test_thompson_sampling_converges_to_best_arm(scale):
    ts = GaussianThompsonSampling(three_arm_space(), cfg=BanditConfig(warm_start="flat", posterior_scale=scale))
    arms = train(ts, 3000, np.array([1.0, 2.0, 5.0]), seed=2024)
    assert np.mean(arms[-500:] == 2) > 0.8
    table = ts.arm_table()
    assert list(table.columns) == ["group", "arm", "mu", "sigma", "n", "selections"]
    assert table["selections"].sum() == 3000
    assert table.loc[2, "mu"] == pytest.approx(5.0, abs=0.1)


def test_ineligible_arms_are_never_chosen(arm_space):
    ts = GaussianThompsonSampling(arm_space, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(0)
    for i in range(400):
        mask = rng.random(arm_space.n_arms) < 0.3
        d = ts.select(CTX, mask, rng)
        expected = mask.copy()
        expected[0] = True  # the null arm is always eligible
        assert expected[d.arm]
        assert np.all(np.isnan(d.scores[~expected]))
        assert np.all(np.isfinite(d.scores[expected]))
        ts.update(d, float(rng.normal(d.arm, 5.0)), weight=0.5 if i % 2 else None)


def test_null_arm_forced_eligible_and_mask_validated(small_arm_space):
    ts = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(1)
    d = ts.select(CTX, np.zeros(small_arm_space.n_arms, dtype=bool), rng)
    assert d.arm == 0
    with pytest.raises(ValueError):
        ts.select(CTX, np.ones(3, dtype=bool), rng)


def test_pinned_null_arm_declines_when_lending_loses(small_arm_space):
    ts = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(3)
    for arm in range(1, small_arm_space.n_arms):
        for r in (-500.0, -520.0, -480.0):
            ts.update(Decision(arm=arm, group="*", policy_name="x"), r)
    picks = [ts.select(CTX, None, rng).arm for _ in range(200)]
    assert np.mean(np.array(picks) == 0) > 0.95
    null_post = ts.posterior("*", 0)
    assert null_post.mu == 0.0 and null_post.sigma2 == pytest.approx(BanditConfig().min_sigma**2)


def test_frozen_policy_drops_and_counts_updates(small_arm_space):
    ts = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(4)
    d = ts.select(CTX, None, rng)
    ts.update(d, 10.0)
    snap = json.dumps(ts.snapshot())
    ts.freeze()
    assert ts.frozen
    for r in (1.0, 2.0, 3.0):
        ts.update(d, r)
    ts.update(d, 4.0, kind=FeedbackKind.INTERIM, weight=0.0)  # weight 0 is not a "drop"
    assert ts.dropped_updates == 3
    assert json.dumps(ts.snapshot()) == snap
    ts.unfreeze()
    ts.update(d, 20.0)
    assert ts.posterior("*", d.arm).n == 2.0
    assert ts.dropped_updates == 3


def test_feedback_kinds_and_weights(small_arm_space):
    cfg = BanditConfig(warm_start="flat", interim_weight=0.25)
    ts = GaussianThompsonSampling(small_arm_space, cfg=cfg)
    d = Decision(arm=2, group="*", policy_name="thompson")
    ts.update(d, 5.0, kind="interim")
    assert ts.posterior("*", 2).n == 0.25
    ts.update(d, 5.0, kind=FeedbackKind.TERMINAL)
    assert ts.posterior("*", 2).n == 1.25
    ts.update(d, 5.0, kind="interim", weight=0.75)
    assert ts.posterior("*", 2).n == 2.0
    ignore = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat", interim_weight=0.0))
    ignore.update(d, 5.0, kind="interim")
    assert ignore.n_updates == 0
    with pytest.raises(ConfigurationError):
        ts.update(d, 1.0, kind="weekly")
    with pytest.raises(NumericalInstabilityError):
        ts.update(d, float("nan"))
    with pytest.raises(ValueError):
        ts.update(Decision(arm=99, group="*", policy_name="thompson"), 1.0)


def test_snapshot_restore_roundtrip_is_deep(small_arm_space):
    ts = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat"))
    train(ts, 50, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), seed=5)
    snap = ts.snapshot()
    frozen_copy = json.loads(json.dumps(snap, allow_nan=False))
    table_before = ts.arm_table()
    train(ts, 50, np.array([4.0, 3.0, 2.0, 1.0, 0.0]), seed=6)
    assert snap == frozen_copy  # later learning does not leak into the snapshot
    assert not ts.arm_table().equals(table_before)
    ts.restore(frozen_copy)
    assert ts.arm_table().equals(table_before)
    assert ts.n_decisions == 50 and ts.n_updates == 50


def test_from_state_reproduces_future_selections(small_arm_space):
    means = np.array([0.0, 3.0, 1.0, 2.5, 2.9])
    ts = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat", posterior_scale="standard_error"))
    train(ts, 120, means, seed=8)
    ts.freeze()
    ts.update(Decision(arm=1, group="*", policy_name="thompson"), 1.0)
    ts.unfreeze()
    state = json.loads(json.dumps(ts.state(), allow_nan=False))
    clone = GaussianThompsonSampling.from_state(state, small_arm_space)
    assert clone.dropped_updates == 1 and clone.cfg == ts.cfg and clone.name == ts.name
    assert isinstance(policy_from_state(state, small_arm_space), GaussianThompsonSampling)
    r1, r2 = np.random.default_rng(99), np.random.default_rng(99)
    env = np.random.default_rng(100)
    for _ in range(200):
        d1 = ts.select(CTX, None, r1)
        d2 = clone.select(CTX, None, r2)
        assert d1 == d2
        r = float(env.normal(means[d1.arm], 1.0))
        ts.update(d1, r)
        clone.update(d2, r)
    assert ts.state() == clone.state()


def test_from_state_rejects_other_arm_space(small_arm_space, arm_space):
    ts = GaussianThompsonSampling(small_arm_space)
    with pytest.raises(ConfigurationError):
        GaussianThompsonSampling.from_state(ts.state(), arm_space)


def test_flat_policy_uses_provider_for_pooled_group(small_arm_space):
    calls = []

    def provider(group):
        calls.append(group)
        k = small_arm_space.n_arms
        return np.arange(k, dtype=float) * 10.0, np.full(k, 3.0)

    ts = GaussianThompsonSampling(small_arm_space, prior_provider=provider)
    d = ts.select(CTX, None, np.random.default_rng(0))
    assert calls == ["*"]
    assert "warm_start_fallback" not in d.meta
    table = ts.arm_table("*")
    assert table["mu"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert table["sigma"].iloc[1:].tolist() == pytest.approx([3.0] * 4)
    assert GaussianThompsonSampling(small_arm_space).select(CTX, None, np.random.default_rng(0)).meta[
        "warm_start_fallback"
    ]


def test_flat_policy_rejects_segmenter(small_arm_space, segmenter):
    with pytest.raises(TypeError):
        GaussianThompsonSampling(small_arm_space, segmenter)  # type: ignore[misc]


def test_decision_serialisation_roundtrip(small_arm_space):
    ts = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat"))
    mask = np.array([True, True, False, True, False])
    d = ts.select(CTX, mask, np.random.default_rng(0))
    payload = json.dumps(d.to_dict(), allow_nan=False)
    again = Decision.from_dict(json.loads(payload))
    assert again == d
    assert np.isnan(again.scores[2])
    assert Decision.from_dict(d.to_dict(include_scores=False)).scores is None
