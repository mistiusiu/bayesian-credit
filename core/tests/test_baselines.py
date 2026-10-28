"""Baseline policies: same protocol, eligibility respected, documented behaviour."""

import json

import numpy as np
import pytest

from bayesian_credit.bandits import (
    POLICY_NAMES,
    UCB1,
    ContextualThompsonSampling,
    Decision,
    EpsilonGreedy,
    GaussianThompsonSampling,
    Greedy,
    HistoricalPolicy,
    RandomPolicy,
    build_policy,
    policy_from_state,
)
from bayesian_credit.config import BanditConfig
from bayesian_credit.data.arms import ArmSpace, LendingPolicy
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.exceptions import ConfigurationError

CTX = BorrowerContext(
    loan_id=1, features={"grade": "B", "purpose": "credit_card", "home_ownership": "RENT", "annual_inc": 60_000.0}
)
FLAT = BanditConfig(warm_start="flat")


def test_historical_policy_returns_the_historical_arm(small_arm_space):
    pol = HistoricalPolicy(small_arm_space, cfg=FLAT)
    rng = np.random.default_rng(0)
    assert pol.select(CTX, None, rng, historical_arm=3).arm == 3
    mask = np.array([True, True, True, False, True])
    d = pol.select(CTX, mask, rng, historical_arm=3)
    assert d.arm == 0 and d.meta["historical_missing"] is False  # ineligible -> decline
    for missing in (None, -1, float("nan"), 99):
        assert pol.select(CTX, None, rng, historical_arm=missing).arm == 0
    before = [pol.select(CTX, None, rng, historical_arm=2).arm for _ in range(5)]
    for r in (-100.0, -200.0, -300.0):
        pol.update(Decision(arm=2, group="*", policy_name="historical"), r)
    after = [pol.select(CTX, None, rng, historical_arm=2).arm for _ in range(5)]
    assert before == after == [2] * 5  # learning never changes behaviour
    assert pol.posterior("*", 2).n == 3.0  # ...but rewards are recorded


def test_random_policy_never_declines_when_it_can_lend(arm_space):
    pol = RandomPolicy(arm_space, cfg=FLAT)
    rng = np.random.default_rng(1)
    mask = arm_space.eligible(45_000.0)
    picks = np.array([pol.select(CTX, mask, rng).arm for _ in range(3000)])
    assert not np.any(picks == 0)
    assert np.all(mask[picks])
    lend = np.flatnonzero(mask)[1:]
    counts = np.bincount(picks, minlength=arm_space.n_arms)[lend]
    assert counts.min() > 0 and counts.max() / counts.min() < 3  # roughly uniform
    only_null = np.zeros(arm_space.n_arms, dtype=bool)
    assert pol.select(CTX, only_null, rng).arm == 0


def test_ucb_tries_every_eligible_arm_first():
    space = ArmSpace.from_grid([5_000, 10_000, 15_000], [0.10, 0.20], [36])  # 7 arms incl. null
    mask = np.ones(space.n_arms, dtype=bool)
    mask[[2, 5]] = False
    pol = UCB1(space, cfg=FLAT)
    rng = np.random.default_rng(2)
    eligible = [0, 1, 3, 4, 6]
    true = {0: 0.0, 1: 1.0, 3: 5.0, 4: 2.0, 6: 3.0}
    first = []
    for _ in range(len(eligible)):
        d = pol.select(CTX, mask, rng)
        assert d.meta["untried"] is True
        first.append(d.arm)
        pol.update(d, true[d.arm])
    assert first == eligible  # each eligible arm once, lowest index first
    later = []
    for _ in range(200):
        d = pol.select(CTX, mask, rng)
        assert mask[d.arm] and d.meta["untried"] is False
        pol.update(d, true[d.arm] + float(rng.normal(0, 0.1)))
        later.append(d.arm)
    assert np.mean(np.array(later[-50:]) == 3) > 0.8


def test_epsilon_greedy_explores_at_rate_epsilon(small_arm_space):
    pol = EpsilonGreedy(small_arm_space, cfg=FLAT, epsilon=0.2)
    for arm, r in [(1, 1.0), (2, 9.0), (3, 2.0), (4, 3.0)]:
        pol.update(Decision(arm=arm, group="*", policy_name="epsilon_greedy"), r)
    rng = np.random.default_rng(3)
    decisions = [pol.select(CTX, None, rng) for _ in range(2000)]
    explore = np.array([d.meta["explore"] for d in decisions])
    assert explore.mean() == pytest.approx(0.2, abs=0.03)
    assert all(d.arm == 2 for d, e in zip(decisions, explore, strict=True) if not e)
    assert decisions[0].scores[2] == 9.0
    greedy = EpsilonGreedy(small_arm_space, cfg=FLAT, epsilon=0.0)
    greedy.restore(pol.snapshot())
    assert {greedy.select(CTX, None, rng).arm for _ in range(50)} == {2}
    with pytest.raises(ConfigurationError):
        EpsilonGreedy(small_arm_space, epsilon=1.5)


def test_greedy_breaks_ties_at_lowest_index_and_exploits():
    space = ArmSpace([LendingPolicy(1_000, 0.1, 36), LendingPolicy(2_000, 0.1, 36), LendingPolicy(3_000, 0.1, 36)])
    pol = Greedy(space, cfg=FLAT)
    rng = np.random.default_rng(4)
    assert pol.select(CTX, np.array([False, True, True]), rng).arm == 1
    pol.update(Decision(arm=2, group="*", policy_name="greedy"), 4.0)
    assert pol.select(CTX, None, rng).arm == 2
    assert pol.select(CTX, np.array([True, True, False]), rng).arm == 0


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_every_policy_respects_eligibility(name, arm_space, segmenter):
    pol = build_policy(name, arm_space, segmenter, FLAT)
    rng = np.random.default_rng(5)
    for i in range(250):
        mask = rng.random(arm_space.n_arms) < 0.25
        hist = int(rng.integers(arm_space.n_arms))
        d = pol.select(CTX, mask, rng, historical_arm=hist)
        mask[0] = True
        assert mask[d.arm], f"{name} picked ineligible arm {d.arm}"
        assert d.policy_name == name
        pol.update(d, float(rng.normal(10.0 * (d.arm % 5), 3.0)), kind="interim" if i % 3 else "terminal")


def test_build_policy_factory(arm_space, segmenter):
    assert isinstance(build_policy("thompson", arm_space, segmenter), ContextualThompsonSampling)
    assert isinstance(build_policy("thompson", arm_space, None), GaussianThompsonSampling)
    assert isinstance(build_policy("thompson_flat", arm_space, segmenter), GaussianThompsonSampling)
    eg = build_policy("epsilon_greedy", arm_space, segmenter, BanditConfig(epsilon=0.3))
    assert isinstance(eg, EpsilonGreedy) and eg.epsilon == 0.3 and eg.contextual
    ucb = build_policy("UCB1", arm_space, None, BanditConfig(ucb_c=0.5))
    assert isinstance(ucb, UCB1) and ucb.c == 0.5 and not ucb.contextual
    assert isinstance(build_policy("greedy", arm_space, segmenter), Greedy)
    assert isinstance(build_policy("random", arm_space, segmenter), RandomPolicy)
    assert isinstance(build_policy("historical", arm_space, segmenter), HistoricalPolicy)
    with pytest.raises(ConfigurationError):
        build_policy("oracle", arm_space, segmenter)


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_policy_state_roundtrip_reproduces_decisions(name, small_arm_space, segmenter):
    pol = build_policy(name, small_arm_space, segmenter, BanditConfig(warm_start="global"))
    rng = np.random.default_rng(6)
    for _ in range(40):
        d = pol.select(CTX, None, rng, historical_arm=2)
        pol.update(d, float(rng.normal(d.arm, 1.0)))
    clone = policy_from_state(json.loads(json.dumps(pol.state(), allow_nan=False)), small_arm_space)
    assert type(clone) is type(pol) and clone.name == pol.name
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(40):
        d1 = pol.select(CTX, None, r1, historical_arm=4)
        d2 = clone.select(CTX, None, r2, historical_arm=4)
        assert d1 == d2
        pol.update(d1, 2.0)
        clone.update(d2, 2.0)


def test_baselines_share_freeze_and_snapshot_machinery(small_arm_space, segmenter):
    pol = UCB1(small_arm_space, segmenter, cfg=FLAT, c=1.0)
    rng = np.random.default_rng(8)
    d = pol.select(CTX, None, rng)
    pol.update(d, 1.0)
    snap = pol.snapshot()
    pol.freeze()
    pol.update(d, 5.0)
    pol.update(d, 6.0)
    assert pol.dropped_updates == 2
    assert pol.snapshot() == snap
    pol.unfreeze()
    pol.update(d, 7.0)
    pol.restore(snap)
    assert pol.snapshot() == snap and pol.dropped_updates == 2
    assert pol.arm_table("*")["selections"].sum() == 1  # pooled parent of a contextual baseline
    assert pol.group_keys() == [pol.group_for(CTX)]
