"""CircuitBreaker, PSISentinel review queue and GovernedPolicy (§3.3, G.5, Table 9)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.bandits import ContextualThompsonSampling, Decision, GaussianThompsonSampling
from bayesian_credit.config import BanditConfig, PSIConfig
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.exceptions import ConfigurationError
from bayesian_credit.monitoring import (
    CircuitBreaker,
    GovernanceError,
    GovernedPolicy,
    PSICalculator,
    PSISentinel,
)
from bayesian_credit.types import BreakerState, ReviewAction, SentinelLevel

CLOSED, OPEN, HALF = BreakerState.CLOSED, BreakerState.OPEN, BreakerState.HALF_OPEN
N, W, B = SentinelLevel.NORMAL, SentinelLevel.WATCH, SentinelLevel.BREAKER
CFG = PSIConfig(features=("x", "grade"), recovery_readings=3)
CTX = BorrowerContext(1, {"grade": "B", "purpose": "credit_card", "home_ownership": "RENT"})
OTHER_CTX = BorrowerContext(2, {"grade": "E", "purpose": "car", "home_ownership": "OWN"})


def window(rng: np.random.Generator, n: int = 800, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({"x": rng.normal(shift, 1.0, n), "grade": rng.choice(list("ABC"), n)})


@pytest.fixture
def sentinel() -> PSISentinel:
    return PSISentinel.from_reference(window(np.random.default_rng(0), 4000), CFG)


def test_breaker_state_machine_transitions_exactly():
    br = CircuitBreaker(CFG)
    seen = []
    br.on_transition.append(seen.append)
    assert br.observe(N, "e1") is None and br.observe(W, "e2") is None  # WATCH is a flag
    assert br.state == CLOSED
    assert br.observe(B, "e3").to_state == OPEN
    for i, level in enumerate([N, N, N, W, B]):  # readings never close an OPEN breaker
        assert br.observe(level, f"o{i}") is None and br.state == OPEN
    assert br.acknowledge("e4").to_state == HALF
    for level in (N, N, W, N, N):  # WATCH resets the recovery counter
        assert br.observe(level, "h") is None and br.state == HALF
    assert br.observe(N, "e5").to_state == CLOSED  # third consecutive NORMAL
    assert br.observe(B, "e6").to_state == OPEN
    br.acknowledge("e7")
    retrip = br.observe(B, "e8")  # BREAKER while HALF_OPEN re-opens
    assert (retrip.from_state, retrip.to_state, retrip.reason) == (HALF, OPEN, "psi_breaker_half_open")
    expected = [
        ("e3", CLOSED, OPEN, "psi_breaker"),
        ("e4", OPEN, HALF, "acknowledged"),
        ("e5", HALF, CLOSED, "recovered"),
        ("e6", CLOSED, OPEN, "psi_breaker"),
        ("e7", OPEN, HALF, "acknowledged"),
        ("e8", HALF, OPEN, "psi_breaker_half_open"),
    ]
    assert [(t.epoch, t.from_state, t.to_state, t.reason) for t in br.transitions] == expected
    assert seen == br.transitions
    again = CircuitBreaker.from_dict(json.loads(json.dumps(br.to_dict(), allow_nan=False)))
    assert again.state == OPEN and again.transitions == br.transitions


def test_acknowledge_requires_open_breaker():
    br = CircuitBreaker(CFG)
    with pytest.raises(GovernanceError):
        br.acknowledge("e1")
    instant = CircuitBreaker(PSIConfig(recovery_readings=0))
    instant.observe(B, "e1")
    instant.acknowledge("e2")
    assert instant.state == CLOSED


def test_table9_level_bands(sentinel):
    assert sentinel.classify(0.0999) == N
    assert sentinel.classify(0.10) == W
    assert sentinel.classify(0.25) == W  # Table 9: 0.10-0.25 watch; > 0.25 breaker
    assert sentinel.classify(0.2501) == B
    assert sentinel.classify(float("nan")) == N


def test_trip_opens_review_and_acknowledge_half_opens(sentinel):
    rng = np.random.default_rng(1)
    r1 = sentinel.evaluate(window(rng), "2016-01")
    assert (r1.level, r1.tripped, r1.review_id, r1.breaker_after) == (N, False, None, CLOSED)
    r2 = sentinel.evaluate(window(rng, shift=1.5), "2016-02")
    assert r2.level == B and r2.tripped and r2.breaker_before == CLOSED and r2.breaker_after == OPEN
    assert [r.id for r in sentinel.pending_reviews] == [r2.review_id] == ["review-0001"]
    review = sentinel.pending_reviews[0]
    assert review.status == "pending" and review.action is None
    assert review.psi_aggregate == pytest.approx(r2.psi_aggregate)
    assert sentinel.evaluate(window(rng, shift=1.5), "2016-03").tripped is False  # still OPEN
    acked = sentinel.acknowledge("review-0001", "resume", note="checked")
    assert (acked.status, acked.action, acked.note, acked.acknowledged_epoch) == (
        "acknowledged",
        ReviewAction.RESUME,
        "checked",
        "2016-03",
    )
    assert sentinel.state == HALF and sentinel.pending_reviews == []
    with pytest.raises(GovernanceError):
        sentinel.acknowledge("review-0001", "resume")
    with pytest.raises(GovernanceError):
        sentinel.acknowledge("review-9999", "resume")
    for i in range(3):
        sentinel.evaluate(window(rng), f"2016-0{4 + i}")
    assert sentinel.state == CLOSED
    reasons = [t.reason for t in sentinel.transitions]
    assert reasons == ["psi_breaker", "acknowledged:resume", "recovered"]
    hist = sentinel.history_frame()
    assert len(hist) == 6 and {"psi_x", "psi_grade", "level", "breaker_after"} <= set(hist.columns)


def test_skipped_windows_leave_breaker_untouched(sentinel):
    rng = np.random.default_rng(2)
    sentinel.evaluate(window(rng, shift=2.0), "e1")
    review = sentinel.pending_reviews[0]
    sentinel.acknowledge(review.id, ReviewAction.RESUME)
    sentinel.evaluate(window(rng), "e2")
    empty = sentinel.evaluate(window(rng).iloc[:0], "e3")
    assert empty.skipped and math.isnan(empty.psi_aggregate) and empty.level == N
    sentinel.evaluate(window(rng), "e4")
    assert sentinel.state == HALF  # the empty window did not count as a NORMAL reading
    sentinel.evaluate(window(rng), "e5")
    assert sentinel.state == CLOSED
    small = PSISentinel(sentinel.calculator, CFG, min_window_size=100)
    assert small.evaluate(window(rng, n=50, shift=3.0), "e6").skipped and small.state == CLOSED


def test_monitor_only_never_trips(sentinel):
    mon = PSISentinel(sentinel.calculator, CFG, monitor_only=True)
    reading = mon.evaluate(window(np.random.default_rng(3), shift=2.0), "e1")
    assert reading.level == B and not reading.tripped and mon.state == CLOSED and mon.reviews == []


def test_sentinel_checkpoint_roundtrip(sentinel):
    rng = np.random.default_rng(4)
    sentinel.evaluate(window(rng), "e1")
    sentinel.evaluate(window(rng, shift=2.0), "e2")
    payload = json.dumps(sentinel.to_dict(), allow_nan=False)
    clone = PSISentinel.from_dict(json.loads(payload))
    assert clone.state == OPEN and [r.to_dict() for r in clone.reviews] == [r.to_dict() for r in sentinel.reviews]
    assert [h.to_dict() for h in clone.history] == [h.to_dict() for h in sentinel.history]
    clone.acknowledge("review-0001", "resegment")
    sentinel.acknowledge("review-0001", "resegment")
    w = window(np.random.default_rng(5), shift=2.5)
    a, b = sentinel.evaluate(w, "e3"), clone.evaluate(w, "e3")
    assert a.to_dict() == b.to_dict() and a.review_id == "review-0002"
    assert isinstance(clone.calculator, PSICalculator)


# ----------------------------------------------------------------------------------------
# GovernedPolicy
# ----------------------------------------------------------------------------------------
def trained_contextual(arm_space, segmenter, seed=6):
    pol = ContextualThompsonSampling(arm_space, segmenter, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(seed)
    means = np.array([0.0, 10.0, 30.0, 20.0, 5.0])
    for _ in range(150):
        d = pol.select(CTX, None, rng)
        pol.update(d, float(rng.normal(means[d.arm], 1.0)))
    return pol


def test_champion_snapshot_fallback_freezes_and_marks_decisions(sentinel, small_arm_space, segmenter):
    pol = trained_contextual(small_arm_space, segmenter)
    gov = GovernedPolicy(pol, sentinel, small_arm_space)
    assert gov.fallback == "champion_snapshot" and not gov.frozen
    rng = np.random.default_rng(7)
    assert gov.select(CTX, None, rng).fallback is False
    sentinel.evaluate(window(rng, shift=2.0), "2017-01")
    assert pol.frozen and gov.frozen and gov.champion_epoch == "2017-01"
    means = pol.means_from_snapshot(gov.champion_snapshot)[pol.group_for(CTX)]
    d = gov.select(CTX, None, rng)
    assert d.fallback and d.arm == int(np.argmax(means)) and d.meta["fallback_mode"] == "champion_snapshot"
    np.testing.assert_array_equal(d.scores, means)
    mask = np.ones(small_arm_space.n_arms, dtype=bool)
    mask[int(np.argmax(means))] = False
    d2 = gov.select(CTX, mask, rng)
    masked = np.where(mask, means, -np.inf)
    assert d2.fallback and d2.arm == int(np.argmax(masked)) and np.isnan(d2.scores[~mask]).all()
    before = json.dumps(pol.snapshot())
    gov.update(d, 1000.0)
    gov.update(d2, 1000.0, kind="interim")
    assert gov.dropped_updates == 2 and json.dumps(pol.snapshot()) == before
    unseen = gov.select(OTHER_CTX, None, rng)  # group unknown to the champion: pooled means
    pooled = pol.means_from_snapshot(gov.champion_snapshot)["*"]
    assert unseen.fallback and unseen.arm == int(np.argmax(pooled))
    assert gov.n_fallback == 3


@pytest.mark.parametrize("mode", ["decline", "historical"])
def test_decline_and_historical_fallbacks(mode, sentinel, small_arm_space):
    pol = GaussianThompsonSampling(small_arm_space, cfg=BanditConfig(warm_start="flat"))
    gov = GovernedPolicy(pol, sentinel, small_arm_space, fallback=mode)
    rng = np.random.default_rng(8)
    sentinel.evaluate(window(rng, shift=2.0), "e1")
    d = gov.select(CTX, None, rng, historical_arm=3)
    assert d.fallback and d.meta["fallback_mode"] == mode
    assert d.arm == (0 if mode == "decline" else 3)
    mask = np.array([True, True, True, False, True])
    assert gov.select(CTX, mask, rng, historical_arm=3).arm == 0  # ineligible historical -> decline
    assert gov.select(CTX, None, rng, historical_arm=None).arm == 0


def test_acknowledge_unfreezes_and_fires_hooks(sentinel, small_arm_space, segmenter):
    pol = trained_contextual(small_arm_space, segmenter)
    gov = GovernedPolicy(pol, sentinel, small_arm_space)
    calls = []
    gov.on_acknowledge.append(calls.append)
    rng = np.random.default_rng(9)
    review_id = sentinel.evaluate(window(rng, shift=2.0), "e1").review_id
    assert gov.frozen
    sentinel.acknowledge(review_id, ReviewAction.RESEGMENT)
    assert calls == [ReviewAction.RESEGMENT]
    assert not gov.frozen and gov.breaker_state == HALF
    d = gov.select(CTX, None, rng)
    assert d.fallback is False and d.meta["breaker_state"] == "half_open"
    n_before = pol.n_updates
    gov.update(d, 5.0)
    assert pol.n_updates == n_before + 1


def test_rollback_restores_the_incident_champion(sentinel, small_arm_space, segmenter):
    pol = trained_contextual(small_arm_space, segmenter)
    gov = GovernedPolicy(pol, sentinel, small_arm_space)
    hooks = []
    gov.on_acknowledge.append(hooks.append)
    rng = np.random.default_rng(10)
    first = sentinel.evaluate(window(rng, shift=2.0), "e1").review_id
    champion = json.loads(json.dumps(gov.champion_snapshot))
    sentinel.acknowledge(first, "resume")
    for _ in range(40):  # learning resumes in HALF_OPEN ...
        d = gov.select(CTX, None, rng)
        gov.update(d, -500.0)
    assert pol.snapshot()["groups"] != champion["groups"]
    second = sentinel.evaluate(window(rng, shift=2.0), "e2")  # ... and the shift persists
    assert second.tripped and second.breaker_before == HALF
    assert gov.champion_epoch == "e1"  # the re-trip keeps the incident's champion
    sentinel.acknowledge(second.review_id, ReviewAction.ROLLBACK)
    assert hooks == [ReviewAction.RESUME, ReviewAction.ROLLBACK]
    restored = pol.snapshot()
    assert restored["groups"] == champion["groups"] and restored["pooled"] == champion["pooled"]
    assert not gov.frozen


def test_governed_state_roundtrip_while_open(sentinel, small_arm_space, segmenter):
    pol = trained_contextual(small_arm_space, segmenter)
    gov = GovernedPolicy(pol, sentinel, small_arm_space)
    rng = np.random.default_rng(11)
    sentinel.evaluate(window(rng, shift=2.0), "e1")
    gov.update(Decision(arm=1, group=pol.group_for(CTX), policy_name=pol.name), 3.0)
    state = json.loads(json.dumps(gov.state(), allow_nan=False))
    sentinel2 = PSISentinel.from_dict(json.loads(json.dumps(sentinel.to_dict(), allow_nan=False)))
    gov2 = GovernedPolicy.from_state(state, small_arm_space, sentinel=sentinel2)
    assert gov2.frozen and gov2.breaker_state == OPEN and gov2.dropped_updates == 1
    assert gov2.champion_epoch == "e1" and gov2.champion_snapshot == gov.champion_snapshot
    d1, d2 = gov.select(CTX, None, rng), gov2.select(CTX, None, rng)
    assert d1 == d2 and d1.fallback
    sentinel2.acknowledge("review-0001", "resume")
    assert not gov2.frozen
    with pytest.raises(ConfigurationError):  # the sentinel is mandatory
        GovernedPolicy.from_state(state, small_arm_space)


def test_manual_freeze_combines_with_breaker(sentinel, small_arm_space):
    pol = GaussianThompsonSampling(small_arm_space)
    gov = GovernedPolicy(pol, sentinel, small_arm_space)
    gov.freeze()
    assert pol.frozen
    gov.unfreeze()
    assert not pol.frozen
    gov.detach()
    sentinel.evaluate(window(np.random.default_rng(12), shift=2.0), "e1")
    assert gov.select(CTX, None, np.random.default_rng(13)).fallback  # synced lazily after detach
