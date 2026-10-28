"""DelayedFeedbackQueue: release order, columnar batches, round trips (SPEC §2.11)."""

import pickle

import numpy as np
import pytest

from bayesian_credit.bandits import ContextualThompsonSampling, Decision
from bayesian_credit.config import BanditConfig
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.simulation.feedback import (
    QUEUE_COLUMNS,
    DelayedFeedbackQueue,
    FeedbackItem,
    decision_payload,
)
from bayesian_credit.types import FeedbackKind, State


def item(release, loan_id, kind=FeedbackKind.TERMINAL, policy="thompson", arm=3, month=0, state=None, match=False):
    return FeedbackItem(
        release_epoch=release,
        kind=kind,
        policy_name=policy,
        loan_id=loan_id,
        decision=decision_payload(arm, "B|credit_card|RENT", policy, False, 0),
        group="B|credit_card|RENT",
        arm=arm,
        month=month,
        months_paid=month,
        state=state,
        is_historical_match=match,
        decision_epoch="2015-01",
    )


def test_item_round_trip_and_decision_payload():
    it = item("2015-04", 7, FeedbackKind.INTERIM, month=3, state=int(State.DELINQUENT), match=True)
    again = FeedbackItem.from_dict(it.to_dict())
    assert again == it
    assert again.kind is FeedbackKind.INTERIM and again.state == int(State.DELINQUENT)
    dec = Decision.from_dict(it.decision)
    assert (dec.arm, dec.group, dec.policy_name, dec.meta) == (3, "B|credit_card|RENT", "thompson", {"seg_version": 0})
    assert FeedbackItem.from_dict(item("2015-04", 1).to_dict()).state is None


def test_pop_due_releases_everything_due_in_epoch_then_insertion_order():
    q = DelayedFeedbackQueue()
    order = [("2015-05", 1), ("2015-03", 2), ("2015-05", 3), ("2015-01", 4), ("2015-03", 5), ("2016-01", 6)]
    for rel, lid in order:
        q.schedule(item(rel, lid))
    assert len(q) == 6 and q.next_release() == "2015-01"
    assert q.pop_due("2014-12") == []
    due = q.pop_due("2015-04")
    assert [(i.release_epoch, i.loan_id) for i in due] == [("2015-01", 4), ("2015-03", 2), ("2015-03", 5)]
    assert len(q) == 3
    assert [(i.release_epoch, i.loan_id) for i in q.pop_due("2015-05")] == [("2015-05", 1), ("2015-05", 3)]
    assert q.pop_due("2015-05") == []
    assert [i.loan_id for i in q.pop_due("2030-01")] == [6]
    assert len(q) == 0 and not q


def test_pending_counts_by_kind_and_policy():
    q = DelayedFeedbackQueue()
    q.schedule(item("2015-02", 1, FeedbackKind.INTERIM, policy="a", month=1, state=0))
    q.schedule(item("2015-03", 1, FeedbackKind.INTERIM, policy="a", month=2, state=0))
    q.schedule(item("2015-09", 1, FeedbackKind.TERMINAL, policy="a"))
    q.schedule(item("2015-09", 1, FeedbackKind.TERMINAL, policy="b"))
    assert q.pending_by_kind() == {"interim": 2, "terminal": 2}
    assert q.pending_by_policy() == {"a": 3, "b": 1}
    q.pop_due("2015-02")
    assert q.pending_by_kind() == {"interim": 1, "terminal": 2}
    assert q.pending_by_policy() == {"a": 2, "b": 1}
    q.pop_due("2015-12")
    assert q.pending_by_policy() == {} and q.pending_by_kind() == {"interim": 0, "terminal": 0}


def test_schedule_batch_groups_by_epoch_and_keeps_relative_order():
    q = DelayedFeedbackQueue()
    q.schedule(item("2015-03", 100))
    n = q.schedule_batch(
        np.array(["2015-03", "2015-02", "2015-03", "2015-02"], dtype=object),
        kind=np.array([1, 0, 1, 0], dtype=np.int8),
        policy="ucb1",
        loan_id=np.array([1, 2, 3, 4]),
        arm=np.array([5, 6, 7, 8]),
        group=np.array(["g1", "g2", "g3", "g4"], dtype=object),
        month=np.array([0, 1, 0, 2]),
        state=np.array([-1, 0, -1, 1]),
        decision_epoch="2015-01",
    )
    assert n == 4 and len(q) == 5
    q.schedule(item("2015-03", 200))
    batch = q.pop_due_batch("2015-03")
    assert batch["loan_id"].tolist() == [2, 4, 100, 1, 3, 200]
    assert batch.release_epoch.tolist() == ["2015-02", "2015-02", "2015-03", "2015-03", "2015-03", "2015-03"]
    items = batch.items()
    assert items[1].kind is FeedbackKind.INTERIM and items[1].state == 1 and items[1].month == 2
    assert items[3].kind is FeedbackKind.TERMINAL and items[3].state is None
    assert set(batch.columns) == set(QUEUE_COLUMNS)


def test_round_trip_to_dict_and_pickle_preserve_release_order():
    rng = np.random.default_rng(0)
    q = DelayedFeedbackQueue()
    for i in range(200):
        rel = f"2015-{int(rng.integers(1, 13)):02d}"
        kind = FeedbackKind.INTERIM if i % 3 else FeedbackKind.TERMINAL
        q.schedule(item(rel, i, kind, policy=f"p{i % 2}", arm=int(rng.integers(0, 129)), month=i % 5, state=0 if i % 3 else None))
    state = pickle.loads(pickle.dumps(q.to_dict()))
    copy = DelayedFeedbackQueue.from_dict(state)
    assert len(copy) == len(q) and copy.pending_by_kind() == q.pending_by_kind()
    assert copy.pending_by_policy() == q.pending_by_policy()
    for epoch in ("2015-03", "2015-07", "2015-12"):
        a, b = q.pop_due(epoch), copy.pop_due(epoch)
        assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert len(q) == len(copy) == 0


def test_to_dict_snapshot_is_independent_of_later_mutation():
    q = DelayedFeedbackQueue()
    q.schedule(item("2015-02", 1))
    snap = q.to_dict()
    q.pop_due("2015-02")
    q.schedule(item("2015-02", 2))
    restored = DelayedFeedbackQueue.from_dict(snap)
    assert [i.loan_id for i in restored.pop_due("2015-02")] == [1]


def test_replayed_decision_updates_a_policy(small_arm_space, segmenter):
    pol = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="flat"))
    ctx = BorrowerContext(1, {"grade": "B", "purpose": "credit_card", "home_ownership": "RENT", "annual_inc": 80_000.0})
    d = pol.select(ctx, None, np.random.default_rng(0))
    q = DelayedFeedbackQueue()
    q.schedule(
        FeedbackItem("2015-02", FeedbackKind.TERMINAL, pol.name, 1, decision_payload(d.arm, d.group, pol.name, False, d.meta["seg_version"]), d.group, d.arm)
    )
    (it,) = q.pop_due("2015-02")
    pol.update(Decision.from_dict(it.decision), 123.0, it.kind)
    assert pol.posterior(d.group, d.arm).n == 1.0 and pol.posterior(d.group, d.arm).mu == pytest.approx(123.0)
