"""Contextual Thompson Sampling: lazy groups, hierarchical warm start, pooled parent."""

import json

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.bandits import (
    ContextualThompsonSampling,
    Decision,
    GaussianThompsonSampling,
    GroupKeyResolver,
    policy_from_state,
)
from bayesian_credit.config import BanditConfig
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.exceptions import ConfigurationError


def ctx(grade="B", purpose="credit_card", home="RENT", loan_id=0, **extra) -> BorrowerContext:
    feats = {"grade": grade, "purpose": purpose, "home_ownership": home, "annual_inc": 70_000.0}
    feats.update(extra)
    return BorrowerContext(loan_id=loan_id, features=feats)


A = ctx("A", "credit_card", "RENT")
B = ctx("C", "debt_consolidation", "MORTGAGE")
KEY_A = "A|credit_card|RENT"
KEY_B = "C|debt_consolidation|MORTGAGE"


def test_groups_are_created_lazily(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(0)
    assert cts.group_keys() == []
    d1 = cts.select(A, None, rng)
    assert d1.group == KEY_A and d1.meta["group_new"] is True and d1.meta["n_group_obs"] == 0
    cts.update(d1, 5.0)
    d2 = cts.select(A, None, rng)
    assert d2.meta["group_new"] is False and d2.meta["n_group_obs"] == 1
    assert cts.group_keys() == [KEY_A]
    d3 = cts.select(B, None, rng)
    assert d3.meta["group_new"] is True
    assert cts.group_keys() == sorted([KEY_A, KEY_B])
    assert cts.group_for(A) == segmenter.key_for(A)


def test_risk_model_warm_start_copies_provider_priors(small_arm_space, segmenter):
    calls = []
    k = small_arm_space.n_arms

    def provider(group):
        calls.append(group)
        base = 100.0 if group == KEY_A else -50.0
        return base + np.arange(k, dtype=float), np.full(k, 7.0)

    cts = ContextualThompsonSampling(small_arm_space, segmenter, prior_provider=provider)
    rng = np.random.default_rng(1)
    d = cts.select(A, None, rng)
    cts.select(A, None, rng)
    cts.select(B, None, rng)
    assert calls == [KEY_A, KEY_B]  # once per group, on first visit
    assert "warm_start_fallback" not in d.meta
    table_a = cts.arm_table(KEY_A)
    assert table_a["mu"].tolist() == [0.0, 101.0, 102.0, 103.0, 104.0]  # null arm pinned to 0
    assert table_a["sigma"].iloc[1:].tolist() == pytest.approx([7.0] * (k - 1))
    assert cts.arm_table(KEY_B)["mu"].iloc[1:].tolist() == [-49.0, -48.0, -47.0, -46.0]


def test_provider_shape_is_validated(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(
        small_arm_space, segmenter, prior_provider=lambda g: (np.zeros(2), np.ones(2))
    )
    with pytest.raises(ConfigurationError):
        cts.select(A, None, np.random.default_rng(0))


def test_risk_model_without_provider_falls_back_to_global(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter)  # warm_start="risk_model" default
    assert cts.warm_start_fallback
    rng = np.random.default_rng(2)
    for r in (10.0, 12.0, 14.0):
        d = cts.select(A, None, rng)
        assert d.meta["warm_start_fallback"] is True
        cts.update(Decision(arm=3, group=d.group, policy_name=cts.name, meta=d.meta), r)
    pooled = cts.arm_table("*")
    d_b = cts.select(B, None, rng)
    assert d_b.meta["group_new"] and d_b.meta["warm_start_fallback"]
    table_b = cts.arm_table(KEY_B)
    assert table_b["mu"].tolist() == pooled["mu"].tolist()  # copied from the pooled parent
    assert table_b.loc[3, "sigma"] == pytest.approx(pooled.loc[3, "sigma"])
    assert table_b["n"].sum() == 0.0  # but carries no pseudo-count


def test_flat_warm_start(small_arm_space, segmenter):
    cfg = BanditConfig(warm_start="flat", prior_mu=3.0, prior_sigma=50.0)
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=cfg)
    cts.select(A, None, np.random.default_rng(0))
    t = cts.arm_table(KEY_A)
    assert t["mu"].iloc[1:].tolist() == [3.0] * 4 and t["sigma"].iloc[1:].tolist() == [50.0] * 4


def test_pooled_parent_receives_every_reward(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="flat"))
    assert isinstance(cts.pooled, GaussianThompsonSampling)
    rng = np.random.default_rng(3)
    for i in range(60):
        d = cts.select(A if i % 3 else B, None, rng)
        cts.update(d, float(rng.normal(d.arm * 2.0, 1.0)), weight=1.0 if i % 2 else 0.5)
    pooled = cts.arm_table("*")
    groups = cts.arm_table()
    assert pooled["selections"].sum() == groups["selections"].sum() == 60
    assert pooled["n"].sum() == pytest.approx(groups["n"].sum())
    assert set(groups["group"]) == {KEY_A, KEY_B}
    assert cts.arm_table("nope").empty


def test_contextual_state_roundtrip_reproduces_decisions(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="global"))
    rng = np.random.default_rng(4)
    contexts = [A, B, ctx("B", "car", "OWN")]
    for i in range(90):
        d = cts.select(contexts[i % 3], None, rng)
        cts.update(d, float(rng.normal(d.arm, 1.0)))
    state = json.loads(json.dumps(cts.state(), allow_nan=False))
    clone = policy_from_state(state, small_arm_space)  # segmenter restored from the state
    assert isinstance(clone, ContextualThompsonSampling)
    assert clone.segmenter == segmenter
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for i in range(60):
        d1 = cts.select(contexts[i % 3], None, r1)
        d2 = clone.select(contexts[i % 3], None, r2)
        assert d1 == d2
        cts.update(d1, 1.0)
        clone.update(d2, 1.0)
    assert cts.state() == clone.state()


def test_simpsons_check_flags_reversals(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="flat"))
    feed = {KEY_A: {1: 10.0, 2: 5.0}, KEY_B: {1: -20.0, 2: 4.0}}
    for group, rewards in feed.items():
        for arm, r in rewards.items():
            for _ in range(5):
                cts.update(Decision(arm=arm, group=group, policy_name="thompson"), r)
    check = cts.simpsons_check().set_index("group")
    assert check.loc[KEY_A, "pooled_best_arm"] == 2  # pooled: arm1 = -5, arm2 = 4.5
    assert check.loc[KEY_A, "group_best_arm"] == 1
    assert bool(check.loc[KEY_A, "reversal"]) is True
    assert check.loc[KEY_A, "gap"] == pytest.approx(5.0)
    assert bool(check.loc[KEY_B, "reversal"]) is False and check.loc[KEY_B, "gap"] == 0.0
    assert check.loc[KEY_A, "n_group_obs"] == 10


def test_group_key_cache_matches_segmenter(segmenter):
    rng = np.random.default_rng(6)
    refined = segmenter.refine().refine()  # adds the numeric fico/dti bands
    grades = ["A", "G", None, np.nan, pd.NA]
    purposes = ["credit_card", "car", None]
    homes = ["RENT", "own", "NONE", None]
    contexts = []
    for i in range(24):
        feats = {
            "grade": grades[i % len(grades)],
            "purpose": purposes[i % len(purposes)],
            "fico_range_low": float(rng.choice([640.0, 745.0, np.nan])),
            "dti": float(rng.uniform(0, 40)) if i % 7 else None,
        }
        if i % 5:
            feats["home_ownership"] = homes[i % len(homes)]  # sometimes absent entirely
        contexts.append(BorrowerContext(loan_id=i, features=feats) if i % 2 else feats)
    for seg in (segmenter, refined):
        resolver = GroupKeyResolver(seg)
        expected = [seg.key_for(c) for c in contexts]
        assert [resolver(c) for c in contexts] == expected
        calls = []
        original = seg.key_for
        seg.key_for = lambda c, _orig=original, _calls=calls: _calls.append(1) or _orig(c)
        try:
            assert [resolver(c) for c in contexts] == expected
        finally:
            del seg.key_for
        assert calls == []  # second pass served entirely from the cache


def test_prime_groups_fills_the_cache_for_row_built_contexts(small_arm_space, segmenter):
    rng = np.random.default_rng(9)
    n = 200
    df = pd.DataFrame(
        {
            "loan_id": np.arange(n),
            "grade": pd.Series(rng.choice(list("ABCDEFG"), n), dtype="string"),
            "purpose": rng.choice(["credit_card", "car", "debt_consolidation"], n),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", None], n),
            "fico_range_low": rng.choice(np.arange(660.0, 850.0, 5.0), n),
            "dti": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 40, n).round(2)),
            "annual_inc": rng.uniform(2e4, 2e5, n).round(0),
        }
    )
    refined = segmenter.refine().refine()  # continuous fico/dti bands: mostly distinct keys
    cts = ContextualThompsonSampling(small_arm_space, refined, cfg=BanditConfig(warm_start="flat"))
    keys = cts.prime_groups(df)
    assert keys.tolist() == refined.assign(df).tolist()
    calls = []
    original = refined.key_for
    refined.key_for = lambda c, _orig=original, _calls=calls: _calls.append(1) or _orig(c)
    try:
        from_rows = [BorrowerContext.from_row(row) for _, row in df.iterrows()]  # numpy scalars
        from_records = [
            BorrowerContext(loan_id=r.pop("loan_id"), features=r) for r in df.to_dict("records")
        ]  # Python scalars
        groups = [cts.select(c, None, rng).group for c in from_rows + from_records]
    finally:
        del refined.key_for
    assert calls == []
    assert groups == keys.tolist() * 2
    flat = GaussianThompsonSampling(small_arm_space)
    assert set(flat.prime_groups(df)) == {"*"}


def test_resegment_keeps_pooled_and_routes_stale_feedback(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="global"))
    rng = np.random.default_rng(7)
    old = cts.select(A, None, rng)
    cts.update(old, 3.0)
    pooled_n = cts.arm_table("*")["n"].sum()
    finer = segmenter.refine()
    cts.resegment(finer)
    assert cts.group_keys() == [] and cts.segmentation_version == 1 and cts.segmenter == finer
    cts.update(old, 4.0)  # decision made under the old segmentation: pooled level only
    assert cts.group_keys() == []
    assert cts.arm_table("*")["n"].sum() == pooled_n + 1
    new = cts.select(ctx("A", "credit_card", "RENT", fico_range_low=720.0), None, rng)
    assert new.group == finer.key_for(ctx("A", "credit_card", "RENT", fico_range_low=720.0))
    assert new.meta["seg_version"] == 1
    flat = GaussianThompsonSampling(small_arm_space)
    with pytest.raises(ConfigurationError):
        flat.resegment(finer)


def test_restore_then_feedback_for_unknown_group(small_arm_space, segmenter):
    cts = ContextualThompsonSampling(small_arm_space, segmenter, cfg=BanditConfig(warm_start="flat"))
    rng = np.random.default_rng(8)
    cts.update(cts.select(A, None, rng), 1.0)
    champion = cts.snapshot()
    late = cts.select(B, None, rng)  # group created after the snapshot
    cts.restore(champion)
    assert cts.group_keys() == [KEY_A]
    cts.update(late, 2.0)  # pending feedback re-creates the group lazily
    assert cts.group_keys() == sorted([KEY_A, KEY_B])
    assert cts.posterior(KEY_B, late.arm).n == 1.0


def test_contextual_requires_segmenter(small_arm_space):
    with pytest.raises(ConfigurationError):
        ContextualThompsonSampling(small_arm_space, None)  # type: ignore[arg-type]
