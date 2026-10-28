"""TwoStageBetaBeta facade and RiskEstimate (SPEC §2.4)."""

import json

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, HierarchyConfig, RecoveryConfig
from bayesian_credit.exceptions import DataSourceError
from bayesian_credit.models.two_stage import RiskEstimate, TwoStageBetaBeta

GROUPS = ["A|credit_card|RENT", "B|other|OWN", "C|debt_consolidation|MORTGAGE"]
RATES = {GROUPS[0]: 0.05, GROUPS[1]: 0.15, GROUPS[2]: 0.30}


@pytest.fixture
def table():
    rng = np.random.default_rng(42)
    n = 4000
    df = pd.DataFrame(
        {
            "loan_id": np.arange(n),
            "y_group": pd.Series(rng.choice(GROUPS, n), dtype="string"),
            "y_historical_arm": rng.integers(-1, 8, n),
            "y_resolved": rng.random(n) < 0.85,
        }
    )
    p = df["y_group"].map(RATES).to_numpy(dtype=float)
    default = (rng.random(n) < p).astype(float)
    df["y_default"] = np.where(df["y_resolved"], default, np.nan)
    df["y_recovery_rate"] = np.where(df["y_default"] == 1.0, rng.beta(1.2, 6.0, n), np.nan)
    return df


@pytest.fixture
def model(table):
    return TwoStageBetaBeta().fit(table)


def test_fit_uses_resolved_mapped_rows(model, table):
    use = table[table["y_resolved"] & (table["y_historical_arm"] >= 0)]
    for (g, a), grp in use.groupby(["y_group", "y_historical_arm"]):
        d = int(grp["y_default"].sum())
        assert model.default_model.counts(g, a) == (d, len(grp) - d)
    defaulted = use[use["y_default"] == 1.0]
    assert model.recovery_model.stats().n == len(defaulted)
    assert model.recovery_model.stats().mean == pytest.approx(defaulted["y_recovery_rate"].mean())
    assert model.default_model.diagnostics["skipped"] == int((table["y_resolved"] & (table["y_historical_arm"] < 0)).sum())


def test_risk_estimate_fields(model):
    est = model.risk(GROUPS[2], 3)
    assert isinstance(est, RiskEstimate)
    assert (est.group, est.arm) == (GROUPS[2], 3)
    assert est.p_default == model.default_model.posterior(GROUPS[2], 3)
    assert est.recovery == model.recovery_model.posterior(GROUPS[2], 3)
    assert est.n_default_obs == model.default_model.n_obs(GROUPS[2], 3)
    assert est.n_recovery_obs == model.recovery_model.n_obs(GROUPS[2], 3)
    assert est.recovery_source in {"leaf", "group", "global", "prior"}
    assert est.expected_loss_fraction == pytest.approx(est.p_default.mean * (1 - est.recovery.mean))
    assert est.expected_default == est.p_default.mean
    assert est.expected_recovery == est.recovery.mean
    # higher-risk group is estimated riskier
    assert model.risk(GROUPS[2], 3).expected_default > model.risk(GROUPS[0], 3).expected_default
    d = est.to_dict()
    assert d["group"] == GROUPS[2] and "hdi_low" in d["p_default"]


def test_unseen_group_and_arm(model):
    est = model.risk("Z|never|seen", 77)
    assert est.n_default_obs == 0 and est.n_recovery_obs == 0
    assert est.p_default == model.default_model.group_prior()
    assert est.recovery_source == "global"


def test_update_routes_recovery_only_for_defaults(model):
    g, a = GROUPS[0], 2
    d0, n0 = model.default_model.counts(g, a)
    r0 = model.recovery_model.n_obs(g, a)
    model.update(g, a, False, recovery_rate=0.9)  # ignored by stage 2
    model.update(g, a, True, recovery_rate=0.4)
    model.update(g, a, True)  # default without a known recovery
    assert model.default_model.counts(g, a) == (d0 + 2, n0 + 1)
    assert model.recovery_model.n_obs(g, a) == r0 + 1


def test_sample_returns_independent_posterior_draws(model, rng):
    p, r = model.sample(GROUPS[1], 4, 20_000, rng)
    est = model.risk(GROUPS[1], 4)
    assert p.shape == r.shape == (20_000,)
    assert p.mean() == pytest.approx(est.p_default.mean, abs=0.005)
    assert r.mean() == pytest.approx(est.recovery.mean, abs=0.01)
    assert abs(np.corrcoef(p, r)[0, 1]) < 0.03


def test_predict_table_matches_risk_per_row(model, table):
    pred = model.predict_table(table)
    assert list(pred.columns) == [
        "p_default_mean", "p_default_hdi_low", "p_default_hdi_high",
        "expected_recovery", "expected_loss_fraction",
    ]
    assert pred.index.equals(table.index)
    unmapped = table["y_historical_arm"] < 0
    assert pred[unmapped].isna().all().all()
    assert pred[~unmapped].notna().all().all()
    for idx in table.index[~unmapped][:25]:
        row = table.loc[idx]
        est = model.risk(row["y_group"], int(row["y_historical_arm"]))
        assert pred.loc[idx, "p_default_mean"] == pytest.approx(est.p_default.mean, rel=1e-12)
        assert pred.loc[idx, "expected_recovery"] == pytest.approx(est.recovery.mean, rel=1e-12)
        assert pred.loc[idx, "expected_loss_fraction"] == pytest.approx(est.expected_loss_fraction, rel=1e-12)
    idx = table.index[~unmapped][0]
    row = table.loc[idx]
    lo, hi = model.risk(row["y_group"], int(row["y_historical_arm"])).p_default.hdi(0.95)
    assert pred.loc[idx, "p_default_hdi_low"] == pytest.approx(lo, abs=1e-4)
    assert pred.loc[idx, "p_default_hdi_high"] == pytest.approx(hi, abs=1e-4)


def test_predict_table_custom_columns_and_empty(model, table):
    shadow = table.assign(arm=3).set_index("loan_id")
    pred = model.predict_table(shadow, arm_col="arm")
    assert pred.index.equals(shadow.index)
    assert pred["p_default_mean"].nunique() == 3  # one leaf per group
    assert model.predict_table(table.iloc[:0]).empty


def test_fit_requires_derived_columns(table):
    with pytest.raises(DataSourceError):
        TwoStageBetaBeta().fit(table.drop(columns=["y_recovery_rate"]))


def test_config_entry_points(table):
    cfg = FrameworkConfig(hierarchy=HierarchyConfig(kappa_leaf=3.0), recovery=RecoveryConfig(min_obs=12))
    for m in (TwoStageBetaBeta(cfg), TwoStageBetaBeta.from_config(cfg), TwoStageBetaBeta(cfg.hierarchy, cfg.recovery)):
        assert m.default_model.cfg.kappa_leaf == 3.0
        assert m.recovery_model.cfg.min_obs == 12


def test_nullable_dtypes_are_accepted(table):
    t = table.astype({"y_resolved": "boolean", "y_default": "Float64", "y_historical_arm": "Int64"})
    a = TwoStageBetaBeta().fit(t)
    b = TwoStageBetaBeta().fit(table)
    assert a.to_dict() == b.to_dict()


def test_to_dict_round_trip(model):
    model.update(GROUPS[0], 1, True, 0.3)
    state = model.to_dict()
    restored = TwoStageBetaBeta.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state
    for g in GROUPS + ["new"]:
        for a in (0, 1, 5):
            assert restored.risk(g, a) == model.risk(g, a)


# ----------------------------------------------------------------------------------------
# structured leaf prior (policy effect)
# ----------------------------------------------------------------------------------------
from functools import lru_cache  # noqa: E402

from bayesian_credit.config import ArmSpaceConfig  # noqa: E402
from bayesian_credit.data.arms import ArmSpace  # noqa: E402


@lru_cache(maxsize=1)
def _synthetic():
    from bayesian_credit.pipeline import prepare_dataset

    cfg = FrameworkConfig()
    df, _ = prepare_dataset("synthetic", None, cfg=cfg, n=30_000, seed=11)
    arms = ArmSpace.from_config(cfg.arms)
    train = df.loc[df["issue_d"] < "2015-01-01"]
    return cfg, arms, train, TwoStageBetaBeta.from_config(cfg, arms).fit(train)


def test_policy_effect_makes_grade_a_riskier_at_28_than_at_7_percent():
    cfg, arms, train, model = _synthetic()
    info = model.policy_effect_info
    assert info["status"] == "fitted" and info["installed"] and info["n_fit"] >= cfg.hierarchy.policy_effect_cfg.min_rows
    assert info["beta"]["int_rate"] > 0 and info["beta"]["pti"] > 0
    assert model.policy_effect is not None and model.arm_space == arms
    a7, a28 = arms.nearest(10_000, 0.07, 36), arms.nearest(10_000, 0.28, 36)
    a_groups = [g for g in model.default_model.groups if g.split("|")[0] == "A"]
    assert a_groups
    for g in a_groups:
        assert model.risk(g, a28).expected_default > model.risk(g, a7).expected_default
    g = max(a_groups, key=lambda k: sum(model.default_model.group_counts(k)))
    assert model.risk(g, a28).n_default_obs == 0  # nobody in grade A was ever priced at 28 % ...
    assert model.risk(g, a28).expected_default > 2 * model.default_model.group_posterior(g).mean  # ... and it is not "group average"
    # the decline arm is never tilted; an unseen group is tilted around the global means
    assert model.default_model.prior_tilt(g, 0) == 0.0
    assert model.risk("Z|never|SEEN", a28).expected_default > model.risk("Z|never|SEEN", a7).expected_default
    # without an arm space the leaf prior is the plain shrunk group mean
    plain = TwoStageBetaBeta.from_config(cfg).fit(train)
    assert plain.policy_effect is None and plain.policy_effect_info["status"] == "no_arm_space"
    assert plain.risk(g, a28).expected_default == pytest.approx(plain.default_model.group_posterior(g).mean)


def test_policy_effect_switches_and_stale_tilts(table):
    cfg, arms, train, _ = _synthetic()
    no_cols = TwoStageBetaBeta(arm_space=arms).fit(table)  # the toy table has no policy columns
    assert no_cols.policy_effect_info["status"].startswith("missing_columns") and no_cols.policy_effect is None
    off = TwoStageBetaBeta(HierarchyConfig(policy_effect=False), arm_space=arms).fit(train)
    assert off.policy_effect_info["status"] == "disabled" and off.default_model.leaf_prior_tilt is None
    model = TwoStageBetaBeta.from_config(cfg, arms).fit(train)
    assert model.default_model.leaf_prior_tilt is not None
    model.fit(train.drop(columns=["installment"]))  # a re-fit that cannot fit the effect drops the old tilt
    assert model.default_model.leaf_prior_tilt is None and "installment" in model.policy_effect_info["status"]
    small = TwoStageBetaBeta(FrameworkConfig(), arm_space=arms).fit(train.head(800))  # too few rows -> beta = 0
    assert small.policy_effect_info["status"] == "fallback" and not small.policy_effect.has_effect
    ref = TwoStageBetaBeta(FrameworkConfig(), arm_space=arms).fit(train.head(800))
    small.default_model.set_leaf_prior_tilt(None)
    for a in (1, 64, 128):  # a zero tilt is exactly no tilt
        assert ref.risk("A|credit_card|RENT", a) == small.risk("A|credit_card|RENT", a)
    assert ArmSpaceConfig().rate_corridor == 0.05


def test_to_dict_round_trip_restores_the_policy_effect():
    cfg, arms, train, fitted = _synthetic()
    model = TwoStageBetaBeta.from_dict(json.loads(json.dumps(fitted.to_dict())))  # independent copy
    model.update(fitted.default_model.groups[0], 5, True, 0.2)
    state = model.to_dict()
    restored = TwoStageBetaBeta.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state
    assert restored.policy_effect.coefficients == fitted.policy_effect.coefficients
    assert restored.arm_space == arms and restored.policy_effect_info["status"] == "fitted"
    for g in fitted.default_model.groups[:5] + ["Z|never|SEEN"]:
        for a in (0, 1, 30, 77, 128):
            assert restored.risk(g, a) == model.risk(g, a)


def test_recovery_term_level_follows_the_arm_space_and_can_be_switched_off():
    cfg, arms, train, model = _synthetic()
    rm = model.recovery_model
    assert rm.has_terms and rm.term_of(arms.nearest(10_000, 0.13, 60)) == 60
    sources = rm.diagnostics
    assert sources["term_level"] and sources["source_group_term"] > 0
    g = max(model.default_model.groups, key=lambda k: sum(model.default_model.group_counts(k)))
    a60 = arms.nearest(40_000, 0.28, 60)
    est = model.risk(g, a60)
    assert est.recovery_source in ("group_term", "group", "global") and est.n_recovery_obs == 0
    if est.recovery_source == "group_term":
        assert est.recovery == rm.group_term_posterior(g, 60)[0]
    off = TwoStageBetaBeta(FrameworkConfig(recovery=RecoveryConfig(term_level=False)), arm_space=arms).fit(train)
    assert not off.recovery_model.has_terms and off.recovery_model.diagnostics["source_group_term"] == 0
    back = TwoStageBetaBeta.from_dict(json.loads(json.dumps(model.to_dict())))
    assert back.recovery_model.has_terms and back.risk(g, a60) == model.risk(g, a60)
