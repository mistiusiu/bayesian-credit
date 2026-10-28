"""Structured leaf prior: PolicyEffectModel / PolicyEffectTilt (docs/MATH.md §"Structured
leaf prior")."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from bayesian_credit.config import PolicyEffectConfig
from bayesian_credit.data.arms import NULL_POLICY, ArmSpace, LendingPolicy
from bayesian_credit.models.policy_effect import (
    PolicyEffectModel,
    PolicyEffectTilt,
    TiltTable,
    policy_effect_columns,
    tilt_from_dict,
)

TRUE_BETA = np.array([10.0, 0.4, 4.0, -0.2])  # int_rate, term60, pti, log_amount


def annuity(amount, rate, term):
    i = rate / 12.0
    f = (1.0 + i) ** term
    return amount * i * f / (f - 1.0)


def make_loans(n=150_000, beta=TRUE_BETA, seed=0, n_groups=6):
    """Loans whose default follows the model exactly: logit p = a_g + β·(x − x̄_g)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    rate = 0.07 + 0.03 * g + rng.normal(0.0, 0.03, n)
    rate = np.clip(rate, 0.03, 0.35)
    term = np.where(rng.random(n) < 0.3, 60, 36)
    amount = np.round(np.exp(rng.normal(np.log(12_000), 0.5, n)), -2)
    income = np.exp(rng.normal(np.log(65_000), 0.4, n))
    installment = annuity(amount, rate, term)
    X = np.column_stack([rate, (term == 60).astype(float), np.clip(12 * installment / income, 0, 1.5), np.log(amount)])
    means = np.stack([X[g == k].mean(axis=0) for k in range(n_groups)])
    a_g = logit(0.04 + 0.035 * np.arange(n_groups))
    y = (rng.random(n) < expit(a_g[g] + (X - means[g]) @ beta)).astype(float)
    groups = np.array([f"G{k}" for k in g], dtype=object)
    return {"groups": groups, "defaults": y, "int_rate": rate, "term": term, "loan_amnt": amount,
            "installment": installment, "annual_inc": income}


def fit(data, cfg=None, **kw):
    return PolicyEffectModel(cfg).fit_arrays(
        data["groups"], data["defaults"], data["int_rate"], data["term"], data["loan_amnt"], data["installment"],
        data["annual_inc"], **kw,
    )


@pytest.fixture(scope="module")
def loans():
    return make_loans()


@pytest.fixture(scope="module")
def model(loans):
    return fit(loans)


def test_recovers_the_true_coefficients(model):
    d = model.diagnostics
    assert d["status"] == "fitted" and d["converged"] and d["n_iter"] <= 10
    assert d["sample"] == "resolved" and d["n_fit"] == 150_000
    se = np.array([d["std_errors"][k] for k in model.feature_names])
    assert np.all(np.isfinite(se)) and np.all(se > 0)
    assert np.all(np.abs(model.beta - TRUE_BETA) <= np.maximum(4 * se, 0.02)), (model.beta, se)
    assert model.coefficients == pytest.approx(dict(zip(("int_rate", "term60", "pti", "log_amount"), model.beta, strict=True)))
    assert d["mean_log_loss"] < d["mean_log_loss_offset_only"]


def test_centring_makes_beta_invariant_to_a_group_level_rate_shift(loans, model):
    shifted = dict(loans)
    k = np.array([int(g[1:]) for g in loans["groups"]])
    shifted["int_rate"] = loans["int_rate"] + 0.02 * k  # a constant per group: pricing differences between groups
    other = fit(shifted)
    np.testing.assert_allclose(other.beta, model.beta, rtol=1e-8, atol=1e-10)
    # the group means move with the shift, so an arm's effect is measured from the group's own mean rate
    pol = LendingPolicy(10_000, 0.20, 36)
    assert other.group_means["G3"][0] == pytest.approx(model.group_means["G3"][0] + 0.06)
    assert other.effect("G3", pol) == pytest.approx(model.effect("G3", pol) - other.beta[0] * 0.06, abs=1e-9)


def test_effect_formula_clip_and_null_arm(model):
    g = "G2"
    pol = LendingPolicy(15_000, 0.16, 60)
    x = np.array([0.16, 1.0, min(pol.installment * 12 / model.group_median_income[g], 1.5), math.log(15_000)])
    expected = float(model.beta @ (x - model.group_means[g]))
    assert model.arm_features(g, pol) == pytest.approx(list(x))
    assert model.effect(g, pol) == pytest.approx(expected, abs=1e-12)
    assert abs(expected) < 3.0
    extreme = LendingPolicy(40_000, 0.95, 60)
    assert model.raw_effect(g, extreme) > 3.0 and model.effect(g, extreme) == 3.0
    steep = PolicyEffectModel.from_dict(model.to_dict())
    steep.beta = 2.0 * model.beta
    cheap = LendingPolicy(1_000, 0.0001, 36)
    assert steep.raw_effect("G5", cheap) < -3.0 and steep.effect("G5", cheap) == -3.0
    tight = PolicyEffectModel(PolicyEffectConfig(max_abs_logit_shift=0.25))
    tight.beta, tight.group_means, tight.group_median_income = model.beta, model.group_means, model.group_median_income
    assert abs(tight.effect(g, pol)) <= 0.25
    assert model.effect(g, NULL_POLICY) == 0.0 and model.raw_effect(g, None) == 0.0


def test_unseen_group_uses_the_global_means(model):
    pol = LendingPolicy(10_000, 0.13, 36)
    x = np.array(model.arm_features("never-seen", pol))
    assert x[2] == pytest.approx(pol.installment * 12 / model.global_median_income)
    assert model.effect("never-seen", pol) == pytest.approx(float(model.beta @ (x - model.global_means)), abs=1e-12)


def test_to_dict_round_trip_is_exact(model):
    state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    back = PolicyEffectModel.from_dict(state)
    assert back.to_dict() == state
    for g in ("G0", "G4", "unseen"):
        for pol in (LendingPolicy(5_000, 0.07, 36), LendingPolicy(40_000, 0.28, 60)):
            assert back.effect(g, pol) == model.effect(g, pol)
    assert back.coefficients == model.coefficients and back.n_fit == model.n_fit


def test_fallback_on_singular_design_and_other_failures(loans):
    n = 8_000
    base = {k: v[:n].copy() for k, v in loans.items()}
    # 1) no within-group variation at all -> beta = 0
    flat = dict(base)
    k = np.array([int(g[1:]) for g in base["groups"]])
    flat["int_rate"] = 0.07 + 0.03 * k
    flat["term"] = np.full(n, 36)
    flat["loan_amnt"] = np.full(n, 10_000.0)
    flat["annual_inc"] = np.full(n, 60_000.0)
    flat["installment"] = annuity(flat["loan_amnt"], flat["int_rate"], flat["term"])
    m = fit(flat)
    assert m.diagnostics["status"] == "fallback" and "singular" in m.diagnostics["reason"]
    assert not m.has_effect and np.all(m.beta == 0.0)
    assert m.effect("G1", LendingPolicy(40_000, 0.28, 60)) == 0.0
    # 2) perfectly collinear features without a ridge -> singular Hessian -> beta = 0 ...
    col = dict(base)
    col["loan_amnt"] = np.full(n, 10_000.0)
    col["term"] = np.full(n, 36)
    col["installment"] = 2.0 * col["int_rate"] * col["annual_inc"] / 12.0  # pti = 2 * rate
    m0 = fit(col, PolicyEffectConfig(ridge=0.0, min_rows=100))
    assert m0.diagnostics["status"] == "fallback" and "singular" in m0.diagnostics["reason"]
    # ... while the default ridge regularises the same design
    m1 = fit(col, PolicyEffectConfig(min_rows=100))
    assert m1.diagnostics["status"] == "fitted" and np.all(np.isfinite(m1.beta))
    assert set(m1.diagnostics["dropped_features"]) == {"term60", "log_amount"}
    # 3) too few rows, 4) no convergence
    few = fit(base)
    assert few.diagnostics["status"] == "fitted"
    small = fit({k: v[:200] for k, v in base.items()})
    assert small.diagnostics["status"] == "fallback" and "insufficient" in small.diagnostics["reason"]
    slow = fit(base, PolicyEffectConfig(max_iter=1))
    assert slow.diagnostics["status"] == "fallback" and "converge" in slow.diagnostics["reason"]
    assert np.all(slow.beta == 0.0)


def test_rows_with_unusable_features_are_dropped(loans):
    data = {k: v[:20_000].copy() for k, v in loans.items()}
    data["annual_inc"][:50] = 0.0
    data["annual_inc"][50:60] = np.nan
    data["loan_amnt"][60:70] = np.nan
    m = fit(data)
    assert m.diagnostics["n_dropped_invalid"] == 70 and m.n_fit == 20_000 - 70
    only_rate = fit(data, PolicyEffectConfig(features=("int_rate",)))
    assert only_rate.feature_names == ("int_rate",) and only_rate.diagnostics["n_dropped_invalid"] == 0
    assert policy_effect_columns(("int_rate",)) == ("int_rate",)
    assert policy_effect_columns() == ("int_rate", "term", "installment", "annual_inc", "loan_amnt")


def test_fit_on_a_loan_table_prefers_elapsed_term_rows(loans):
    n = 30_000
    rng = np.random.default_rng(5)
    issue = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, n), unit="D")
    table = pd.DataFrame(
        {
            "y_group": pd.array(loans["groups"][:n], dtype="string"),
            "y_resolved": np.ones(n, dtype=bool),
            "y_default": loans["defaults"][:n],
            "y_historical_arm": np.ones(n, dtype=np.int64),
            "issue_d": issue.to_numpy(dtype="datetime64[ns]"),
            **{c: loans[c][:n] for c in ("int_rate", "term", "loan_amnt", "installment", "annual_inc")},
        }
    )
    end = issue.to_period("M").to_timestamp() + pd.to_timedelta(0, unit="D")
    elapsed = (pd.DatetimeIndex(end) + pd.to_timedelta(table["term"] * 30.44, unit="D")) <= pd.Timestamp("2019-03-31")
    m = PolicyEffectModel(PolicyEffectConfig(min_elapsed_rows=1_000)).fit(table, snapshot_date="2019-03-31")
    assert m.diagnostics["sample"] == "elapsed"
    assert abs(m.n_fit - int(elapsed.sum())) <= 0.01 * n
    assert m.diagnostics["n_reference"] == n  # x̄_g / median income use every resolved row
    resolved_only = PolicyEffectModel(PolicyEffectConfig(min_elapsed_rows=10**9)).fit(table, snapshot_date="2019-03-31")
    assert resolved_only.diagnostics["sample"] == "resolved" and resolved_only.n_fit == n
    no_snapshot = PolicyEffectModel().fit(table)
    assert no_snapshot.diagnostics["sample"] == "resolved"
    # unresolved / unmapped rows are ignored
    t2 = table.assign(y_resolved=np.arange(n) % 2 == 0, y_historical_arm=np.where(np.arange(n) % 3 == 0, -1, 1))
    m2 = PolicyEffectModel(PolicyEffectConfig(min_rows=100)).fit(t2)
    keep = (np.arange(n) % 2 == 0) & (np.arange(n) % 3 != 0)
    assert m2.diagnostics["n_reference"] == int(keep.sum())


def test_tilt_binds_the_model_to_an_arm_space(model, arm_space):
    tilt = PolicyEffectTilt(model, arm_space)
    for a in (0, 1, 40, 128):
        assert tilt("G1", a) == model.effect("G1", arm_space[a])
    assert tilt("G1", 0) == 0.0 and tilt("G1", 999) == 0.0 and tilt("G1", -1) == 0.0
    assert tilt.n_arms == arm_space.n_arms
    d = tilt.diagnostics()
    assert d["beta"] == model.coefficients and d["n_fit"] == model.n_fit and d["status"] == "fitted"
    back = tilt_from_dict(json.loads(json.dumps(tilt.to_dict())))
    assert isinstance(back, PolicyEffectTilt) and back.arm_space == arm_space
    assert all(back("G4", a) == tilt("G4", a) for a in range(arm_space.n_arms))
    table = TiltTable([["G1", 3, 0.5]])
    assert table("G1", 3) == 0.5 and table("G1", 4) == 0.0
    assert tilt_from_dict(table.to_dict()).table == table.table and tilt_from_dict(None) is None
    with pytest.raises(ValueError):
        tilt_from_dict({"type": "Nope"})


def test_invalid_inputs():
    with pytest.raises(ValueError):
        PolicyEffectModel().fit_arrays(["A"], [0.5], [0.1], [36], [1000.0], [30.0], [5e4])
    with pytest.raises(ValueError):
        PolicyEffectModel().fit_arrays(["A", "B"], [0, 1], [0.1], [36], [1000.0], [30.0], [5e4])
    from bayesian_credit.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        PolicyEffectModel(PolicyEffectConfig(features=("int_rate", "fico")))
    with pytest.raises(ConfigurationError):
        PolicyEffectModel(PolicyEffectConfig(max_abs_logit_shift=0.0))
    empty = PolicyEffectModel().fit_arrays([], [], [], [], [], [], [])
    assert empty.diagnostics["status"] == "fallback" and not empty.has_effect
    assert ArmSpace.from_grid([5_000], [0.1], [36]).n_arms == 2
