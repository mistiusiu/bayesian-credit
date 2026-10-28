"""ShadowLendingEngine: per-borrower reports and vectorised portfolio (SPEC §2.10, G.4)."""

import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, RewardConfig
from bayesian_credit.data.arms import LendingPolicy
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.exceptions import LeakageError
from bayesian_credit.pipeline import fit_models, prepare_dataset
from bayesian_credit.rewards import CompoundRewardFunction
from bayesian_credit.shadow import (
    ARM_TABLE_COLUMNS,
    OUTCOME_CLASSES,
    PER_LOAN_COLUMNS,
    ShadowLendingEngine,
    eligibility_matrix,
)


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, _ = prepare_dataset("synthetic", None, cfg=cfg, n=5000, seed=17)
    fitted = fit_models(df, cfg, "2015-01-01", evaluate=False)
    return cfg, df, fitted


def fresh_engine(n_samples: int = 2000, reward_cfg: RewardConfig | None = None) -> ShadowLendingEngine:
    _, _, f = _data()
    reward_fn = f.reward_fn if reward_cfg is None else CompoundRewardFunction(reward_cfg)
    return ShadowLendingEngine(f.two_stage, f.mctm, reward_fn, f.arm_space, f.segmenter, n_samples, seed=0)


@lru_cache(maxsize=1)
def _shared_engine() -> ShadowLendingEngine:
    return fresh_engine()


def engine(reward_cfg: RewardConfig | None = None) -> ShadowLendingEngine:
    """Shared engine (group tables cached across tests) unless a custom reward is needed."""
    return _shared_engine() if reward_cfg is None else fresh_engine(reward_cfg=reward_cfg)


def holdout() -> pd.DataFrame:
    _, df, _ = _data()
    return df.loc[df["issue_d"] >= "2015-01-01"].reset_index(drop=True)


def test_eligibility_matrix_matches_arm_space(arm_space):
    incomes = [np.nan, 0.0, -5.0, 1_000.0, 12_000.0, 35_000.0, 65_000.0, 250_000.0]
    mat = eligibility_matrix(arm_space, incomes)
    for row, inc in zip(mat, incomes, strict=True):
        np.testing.assert_array_equal(row, arm_space.eligible(None if np.isnan(inc) else inc))
    dti = [10.0, np.nan, 20.0, 45.0, 30.0, 5.0, 38.0, 12.0]
    ref = [0.07, 0.13, np.nan, 0.2, 0.28, 0.1, 0.16, 0.25]
    mat = eligibility_matrix(arm_space, incomes, dti, ref)
    for row, inc, d, r in zip(mat, incomes, dti, ref, strict=True):
        np.testing.assert_array_equal(row, arm_space.eligible(None if np.isnan(inc) else inc, dti=d, reference_rate=r))


def test_group_table_columns_and_monte_carlo_consistency():
    eng = engine()
    _, df, _ = _data()
    g = str(df["y_group"].value_counts().index[0])
    tab = eng.group_table(g)
    assert tuple(tab.columns) == ARM_TABLE_COLUMNS and len(tab) == eng.arm_space.n_arms
    lend = ~tab["is_null"]
    assert (tab.loc[lend, "reward_q05"] <= tab.loc[lend, "reward_q50"]).all()
    assert (tab.loc[lend, "reward_q50"] <= tab.loc[lend, "reward_q95"]).all()
    assert (tab.loc[lend, "p_default_hdi_low"] <= tab.loc[lend, "p_default_mean"]).all()
    assert (tab.loc[lend, "p_default_mean"] <= tab.loc[lend, "p_default_hdi_high"]).all()
    assert tab.loc[~lend, ["reward_mean", "reward_std", "reward_q50", "prob_positive"]].abs().to_numpy().sum() == 0.0
    # the Monte Carlo median sits close to the closed-form mean (rewards are near-symmetric)
    rel = (tab.loc[lend, "reward_q50"] - tab.loc[lend, "reward_mean"]).abs() / tab.loc[lend, "reward_std"].clip(lower=1e-9)
    assert rel.median() < 0.5
    pd.testing.assert_frame_equal(eng.group_table(g), tab)  # cached & deterministic
    again = fresh_engine().group_table(g)  # seeded per group: independent of evaluation order
    pd.testing.assert_frame_equal(again, tab)


def test_recommended_arm_is_the_best_admissible_arm():
    eng = engine()
    rows = holdout().head(40)
    for _, row in rows.iterrows():
        rep = eng.evaluate_loan(row)
        score = rep.arms["risk_adjusted"].to_numpy()
        mask = rep.eligible_mask
        assert mask[rep.recommended_arm]
        assert score[rep.recommended_arm] == pytest.approx(score[mask].max())
        assert rep.historical_arm == int(row["y_historical_arm"]) and mask[rep.historical_arm]
        assert rep.uplift_expected >= 0.0
        assert rep.lendable == bool(np.any(mask & ~rep.arms["is_null"].to_numpy() & (rep.arms["prob_positive"] >= 0.6)))
        if row["y_resolved"]:
            assert rep.historical_realized_reward == pytest.approx(eng.reward_fn.terminal(row))
            assert rep.uplift_vs_realized == pytest.approx(rep.arms["reward_mean"].iloc[rep.recommended_arm] - rep.historical_realized_reward)
        else:
            assert rep.historical_realized_reward is None and rep.uplift_vs_realized is None
    d = rep.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["recommended_arm"] == rep.recommended_arm and len(d["arms"]) == eng.arm_space.n_arms
    assert d["leakage_safe"] is True and set(d["context_columns"]) <= set(BorrowerContext.from_row(row).features)


def test_decline_is_recommended_when_every_lending_arm_loses():
    eng = engine(reward_cfg=RewardConfig(funding_rate=5.0))  # funding cost swamps every interest income
    row = holdout().iloc[0]
    rep = eng.evaluate(BorrowerContext.from_row(row))
    lend = ~rep.arms["is_null"]
    assert (rep.arms.loc[lend, "reward_mean"] < 0).all()
    assert rep.recommended_arm == 0 and rep.declined and not rep.lendable
    rep2 = eng.evaluate(BorrowerContext.from_row(row), LendingPolicy(10_000, 0.13, 36))
    assert rep2.recommended_arm == 0 and rep2.uplift_expected > 0


def test_context_with_later_stage_columns_is_rejected():
    eng = engine()
    with pytest.raises(LeakageError):
        eng.evaluate({"grade": "B", "purpose": "credit_card", "home_ownership": "RENT", "loan_status": "Charged Off"})
    rep = eng.evaluate({"grade": "B", "purpose": "credit_card", "home_ownership": "RENT", "annual_inc": 55_000.0},
                       {"amount": 12_000, "rate": 0.11, "term": 36}, realized_reward=-400.0)
    assert rep.group == "B|credit_card|RENT" and rep.historical_arm == eng.arm_space.nearest(12_000, 0.11, 36)
    assert rep.historical_realized_reward == -400.0


def test_portfolio_is_consistent_with_per_borrower_reports():
    eng = engine()
    data = holdout()
    per_loan, summary = eng.evaluate_portfolio(data, max_loans=1500, seed=3)
    assert tuple(per_loan.columns) == PER_LOAN_COLUMNS and len(per_loan) == 1500
    assert per_loan["loan_id"].is_unique
    assert (per_loan["uplift_expected"] >= -1e-9).all()
    # rescued ⊆ historically loss-making loans; flags imply the class precedence
    rescued = per_loan["rescued"]
    assert (per_loan.loc[rescued, "historical_realized"] < 0).all()
    assert not (rescued & per_loan["declined"]).any()
    assert (per_loan.loc[rescued, "recommended_expected"] > 0).all()
    assert set(per_loan["outcome_class"]) <= set(OUTCOME_CLASSES)
    counts = per_loan["outcome_class"].value_counts()
    assert sum(summary.counts[c] for c in OUTCOME_CLASSES) == len(per_loan)
    for c in OUTCOME_CLASSES:
        assert summary.counts[c] == int(counts.get(c, 0))
    assert summary.counts["n_rescued"] == int(rescued.sum())
    assert summary.n_lendable == int(per_loan["lendable"].sum())
    assert summary.expected_profit_shadow == pytest.approx(per_loan["recommended_expected"].sum())
    assert summary.uplift == pytest.approx(summary.expected_profit_shadow - summary.expected_profit_historical)
    assert int(summary.by_grade["n"].sum()) == len(per_loan)
    assert set(summary.inclusion.columns) >= {"grade", "share_lendable", "mean_rate_historical", "mean_rate_shadow"}
    json.dumps(summary.to_dict(), allow_nan=False)
    # vectorised portfolio == one-by-one evaluate_loan
    sample = data.set_index("loan_id").loc[per_loan["loan_id"].head(25)]
    for (lid, row), (_, pl) in zip(sample.iterrows(), per_loan.head(25).iterrows(), strict=True):
        rep = eng.evaluate_loan(row.assign(loan_id=lid) if hasattr(row, "assign") else row)
        assert rep.recommended_arm == pl["recommended_arm"]
        assert rep.lendable == pl["lendable"]
        assert rep.uplift_expected == pytest.approx(pl["uplift_expected"])
        assert rep.historical_expected_reward == pytest.approx(pl["historical_expected"])


def test_portfolio_of_200k_loans_runs_under_ten_seconds():
    eng = fresh_engine()  # includes building every group table
    data = holdout()
    big = pd.concat([data] * (200_000 // len(data) + 1), ignore_index=True).head(200_000)
    big = big.assign(loan_id=np.arange(len(big)))
    t0 = time.perf_counter()
    per_loan, summary = eng.evaluate_portfolio(big)
    elapsed = time.perf_counter() - t0
    assert len(per_loan) == 200_000 and summary.n_loans == 200_000
    assert elapsed < 10.0, elapsed


# ----- guardrails with reference rates (the engine a fit builds) ---------------------------------------
@lru_cache(maxsize=1)
def _fitted_engine() -> ShadowLendingEngine:
    _, _, f = _data()
    return f.shadow_engine(n_samples=300)


def test_engine_from_a_fit_applies_the_rate_corridor_and_reports_eligibility():
    eng = _fitted_engine()
    _, _, f = _data()
    assert eng.reference_rates is f.reference_rates and f.reference_rates is not None
    rows = holdout().head(60)
    corridor = eng.arm_space.rate_corridor
    for _, row in rows.iterrows():
        rep = eng.evaluate_loan(row)
        ref = f.reference_rates.for_context(row)
        assert rep.reference_rate == pytest.approx(ref)
        base = eng.arm_space.eligible(row["annual_inc"], dti=row["dti"], reference_rate=ref)
        expected = base.copy()
        expected[rep.historical_arm] = True
        np.testing.assert_array_equal(rep.eligible_mask, expected)
        rec = eng.arm_space[rep.recommended_arm]
        if not rec.is_null and rep.recommended_arm != rep.historical_arm:
            assert abs(rec.rate - ref) <= corridor + 1e-9
            assert rec.amount <= 0.6 * row["annual_inc"] + 1e-6
        d = rep.to_dict()
        el = d["eligibility"]
        assert el["reference_rate"] == pytest.approx(ref, abs=1e-6) and el["n_eligible"] == int(expected.sum())
        assert el["n_eligible_lending"] == int((base & eng.arm_space.is_lending).sum())
        assert el["historical_admitted"] == (not bool(base[rep.historical_arm])) and el["rate_corridor_applied"] is True
        assert el["guardrails"] == eng.arm_space.guardrails
        json.dumps(d, allow_nan=False)
    # an explicit reference rate overrides the lookup; no reference rates -> no corridor
    ctx = {"grade": "B", "sub_grade": "B2", "purpose": "credit_card", "home_ownership": "RENT", "annual_inc": 80_000.0, "dti": 10.0}
    rep = eng.evaluate(ctx, reference_rate=0.25)
    lend = rep.eligible_mask & eng.arm_space.is_lending
    assert rep.reference_rate == 0.25 and (np.abs(eng.arm_space.rates[lend] - 0.25) <= corridor + 1e-9).all()
    rep_free = engine().evaluate(ctx)
    assert rep_free.reference_rate is None and rep_free.to_dict()["eligibility"]["rate_corridor_applied"] is False


def test_portfolio_with_guardrails_matches_single_reports_and_shows_the_mix():
    eng = _fitted_engine()
    _, _, f = _data()
    data = holdout()
    per_loan, summary = eng.evaluate_portfolio(data, max_loans=800, seed=5)
    assert tuple(per_loan.columns) == PER_LOAN_COLUMNS
    sample = data.set_index("loan_id").loc[per_loan["loan_id"]]
    np.testing.assert_allclose(per_loan["reference_rate"].to_numpy(), f.reference_rates.for_frame(sample.reset_index()))
    for (lid, row), (_, pl) in zip(sample.head(30).iterrows(), per_loan.head(30).iterrows(), strict=True):
        rep = eng.evaluate_loan(row.assign(loan_id=lid) if hasattr(row, "assign") else row)
        assert rep.recommended_arm == pl["recommended_arm"] and rep.lendable == pl["lendable"]
        assert rep.to_dict()["eligibility"]["n_eligible_lending"] == pl["n_eligible_lending"]
    mix = summary.recommendation_mix
    assert {"recommended", "historical", "mean_rate_recommended", "mean_rate_historical", "mean_amount_recommended",
            "mean_amount_historical", "share_36m_recommended", "reference"} <= set(mix)
    rec = mix["recommended"]
    assert set(rec["by_term"]) == {"36", "60"} and sum(rec["by_rate"].values()) == pytest.approx(1.0)
    assert sum(rec["by_amount"].values()) == pytest.approx(1.0)
    lend = ~per_loan["declined"].to_numpy()
    rates = eng.arm_space.rates[per_loan["recommended_arm"].to_numpy()][lend]
    assert rec["mean_rate"] == pytest.approx(rates.mean()) and rec["n_lending"] == int(lend.sum())
    assert mix["reference"]["share_within_corridor"] > 0.9  # only historical arms can sit outside the corridor
    out = summary.to_dict()
    json.dumps(out, allow_nan=False)
    assert out["recommendation_mix"]["recommended"]["n"] == 800 and out["eligibility"]["reference_rates"] is True
    dec = summary.uplift_decomposition  # nested admissible sets -> nested optima (λ = 0: score = mean)
    hist = dec["expected_profit_historical"]
    assert hist == pytest.approx(summary.expected_profit_historical)
    reprice, resize, full = (dec[k]["expected_profit"] for k in ("repricing_only", "repricing_and_resizing", "unrestricted"))
    assert hist - 1e-6 <= reprice <= resize + 1e-6 and resize <= full + 1e-6
    assert full == pytest.approx(summary.expected_profit_shadow)
    assert out["uplift_decomposition"]["unrestricted"]["uplift"] == pytest.approx(summary.uplift, abs=0.01)


def test_arm_tables_use_the_prepayment_factor_and_the_risk_aversion_sweep_is_consistent():
    eng = _fitted_engine()
    _, _, f = _data()
    assert eng.interest_realization is f.interest_realization and f.interest_realization is not None
    g = str(holdout()["y_group"].value_counts().index[0])
    tab = eng.group_table(g)
    assert tuple(tab.columns) == ARM_TABLE_COLUMNS
    for a in (0, 1, 70, 128):
        rho = f.realization(g, a)
        assert tab["realization"].iloc[a] == rho == eng.realization(g, a)
        if a:
            ref = eng.reward_fn.expected(eng.arm_space[a], eng.two_stage.risk(g, a), eng.mctm, realization=rho)
            assert tab["reward_mean"].iloc[a] == ref.mean
    assert tab.loc[tab["term"] == 60, "realization"].lt(1.0).all()
    per_loan, summary = eng.evaluate_portfolio(holdout(), max_loans=700, seed=9)
    sweep = summary.by_risk_aversion
    assert list(sweep) == ["0.0", "0.5", "1.0"]
    base = sweep["0.0"]  # the engine's own λ (0): identical to the main summary
    assert base["expected_profit_shadow"] == pytest.approx(summary.expected_profit_shadow)
    assert base["uplift"] == pytest.approx(summary.uplift)
    mix = summary.recommendation_mix
    assert base["mean_amount_recommended"] == pytest.approx(mix["mean_amount_recommended"])
    assert base["share_36m_recommended"] == pytest.approx(mix["share_36m_recommended"])
    for key in ("0.5", "1.0"):  # λ = 0 maximises the expected reward loan by loan
        assert sweep[key]["expected_profit_shadow"] <= base["expected_profit_shadow"] + 1e-6
        assert sweep[key]["risk_aversion"] == float(key)
    none = eng.evaluate_portfolio(holdout(), max_loans=50, seed=9, risk_aversions=None)[1]
    assert none.by_risk_aversion == {} and "by_risk_aversion" in none.to_dict()
