"""Policy-performance metrics on a small synthetic decision log (Table 11)."""

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.data.arms import ArmSpace
from bayesian_credit.evaluation.policy import (
    LOG_COLUMNS,
    convergence_epoch,
    cumulative_regret,
    cumulative_reward,
    decline_rate,
    inclusion_metrics,
    profitability_vs_historical,
    rolling_reward,
    selection_frequency,
)

SPACE = ArmSpace.from_grid([5_000, 10_000], [0.10, 0.20], [36])  # arms 0..4, 0 = decline


@pytest.fixture
def log() -> pd.DataFrame:
    """Two policies x two epochs x two borrowers."""
    return pd.DataFrame(
        {
            "epoch": ["2015-01", "2015-01", "2015-02", "2015-02"] * 2,
            "policy": ["thompson"] * 4 + ["historical"] * 4,
            "loan_id": [1, 2, 3, 4] * 2,
            "group": ["g"] * 8,
            "arm": [1, 0, 2, 4, 3, 3, 1, 0],
            "amount": [5_000.0, 0.0, 5_000.0, 10_000.0, 10_000.0, 10_000.0, 5_000.0, 0.0],
            "rate": [0.10, 0.0, 0.20, 0.20, 0.10, 0.10, 0.10, 0.0],
            "term": [36, 0, 36, 36, 36, 36, 36, 0],
            "declined": [False, True, False, False, False, False, False, True],
            "expected_reward": [10.0, 0.0, 20.0, 30.0, 5.0, 5.0, 10.0, 0.0],
            "oracle_reward": [30.0, 10.0, 30.0, 30.0, 30.0, 10.0, 30.0, 30.0],
            "realized_reward": pd.array([12.0, None, None, 25.0, 4.0, None, 9.0, None], dtype="Float64"),
            "fallback": [False] * 7 + [True],
        }
    )


def test_log_fixture_has_the_documented_schema(log):
    assert tuple(log.columns) == LOG_COLUMNS


def test_cumulative_reward(log):
    out = cumulative_reward(log).set_index(["policy", "epoch"])
    assert out.loc[("thompson", "2015-01")].tolist() == [2, 10.0, 10.0, 12.0, 12.0]
    assert out.loc[("thompson", "2015-02")].tolist() == [2, 50.0, 60.0, 25.0, 37.0]
    assert out.loc[("historical", "2015-02")].tolist() == [2, 10.0, 20.0, 9.0, 13.0]
    assert list(cumulative_reward(log).columns) == ["policy", "epoch", "n", "reward", "cum_reward", "realized", "cum_realized"]


def test_cumulative_regret(log):
    out = cumulative_regret(log).set_index(["policy", "epoch"])
    ts1, ts2 = out.loc[("thompson", "2015-01")], out.loc[("thompson", "2015-02")]
    assert (ts1["regret"], ts1["avg_regret"], ts1["cum_regret"]) == (30.0, 15.0, 30.0)
    assert (ts2["regret"], ts2["avg_regret"], ts2["cum_regret"], ts2["cum_avg_regret"]) == (10.0, 5.0, 40.0, 10.0)
    hist = out.loc[("historical", "2015-02")]
    assert (hist["cum_n"], hist["cum_regret"], hist["cum_avg_regret"]) == (4, 80.0, 20.0)


def test_rolling_reward_is_decision_weighted(log):
    out = rolling_reward(log, window=2).set_index(["policy", "epoch"])
    assert out.loc[("thompson", "2015-01"), "rolling_mean_reward"] == 5.0
    assert out.loc[("thompson", "2015-02"), "mean_reward"] == 25.0
    assert out.loc[("thompson", "2015-02"), "rolling_mean_reward"] == 15.0
    with pytest.raises(ValueError):
        rolling_reward(log, window=0)


def test_selection_frequency_covers_the_whole_grid(log):
    out = selection_frequency(log, SPACE)
    assert list(out.columns) == ["policy", "arm", "term", "amount", "rate", "count", "share"]
    assert len(out) == 2 * SPACE.n_arms
    ts = out[out["policy"] == "thompson"].set_index("arm")
    assert ts["count"].tolist() == [1, 1, 1, 0, 1]
    assert ts["share"].sum() == pytest.approx(1.0)
    assert ts.loc[0, ["term", "amount", "rate"]].tolist() == [0, 0.0, 0.0]
    assert ts.loc[4, ["term", "amount", "rate"]].tolist() == [36, 10_000.0, 0.20]
    pivot = ts[ts["term"] == 36].pivot_table(index="amount", columns="rate", values="count")
    assert pivot.shape == (2, 2)


def test_decline_rate(log):
    by_epoch = decline_rate(log).set_index(["policy", "epoch"])
    assert by_epoch.loc[("thompson", "2015-01"), "decline_rate"] == 0.5
    assert by_epoch.loc[("historical", "2015-02"), "n_declined"] == 1
    overall = decline_rate(log, by_epoch=False).set_index("policy")
    assert overall["decline_rate"].tolist() == [0.25, 0.25]
    derived = decline_rate(log.drop(columns=["declined"]), by_epoch=False)
    assert derived["n_declined"].tolist() == [1, 1]  # from amount <= 0


def test_profitability_vs_historical(log):
    out = profitability_vs_historical(log).set_index("policy")
    assert out.loc["thompson", "total_expected"] == 60.0 and out.loc["historical", "total_expected"] == 20.0
    assert out.loc["thompson", "total_realized"] == 37.0
    assert out.loc["thompson", "uplift_vs_historical"] == 40.0
    assert out.loc["thompson", "uplift_pct"] == pytest.approx(200.0)
    assert out.loc["historical", "uplift_pct"] == 0.0
    missing = profitability_vs_historical(log[log["policy"] == "thompson"])
    assert missing["uplift_pct"].isna().all()


def test_inclusion_metrics_join_on_loan_id(log):
    ctx = pd.DataFrame({"loan_id": ["1", "2", "3", "4"], "grade": ["A", "B", "A", "B"]})  # str ids
    out = inclusion_metrics(log, ctx, by="grade").set_index(["policy", "grade"])
    a = out.loc[("thompson", "A")]
    assert (a["n"], a["approval_share"], a["mean_amount"], a["mean_rate"], a["mean_expected_reward"]) == (
        2,
        1.0,
        5_000.0,
        pytest.approx(0.15),
        15.0,
    )
    b = out.loc[("thompson", "B")]
    assert b["approval_share"] == 0.5 and b["mean_amount"] == 10_000.0 and b["mean_expected_reward"] == 15.0
    assert out.loc[("historical", "B"), "mean_rate"] == pytest.approx(0.10)
    with pytest.raises(KeyError):
        inclusion_metrics(log, ctx, by="fico_band")


def test_convergence_epoch():
    epochs = [f"2016-{m:02d}" for m in range(1, 9)]
    regret = pd.DataFrame(
        {
            "policy": ["thompson"] * 8 + ["random"] * 8,
            "epoch": epochs * 2,
            "avg_regret": [100, 50, 20, 10, 9, 10, 9.5, 10] + [100, 20, 100, 20, 100, 20, 100, 20],
        }
    )
    out = convergence_epoch(regret, tol=0.05, patience=3).set_index("policy")
    assert out.loc["thompson", "convergence_epoch"] == "2016-04"
    assert out.loc["thompson", "final_value"] == pytest.approx((9.5 + 10 + 10) / 3)
    assert bool(out.loc["thompson", "converged"]) is True
    assert out.loc["random", "convergence_epoch"] is None and not out.loc["random", "converged"]
    strict = convergence_epoch(regret, tol=0.01, patience=3, absolute=True).set_index("policy")
    assert not strict.loc["thompson", "converged"]


def test_convergence_from_raw_log_and_datetime_epochs(log):
    raw = log.assign(epoch=pd.to_datetime(log["epoch"]))
    out = convergence_epoch(raw, patience=1)
    assert set(out["policy"]) == {"thompson", "historical"}
    assert cumulative_reward(raw)["epoch"].tolist()[:2] == ["2015-01", "2015-02"]


def test_all_metrics_cope_with_empty_logs(log):
    empty = log.iloc[:0]
    ctx = pd.DataFrame({"loan_id": [], "grade": []})
    outputs = {
        "cumulative_reward": cumulative_reward(empty),
        "cumulative_regret": cumulative_regret(empty),
        "rolling_reward": rolling_reward(empty),
        "selection_frequency": selection_frequency(empty, SPACE),
        "decline_rate": decline_rate(empty),
        "profitability": profitability_vs_historical(empty),
        "inclusion": inclusion_metrics(empty, ctx),
        "convergence": convergence_epoch(empty),
    }
    for name, out in outputs.items():
        assert out.empty and len(out.columns) > 0, name
    assert list(cumulative_reward(pd.DataFrame()).columns)[:2] == ["policy", "epoch"]
    assert cumulative_regret(None).empty  # type: ignore[arg-type]


def test_missing_required_columns_raise(log):
    with pytest.raises(KeyError):
        cumulative_regret(log.drop(columns=["oracle_reward"]))
    assert np.isfinite(cumulative_reward(log.drop(columns=["realized_reward"]))["cum_reward"]).all()
