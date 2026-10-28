"""Policy-performance metrics (Table 11: policy & bandit performance, counterfactual
comparison, financial inclusion; SPEC §2.12).

Every function takes the per-decision log produced by the simulation — one row per
(policy, borrower) with columns

    epoch ('YYYY-MM'), policy, loan_id, group, arm, amount, rate, term, declined (bool),
    expected_reward, oracle_reward, realized_reward (nullable), fallback (bool)

and returns a tidy DataFrame (empty, with the documented columns, for an empty log).

* reward: ``expected_reward`` is the model-based expected risk-adjusted reward of the
  chosen arm; ``realized_reward`` is the observed reward where known (replay/hybrid).
* regret: per decision ``oracle_reward − expected_reward`` (pseudo-regret against the
  hindsight oracle arm, §2.11), summed per epoch and cumulated.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..data.arms import ArmSpace

LOG_COLUMNS = (
    "epoch",
    "policy",
    "loan_id",
    "group",
    "arm",
    "amount",
    "rate",
    "term",
    "declined",
    "expected_reward",
    "oracle_reward",
    "realized_reward",
    "fallback",
)

CUMULATIVE_REWARD_COLUMNS = ("policy", "epoch", "n", "reward", "cum_reward", "realized", "cum_realized")
CUMULATIVE_REGRET_COLUMNS = (
    "policy",
    "epoch",
    "n",
    "regret",
    "avg_regret",
    "cum_n",
    "cum_regret",
    "cum_avg_regret",
)
ROLLING_REWARD_COLUMNS = ("policy", "epoch", "n", "mean_reward", "rolling_mean_reward")
SELECTION_COLUMNS = ("policy", "arm", "term", "amount", "rate", "count", "share")
DECLINE_COLUMNS = ("policy", "epoch", "n", "n_declined", "decline_rate")
PROFIT_COLUMNS = ("policy", "n", "total_expected", "total_realized", "uplift_vs_historical", "uplift_pct")
CONVERGENCE_COLUMNS = ("policy", "convergence_epoch", "final_value", "converged")

_NUMERIC = ("arm", "amount", "rate", "term", "expected_reward", "oracle_reward", "realized_reward")


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


def _is_empty(df: pd.DataFrame | None) -> bool:
    return df is None or len(df) == 0


def _epoch_strings(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m")
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.dt.strftime("%Y-%m")
    return s.astype(str)


def _prepare(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"decision log is missing columns {missing}")
    out = df.copy()
    if "epoch" in out.columns:
        out["epoch"] = _epoch_strings(out["epoch"])
    if "policy" in out.columns:
        out["policy"] = out["policy"].astype(str)
    for c in _NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in ("declined", "fallback"):
        if c in out.columns:
            out[c] = out[c].fillna(False).astype(bool)
    if "declined" not in out.columns and "amount" in out.columns:
        out["declined"] = ~(out["amount"] > 0)
    return out


def _cumsum_by_policy(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("policy", sort=False)[col].cumsum()


# --------------------------------------------------------------------------------------
# Reward / regret curves
# --------------------------------------------------------------------------------------
def cumulative_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Per policy and epoch: decisions ``n``, expected ``reward`` (sum) and its cumulative
    sum, and the ``realized`` reward where known (unknown rewards contribute 0)."""
    if _is_empty(df):
        return _empty(CUMULATIVE_REWARD_COLUMNS)
    d = _prepare(df, ["policy", "epoch", "expected_reward"])
    d["_realized"] = d["realized_reward"] if "realized_reward" in d.columns else np.nan
    out = (
        d.groupby(["policy", "epoch"], sort=True)
        .agg(
            n=("expected_reward", "size"),
            reward=("expected_reward", "sum"),
            realized=("_realized", "sum"),
        )
        .reset_index()
    )
    out["cum_reward"] = _cumsum_by_policy(out, "reward")
    out["cum_realized"] = _cumsum_by_policy(out, "realized")
    return out.loc[:, list(CUMULATIVE_REWARD_COLUMNS)]


def cumulative_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-regret ``oracle_reward − expected_reward`` per policy and epoch.

    ``n`` counts decisions with a defined regret; ``avg_regret`` is the epoch mean,
    ``cum_avg_regret = cum_regret / cum_n`` the running average regret per decision.
    """
    if _is_empty(df):
        return _empty(CUMULATIVE_REGRET_COLUMNS)
    d = _prepare(df, ["policy", "epoch", "expected_reward", "oracle_reward"])
    d["_regret"] = d["oracle_reward"] - d["expected_reward"]
    out = (
        d.groupby(["policy", "epoch"], sort=True)
        .agg(n=("_regret", "count"), regret=("_regret", "sum"))
        .reset_index()
    )
    out["avg_regret"] = out["regret"] / out["n"].where(out["n"] > 0)
    out["cum_n"] = _cumsum_by_policy(out, "n")
    out["cum_regret"] = _cumsum_by_policy(out, "regret")
    out["cum_avg_regret"] = out["cum_regret"] / out["cum_n"].where(out["cum_n"] > 0)
    return out.loc[:, list(CUMULATIVE_REGRET_COLUMNS)]


def rolling_reward(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Mean expected reward per decision, per epoch and over a rolling ``window`` of
    epochs (decision-weighted: rolling reward sum / rolling decision count)."""
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if _is_empty(df):
        return _empty(ROLLING_REWARD_COLUMNS)
    cr = cumulative_reward(df)
    cr["mean_reward"] = cr["reward"] / cr["n"]
    grouped = cr.groupby("policy", sort=False)
    roll_sum = grouped["reward"].transform(lambda s: s.rolling(int(window), min_periods=1).sum())
    roll_n = grouped["n"].transform(lambda s: s.rolling(int(window), min_periods=1).sum())
    cr["rolling_mean_reward"] = roll_sum / roll_n
    return cr.loc[:, list(ROLLING_REWARD_COLUMNS)]


# --------------------------------------------------------------------------------------
# Selection behaviour
# --------------------------------------------------------------------------------------
def selection_frequency(df: pd.DataFrame, arm_space: ArmSpace) -> pd.DataFrame:
    """Arm selection counts and shares per policy over the full arm grid (zeros kept), with
    the arm's ``term, amount, rate`` so it pivots directly into amount × rate heatmaps per
    term (the null arm appears as term = amount = rate = 0). Shares are relative to all of
    the policy's decisions; arms outside the arm space are ignored."""
    if _is_empty(df):
        return _empty(SELECTION_COLUMNS)
    d = _prepare(df, ["policy", "arm"])
    arms = arm_space.to_frame().loc[:, ["arm", "term", "amount", "rate"]]
    policies = sorted(d["policy"].unique())
    counts = d.dropna(subset=["arm"]).astype({"arm": np.int64}).groupby(["policy", "arm"]).size()
    grid = pd.MultiIndex.from_product([policies, arms["arm"].to_numpy()], names=["policy", "arm"])
    out = counts.reindex(grid, fill_value=0).rename("count").reset_index()
    out = out.merge(arms, on="arm", how="left")
    totals = d.groupby("policy").size()
    out["share"] = out["count"] / out["policy"].map(totals).astype(float)
    out["count"] = out["count"].astype(np.int64)
    return out.loc[:, list(SELECTION_COLUMNS)]


def decline_rate(df: pd.DataFrame, by_epoch: bool = True) -> pd.DataFrame:
    """Share of declined borrowers per policy (and epoch when ``by_epoch``).

    ``declined`` is derived from ``amount <= 0`` when the column is absent.
    """
    cols = list(DECLINE_COLUMNS) if by_epoch else [c for c in DECLINE_COLUMNS if c != "epoch"]
    if _is_empty(df):
        return _empty(cols)
    keys = ["policy", "epoch"] if by_epoch else ["policy"]
    d = _prepare(df, keys)
    if "declined" not in d.columns:
        raise KeyError("decision log needs 'declined' or 'amount' to compute decline rates")
    out = (
        d.groupby(keys, sort=True)
        .agg(n=("declined", "size"), n_declined=("declined", "sum"))
        .reset_index()
    )
    out["n_declined"] = out["n_declined"].astype(np.int64)
    out["decline_rate"] = out["n_declined"] / out["n"]
    return out.loc[:, cols]


# --------------------------------------------------------------------------------------
# Profitability & inclusion
# --------------------------------------------------------------------------------------
def profitability_vs_historical(
    df: pd.DataFrame, historical_policy_name: str = "historical"
) -> pd.DataFrame:
    """Total expected and realized-where-known reward per policy, and the uplift of the
    total expected reward over the historical (reference) policy.

    ``uplift_pct = 100 · uplift / |historical total|`` (NaN when the historical policy is
    absent from the log or its total is 0). Uplift uses expected rewards because realized
    rewards are only partially observed.
    """
    if _is_empty(df):
        return _empty(PROFIT_COLUMNS)
    d = _prepare(df, ["policy", "expected_reward"])
    d["_realized"] = d["realized_reward"] if "realized_reward" in d.columns else np.nan
    out = (
        d.groupby("policy", sort=True)
        .agg(
            n=("expected_reward", "size"),
            total_expected=("expected_reward", "sum"),
            total_realized=("_realized", "sum"),
        )
        .reset_index()
    )
    hist = out.loc[out["policy"] == str(historical_policy_name), "total_expected"]
    base = float(hist.iloc[0]) if len(hist) else float("nan")
    out["uplift_vs_historical"] = out["total_expected"] - base
    denom = abs(base) if np.isfinite(base) and base != 0 else float("nan")
    out["uplift_pct"] = 100.0 * out["uplift_vs_historical"] / denom
    return out.loc[:, list(PROFIT_COLUMNS)]


def inclusion_metrics(df: pd.DataFrame, context_df: pd.DataFrame, by: str = "grade") -> pd.DataFrame:
    """Financial-inclusion table per policy and borrower segment ``by`` (e.g. grade).

    ``by`` is taken from the log if present, else joined from ``context_df`` on
    ``loan_id``. ``approval_share`` = share not declined; ``mean_amount``/``mean_rate``
    over approved decisions (the offered terms); ``mean_expected_reward`` over all
    decisions (declines contribute their reward, normally 0). Missing segments are kept.
    """
    cols = ["policy", by, "n", "approval_share", "mean_amount", "mean_rate", "mean_expected_reward"]
    if _is_empty(df):
        return _empty(cols)
    d = _prepare(df, ["policy", "expected_reward"])
    if by not in d.columns:
        if context_df is None or by not in context_df.columns or "loan_id" not in context_df.columns:
            raise KeyError(f"'{by}' is neither in the decision log nor in context_df (with loan_id)")
        if "loan_id" not in d.columns:
            raise KeyError("decision log needs 'loan_id' to join the context")
        ctx = context_df.loc[:, ["loan_id", by]].drop_duplicates("loan_id")
        left_key, right_key = d["loan_id"], ctx["loan_id"]
        if left_key.dtype != right_key.dtype:
            d = d.assign(loan_id=left_key.astype(str))
            ctx = ctx.assign(loan_id=right_key.astype(str))
        d = d.merge(ctx, on="loan_id", how="left")
    if "declined" not in d.columns:
        raise KeyError("decision log needs 'declined' or 'amount'")
    approved = ~d["declined"]
    d["_approved"] = approved.astype(float)
    d["_amount"] = d["amount"].where(approved) if "amount" in d.columns else np.nan
    d["_rate"] = d["rate"].where(approved) if "rate" in d.columns else np.nan
    out = (
        d.groupby(["policy", by], sort=True, dropna=False)
        .agg(
            n=("expected_reward", "size"),
            approval_share=("_approved", "mean"),
            mean_amount=("_amount", "mean"),
            mean_rate=("_rate", "mean"),
            mean_expected_reward=("expected_reward", "mean"),
        )
        .reset_index()
    )
    return out.loc[:, cols]


# --------------------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------------------
def convergence_epoch(
    regret_df: pd.DataFrame,
    tol: float = 0.05,
    patience: int = 3,
    column: str = "avg_regret",
    absolute: bool = False,
) -> pd.DataFrame:
    """First epoch after which a per-epoch curve stays within ``tol`` of its final value.

    Works on :func:`cumulative_regret` output (``column="avg_regret"``, the per-epoch
    average regret; a raw decision log is converted first) or any per-policy, per-epoch
    frame (e.g. ``rolling_reward`` with ``column="rolling_mean_reward"`` for reward
    convergence). **[decision]** The final value is the mean of the last ``patience``
    epochs (robust to a noisy last epoch); the band is ``tol × max|curve|`` (relative to
    the curve's peak magnitude, so it stays meaningful when the final value is ≈ 0), or
    ``tol`` itself with ``absolute=True``. A policy has converged when the curve stays in
    the band for at least ``patience`` epochs up to the end; ``convergence_epoch`` is the
    first epoch of that final run (None otherwise).
    """
    if int(patience) < 1:
        raise ValueError("patience must be >= 1")
    if _is_empty(regret_df):
        return _empty(CONVERGENCE_COLUMNS)
    frame = regret_df
    if column not in frame.columns and {"oracle_reward", "expected_reward"} <= set(frame.columns):
        frame = cumulative_regret(frame)
    if column not in frame.columns or "policy" not in frame.columns or "epoch" not in frame.columns:
        raise KeyError(f"convergence_epoch needs columns policy, epoch and {column!r}")
    frame = frame.assign(epoch=_epoch_strings(frame["epoch"]), policy=frame["policy"].astype(str))
    rows = []
    for policy, grp in frame.sort_values(["policy", "epoch"]).groupby("policy", sort=True):
        vals = pd.to_numeric(grp[column], errors="coerce").to_numpy(dtype=float)
        epochs = grp["epoch"].to_numpy()
        keep = np.isfinite(vals)
        vals, epochs = vals[keep], epochs[keep]
        if vals.size == 0:
            rows.append({"policy": policy, "convergence_epoch": None, "final_value": np.nan, "converged": False})
            continue
        final = float(np.mean(vals[-int(patience):]))
        scale = float(np.max(np.abs(vals)))
        band = float(tol) if absolute or scale == 0.0 else float(tol) * scale
        outside = np.flatnonzero(np.abs(vals - final) > band)
        start = int(outside[-1]) + 1 if outside.size else 0
        converged = (vals.size - start) >= int(patience)
        rows.append(
            {
                "policy": policy,
                "convergence_epoch": str(epochs[start]) if converged else None,
                "final_value": final,
                "converged": bool(converged),
            }
        )
    # explicit dtypes: pandas' string inference would turn a None epoch into NaN
    return pd.DataFrame(
        {
            "policy": pd.Series([r["policy"] for r in rows], dtype=object),
            "convergence_epoch": pd.Series([r["convergence_epoch"] for r in rows], dtype=object),
            "final_value": pd.Series([r["final_value"] for r in rows], dtype=float),
            "converged": pd.Series([r["converged"] for r in rows], dtype=bool),
        }
    )


# ----------------------------------------------------------------------------------------
# recommendation / selection mix (makes "largest loan at the highest rate" visible)
# ----------------------------------------------------------------------------------------
#: ``(low, high, label)`` rate buckets (fractions, half-open ``[low, high)``).
RATE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.10, "<10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, 0.20, "15-20%"),
    (0.20, 0.25, "20-25%"),
    (0.25, float("inf"), ">=25%"),
)
#: ``(low, high, label)`` amount buckets (USD, half-open).
AMOUNT_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 10_000.0, "<$10k"),
    (10_000.0, 20_000.0, "$10k-20k"),
    (20_000.0, 30_000.0, "$20k-30k"),
    (30_000.0, float("inf"), ">=$30k"),
)
_BUCKET_EPS = 1e-9


def _shares(values: np.ndarray, buckets: Sequence[tuple[float, float, str]]) -> dict[str, float | None]:
    n = values.size
    out: dict[str, float | None] = {}
    for lo, hi, label in buckets:
        inside = (values >= lo - _BUCKET_EPS) & (values < hi - _BUCKET_EPS)
        out[label] = float(inside.sum() / n) if n else None
    return out


def arm_mix(
    amount: Sequence[float] | np.ndarray,
    rate: Sequence[float] | np.ndarray,
    term: Sequence[float] | np.ndarray,
    lending: Sequence[bool] | np.ndarray | None = None,
    *,
    max_rate: float | None = None,
    max_amount: float | None = None,
) -> dict[str, object]:
    """Distribution of lending decisions over term, rate bucket and amount bucket.

    ``lending`` marks the lending decisions (default ``amount > 0``); shares are among
    them. Returns ``n, n_lending, decline_share, by_term {"36": s, "60": s}, by_rate
    (:data:`RATE_BUCKETS`), by_amount (:data:`AMOUNT_BUCKETS`), mean_rate, mean_amount,
    share_36m, share_at_max_rate, share_at_max_amount`` (the last two need ``max_rate`` /
    ``max_amount``, e.g. the top of the arm grid). Empty inputs give ``None`` shares.
    """
    a = np.asarray(amount, dtype=float).reshape(-1)
    r = np.asarray(rate, dtype=float).reshape(-1)
    t = np.asarray(term, dtype=float).reshape(-1)
    lend = (a > 0) if lending is None else np.asarray(lending, dtype=bool).reshape(-1)
    n = int(a.size)
    a, r, t = a[lend], r[lend], t[lend]
    nl = int(a.size)
    terms = sorted({int(x) for x in t.tolist() if np.isfinite(x)} | {36, 60})
    out: dict[str, object] = {
        "n": n,
        "n_lending": nl,
        "decline_share": float(1.0 - nl / n) if n else None,
        "by_term": {str(k): (float(np.mean(t == k)) if nl else None) for k in terms},
        "by_rate": _shares(r, RATE_BUCKETS),
        "by_amount": _shares(a, AMOUNT_BUCKETS),
        "mean_rate": float(r.mean()) if nl else None,
        "mean_amount": float(a.mean()) if nl else None,
        "share_36m": float(np.mean(t == 36)) if nl else None,
        "share_at_max_rate": float(np.mean(r >= max_rate - _BUCKET_EPS)) if nl and max_rate is not None else None,
        "share_at_max_amount": float(np.mean(a >= max_amount - _BUCKET_EPS)) if nl and max_amount is not None else None,
    }
    return out
