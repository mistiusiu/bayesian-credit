"""Dataset read side: Table 3 variables, feature summaries and the paginated loan browser.

Loans are read straight from ``loans.parquet`` with DuckDB (``SELECT … LIMIT/OFFSET``) so a
page of the 2.26 M-row Lending Club table never loads the whole frame into pandas.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from bayesian_credit.config import RewardConfig
from bayesian_credit.data.schema import PRE_DECISION_COLUMNS, RETAINED_VARIABLES, VARIABLES_BY_NAME
from bayesian_credit.rewards.compound import CompoundRewardFunction
from rest_framework.exceptions import ValidationError

from apps.common.utils import clean, epoch_range, num, read_json, to_epoch, write_json_atomic

from .models import Dataset

DEFAULT_SUMMARY_FEATURES: tuple[str, ...] = (
    "fico_range_low",
    "dti",
    "annual_inc",
    "revol_util",
    "inq_last_6mths",
    "emp_length",
    "loan_amnt",
    "int_rate",
    "grade",
    "purpose",
    "home_ownership",
    "verification_status",
    "term",
)
#: Numeric columns summarised as categories (few distinct values).
CATEGORICAL_NUMERIC = {"term"}
MAX_LEVELS = 12
N_BINS = 20

FEATURE_UNITS: dict[str, str] = {
    "annual_inc": "usd",
    "annual_inc_joint": "usd",
    "dti": "pct",
    "dti_joint": "pct",
    "emp_length": "years",
    "fico_range_low": "score",
    "fico_range_high": "score",
    "delinq_2yrs": "count",
    "delinq_amnt": "usd",
    "inq_last_6mths": "count",
    "inq_last_12m": "count",
    "open_acc": "count",
    "total_acc": "count",
    "revol_bal": "usd",
    "revol_util": "fraction",
    "installment": "usd",
    "mort_acc": "count",
    "pub_rec": "count",
    "pub_rec_bankruptcies": "count",
    "collections_12_mths_ex_med": "count",
    "num_accts_ever_120_pd": "count",
    "pct_tl_nvr_dlq": "fraction",
    "num_sats": "count",
    "tot_coll_amt": "usd",
    "tot_cur_bal": "usd",
    "tot_hi_cred_lim": "usd",
    "total_bal_ex_mort": "usd",
    "total_bc_limit": "usd",
    "bc_util": "fraction",
    "all_util": "fraction",
    "loan_amnt": "usd",
    "int_rate": "fraction",
    "term": "months",
    "funded_amnt": "usd",
    "last_fico_range_low": "score",
    "last_fico_range_high": "score",
    "last_pymnt_amnt": "usd",
    "out_prncp": "usd",
    "out_prncp_inv": "usd",
    "num_tl_30dpd": "count",
    "num_tl_90g_dpd_24m": "count",
    "mths_since_last_delinq": "months",
    "recoveries": "usd",
    "collection_recovery_fee": "usd",
    "total_pymnt": "usd",
    "total_pymnt_inv": "usd",
    "total_rec_prncp": "usd",
    "total_rec_int": "usd",
    "total_rec_late_fee": "usd",
}

OUTCOME_SELECT = (
    "loan_status",
    "y_resolved",
    "y_default",
    "y_recovery_rate",
    "y_realized_revenue",
    "y_realized_loss",
    "y_months_on_book",
    "y_resolution_date",
)
LOAN_COLUMNS: tuple[str, ...] = (
    "loan_id",
    "issue_d",
    "funded_amnt",
    "y_group",
    "y_historical_arm",
    *PRE_DECISION_COLUMNS,
    "loan_amnt",
    "int_rate",
    "term",
    *OUTCOME_SELECT,
)
ORDERING_FIELDS = (
    "loan_id",
    "issue_d",
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "term",
    "grade",
    "sub_grade",
    "fico_range_low",
    "annual_inc",
    "dti",
)

_REWARD_FN = CompoundRewardFunction(RewardConfig())


# ----------------------------------------------------------------------------- variables
def dtype_name(dtype: Any) -> str:
    s = str(dtype)
    if s in {"str", "string", "string[python]", "string[pyarrow]"} or (
        pd.api.types.is_string_dtype(dtype) and s != "object"
    ):
        return "string"
    return s


def variable_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """types.ts ``VariableRow`` for every Table 3 variable, in Table 3 order."""
    rows = []
    n = len(df)
    for v in RETAINED_VARIABLES:
        in_table = v.name in df.columns
        col = df[v.name] if in_table else None
        null_share = float(col.isna().mean()) if (in_table and n) else (0.0 if in_table else None)
        rows.append(
            {
                "name": v.name,
                "stage": v.stage.value,
                "primary_use": v.primary_use,
                "kind": v.kind,
                "present": bool(in_table and n and col.notna().any()),
                "dtype": dtype_name(col.dtype) if in_table else None,
                "null_share": num(null_share, 6),
            }
        )
    return rows


# ----------------------------------------------------------------------------- summary
def _r(x: float, digits: int = 6) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def numeric_summary(name: str, series: pd.Series) -> dict[str, Any]:
    var = VARIABLES_BY_NAME.get(name)
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = x[np.isfinite(x)]
    total = int(x.size)
    out: dict[str, Any] = {
        "feature": name,
        "kind": "numeric",
        "stage": var.stage.value if var else "pre_decision",
        "unit": FEATURE_UNITS.get(name, "none"),
        "n": int(finite.size),
        "null_share": _r(1.0 - finite.size / total) if total else 0.0,
        "mean": None,
        "std": None,
        "p05": None,
        "p50": None,
        "p95": None,
        "bins": [],
    }
    if finite.size == 0:
        return out
    p01, p05, p50, p95, p99 = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
    lo, hi = float(p01), float(p99)
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    integral = bool(np.all(np.equal(np.mod(finite, 1.0), 0.0)))
    if integral and hi - lo <= 30:
        edges = np.arange(math.floor(lo) - 0.5, math.ceil(hi) + 1.0, 1.0)
    else:
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, N_BINS + 1)
    counts, _ = np.histogram(np.clip(finite, edges[0], edges[-1]), bins=edges)
    out.update(
        mean=_r(float(finite.mean())),
        std=_r(float(finite.std(ddof=1))) if finite.size > 1 else 0.0,
        p05=_r(float(p05)),
        p50=_r(float(p50)),
        p95=_r(float(p95)),
        bins=[
            {"lo": _r(float(edges[i])), "hi": _r(float(edges[i + 1])), "count": int(counts[i])}
            for i in range(len(counts))
        ],
    )
    return out


def categorical_summary(name: str, series: pd.Series) -> dict[str, Any]:
    var = VARIABLES_BY_NAME.get(name)
    if pd.api.types.is_numeric_dtype(series.dtype):
        series = series.map(lambda v: None if pd.isna(v) else f"{v:g}")
    s = series.astype("string")
    missing = s.isna()
    total = int(len(s))
    vc = s[~missing].value_counts()
    levels = [(str(k), int(v)) for k, v in vc.items()]
    if len(levels) > MAX_LEVELS:
        head, tail = levels[: MAX_LEVELS - 1], levels[MAX_LEVELS - 1 :]
        rest = sum(c for _, c in tail)
        merged = dict(head)
        merged["other"] = merged.get("other", 0) + rest
        levels = sorted(merged.items(), key=lambda kv: -kv[1])
    return {
        "feature": name,
        "kind": "categorical",
        "stage": var.stage.value if var else "pre_decision",
        "n": int((~missing).sum()),
        "null_share": _r(float(missing.mean())) if total else 0.0,
        "levels": [{"level": k, "count": c} for k, c in levels],
    }


def feature_summary(name: str, series: pd.Series) -> dict[str, Any]:
    var = VARIABLES_BY_NAME.get(name)
    if name in CATEGORICAL_NUMERIC or (var is not None and var.kind == "categorical"):
        return categorical_summary(name, series)
    return numeric_summary(name, series)


def validate_features(features: Sequence[str]) -> list[str]:
    unknown = [f for f in features if f not in VARIABLES_BY_NAME or VARIABLES_BY_NAME[f].kind == "date"]
    if unknown:
        raise ValidationError({"features": [f"unknown or non-summarisable feature(s): {', '.join(unknown)}"]})
    return list(dict.fromkeys(features))


def build_summary(ds: Dataset, features: Sequence[str] | None = None) -> dict[str, Any]:
    """types.ts ``DatasetSummary``; cached as JSON next to the parquet (per feature set)."""
    feats = list(DEFAULT_SUMMARY_FEATURES) if not features else validate_features(features)
    key = "default" if not features else hashlib.sha1(",".join(feats).encode()).hexdigest()[:12]
    cache = ds.directory / f"summary_{key}.json"
    cached = read_json(cache)
    if cached is not None:
        return cached
    available = set(parquet_columns(ds.parquet))
    cols = [c for c in dict.fromkeys([*feats, "issue_d", "grade", "term", "y_resolved", "y_default"]) if c in available]
    df = pd.read_parquet(ds.parquet, columns=cols)
    issue = pd.to_datetime(df["issue_d"])
    month = issue.dt.to_period("M").astype(str)
    resolved = df["y_resolved"].fillna(False).astype(bool) if "y_resolved" in df else pd.Series(False, index=df.index)
    default = (
        (pd.to_numeric(df["y_default"], errors="coerce") == 1.0)
        if "y_default" in df
        else pd.Series(False, index=df.index)
    )
    per = (
        pd.DataFrame({"m": month, "r": resolved.to_numpy(), "d": default.to_numpy()})
        .groupby("m")
        .agg(n=("r", "size"), n_resolved=("r", "sum"), n_default=("d", "sum"))
    )
    arrivals = []
    if len(df):
        lo, hi = to_epoch(issue.min()), to_epoch(issue.max())
        for epoch in epoch_range(lo, hi):
            row = per.loc[epoch] if epoch in per.index else None
            arrivals.append(
                {
                    "epoch": epoch,
                    "n": int(row["n"]) if row is not None else 0,
                    "n_resolved": int(row["n_resolved"]) if row is not None else 0,
                    "n_default": int(row["n_default"]) if row is not None else 0,
                }
            )
    grade_counts = (
        {str(k): int(v) for k, v in df["grade"].astype("string").value_counts().sort_index().items()}
        if "grade" in df
        else {}
    )
    term_counts = (
        {f"{int(k)}": int(v) for k, v in df["term"].dropna().value_counts().sort_index().items()}
        if "term" in df
        else {}
    )
    out = {
        "dataset": ds.pk,
        "n_loans": int(len(df)),
        "features": [feature_summary(f, df[f]) if f in df.columns else _absent_feature(f) for f in feats],
        "monthly_arrivals": arrivals,
        "status_counts": dict(ds.status_counts or {}),
        "grade_counts": grade_counts,
        "term_counts": term_counts,
    }
    out = clean(out)
    write_json_atomic(cache, out)
    return out


def _absent_feature(name: str) -> dict[str, Any]:
    return feature_summary(name, pd.Series([np.nan], dtype=float).iloc[:0])


# ----------------------------------------------------------------------------- parquet access
def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def read_loans_by_id(ds: Dataset, loan_ids: Iterable[int], columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows for the given loan ids with the parquet's own (canonical) dtypes."""
    ids = [int(i) for i in loan_ids]
    if not ids:
        return pd.read_parquet(ds.parquet, columns=list(columns) if columns else None).iloc[:0]
    return pd.read_parquet(
        ds.parquet, columns=list(columns) if columns else None, filters=[("loan_id", "in", ids)]
    ).reset_index(drop=True)


def query_loans(
    ds: Dataset,
    *,
    grade: Sequence[str] = (),
    purpose: Sequence[str] = (),
    home_ownership: Sequence[str] = (),
    group: str | None = None,
    resolved: bool | None = None,
    search: str | None = None,
    ordering: str = "loan_id",
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, pd.DataFrame]:
    """``(count, page)`` for the loan browser."""
    where, params = [], []
    for col, values in (("grade", grade), ("purpose", purpose), ("home_ownership", home_ownership)):
        if values:
            where.append(f"{col} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
    if group:
        where.append("y_group = ?")
        params.append(group)
    if resolved is not None:
        where.append("coalesce(y_resolved, false) = ?")
        params.append(bool(resolved))
    if search:
        where.append("contains(CAST(loan_id AS VARCHAR), ?)")
        params.append(str(search).strip())
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    desc = ordering.startswith("-")
    field = ordering.lstrip("-")
    if field not in ORDERING_FIELDS:
        raise ValidationError({"ordering": [f"must be one of {', '.join(ORDERING_FIELDS)} (prefix - for descending)"]})
    order = f"{field} {'DESC' if desc else 'ASC'} NULLS LAST" + ("" if field == "loan_id" else ", loan_id ASC")
    available = set(parquet_columns(ds.parquet))
    cols = ", ".join(c for c in LOAN_COLUMNS if c in available)
    src = f"read_parquet({_sql_path(ds.parquet)})"
    con = duckdb.connect()
    try:
        count = int(con.execute(f"SELECT count(*) FROM {src} {clause}", params).fetchone()[0])
        page = con.execute(
            f"SELECT {cols} FROM {src} {clause} ORDER BY {order} LIMIT ? OFFSET ?", [*params, int(limit), int(offset)]
        ).df()
    finally:
        con.close()
    return count, page


# ----------------------------------------------------------------------------- loan records
def _val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def context_from_row(row: pd.Series | dict) -> dict[str, Any]:
    """types.ts ``BorrowerContext`` — exactly the 37 PRE_DECISION columns (null when absent)."""
    get = row.get
    out = {}
    for col in PRE_DECISION_COLUMNS:
        v = _val(get(col))
        if isinstance(v, str):
            out[col] = v
        elif v is None:
            out[col] = None
        else:
            out[col] = float(v) if not isinstance(v, bool) else v
    return out


def outcome_from_row(row: pd.Series | dict, realized_reward: float | None) -> dict[str, Any]:
    get = row.get
    resolved = bool(_val(get("y_resolved")) or False)
    y_default = _val(get("y_default"))
    return {
        "loan_status": str(_val(get("loan_status")) or ""),
        "resolved": resolved,
        "defaulted": (bool(y_default == 1.0) if (resolved and y_default is not None) else None),
        "recovery_rate": num(_val(get("y_recovery_rate"))),
        "realized_revenue": num(_val(get("y_realized_revenue"))),
        "realized_loss": num(_val(get("y_realized_loss"))),
        "realized_reward": num(realized_reward) if resolved else None,
        "months_on_book": num(_val(get("y_months_on_book"))),
        "resolution_epoch": to_epoch(_val(get("y_resolution_date"))),
    }


def realized_rewards(frame: pd.DataFrame) -> np.ndarray:
    """Realized reward of the historical policy (revenue − loss − funding on the realized schedule)."""
    if frame.empty:
        return np.zeros(0)
    return _REWARD_FN.terminal_many(frame)


def loan_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """types.ts ``LoanRecord`` per row of a LoanTable slice."""
    rewards = realized_rewards(frame)
    out = []
    for i, (_, row) in enumerate(frame.iterrows()):
        arm = _val(row.get("y_historical_arm"))
        out.append(
            {
                "loan_id": int(row["loan_id"]),
                "issue_d": to_epoch(row.get("issue_d")),
                "funded_amnt": num(_val(row.get("funded_amnt"))),
                "group": str(_val(row.get("y_group")) or ""),
                "historical_arm": int(arm) if arm is not None else -1,
                "context": context_from_row(row),
                "policy": {
                    "loan_amnt": num(_val(row.get("loan_amnt"))),
                    "int_rate": num(_val(row.get("int_rate"))),
                    "term": int(_val(row.get("term")) or 0),
                },
                "outcome": outcome_from_row(row, float(rewards[i]) if i < len(rewards) else None),
            }
        )
    return out
