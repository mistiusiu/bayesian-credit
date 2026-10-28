"""Data layer (SPEC §1, Appendix F): schema & leakage guard, arm space, segmentation,
Lending Club loader, synthetic generator, outcome derivation and the MCTM state panel.

``load_loans(source, **kw)`` is the single entry point used by scripts and the API::

    df = load_loans("synthetic", n=50_000, seed=7, shift_at="2016-06")
    df = load_loans("lending_club", sample_fraction=0.09, seed=42, with_outcomes=True)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import DataConfig
from ..exceptions import DataSourceError
from .arms import NULL_POLICY, ArmSpace, LendingPolicy, annuity_installment
from .loader import (
    CANONICAL_COLUMNS,
    DEFAULT_FILENAME,
    KAGGLE_DATASET,
    STRING_DTYPE,
    LendingClubLoader,
    canonicalize,
    default_lending_club_path,
    sample_bucket,
    subsample,
    validate_loan_table,
)
from .outcomes import (
    add_months,
    derive_outcomes,
    months_between,
    recovery_rate,
    summarize,
    temporal_split,
)
from .panel import LoanStatePanel, interim_state, paths_to_counts, row_segments
from .reference_rates import ReferenceRates
from .schema import (
    ALL_RETAINED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    ID_COLUMN,
    INTERIM_COLUMNS,
    NUMERIC_COLUMNS,
    OUTCOME_COLUMNS,
    POLICY_COLUMNS,
    PRE_DECISION_COLUMNS,
    RETAINED_VARIABLES,
    STATUS_CURRENT,
    STATUS_DEFAULTED,
    STATUS_DELINQUENT,
    STATUS_FULLY_PAID,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED,
    TERMINAL_COLUMNS,
    BorrowerContext,
    LeakageGuard,
    Variable,
)
from .segmentation import Segmenter
from .synthetic import DEFAULT_SHIFT, DEFAULT_SHIFT_AT, SyntheticLendingClub
from .synthetic import generate as generate_synthetic

SOURCES = ("synthetic", "lending_club")
_SYNTHETIC_KEYS = {"n", "seed", "start", "end", "snapshot", "shift_at", "shift"}
_LC_KEYS = {"path", "cache", "cache_dir", "max_rows", "sample_fraction", "seed", "issue_from", "issue_to", "columns", "order"}


def load_loans(
    source: str = "synthetic",
    *,
    with_outcomes: bool = False,
    arm_space: ArmSpace | None = None,
    segmenter: Segmenter | None = None,
    data_cfg: DataConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Canonical LoanTable from ``"synthetic"`` or ``"lending_club"`` (optionally + ``y_*``).

    synthetic: ``n`` (default 100_000; ``max_rows`` is accepted as an alias), ``seed``,
    ``start``, ``end``, ``snapshot`` (default ``data_cfg.snapshot_date``), ``shift_at``,
    ``shift``; ``issue_from`` / ``issue_to`` / ``sample_fraction`` filter the result.
    lending_club: ``path``, ``cache``, ``cache_dir``, ``max_rows``, ``sample_fraction``,
    ``seed``, ``issue_from``, ``issue_to``, ``columns``, ``order`` (see
    :meth:`LendingClubLoader.load`). Keyword arguments set to ``None`` are ignored, so
    callers can pass one parameter dict for both sources.
    """
    cfg = data_cfg or DataConfig()
    kw = {k: v for k, v in kwargs.items() if v is not None}
    src = source.lower().replace("-", "_")
    if src in ("synthetic", "synth"):
        if "max_rows" in kw and "n" not in kw:
            kw["n"] = kw.pop("max_rows")
        issue_from = kw.pop("issue_from", None)
        issue_to = kw.pop("issue_to", None)
        fraction = kw.pop("sample_fraction", None)
        unknown = set(kw) - _SYNTHETIC_KEYS
        if unknown:
            raise DataSourceError(f"unsupported arguments for source='synthetic': {sorted(unknown)}")
        gen = SyntheticLendingClub(seed=int(kw.pop("seed", 0)), chargeoff_lag_months=cfg.chargeoff_lag_months)
        df = gen.generate(int(kw.pop("n", 100_000)), snapshot=kw.pop("snapshot", cfg.snapshot_date), **kw)
        if issue_from is not None:
            df = df.loc[df["issue_d"] >= pd.Timestamp(issue_from)].reset_index(drop=True)
        if issue_to is not None:
            df = df.loc[df["issue_d"] <= pd.Timestamp(issue_to)].reset_index(drop=True)
        if fraction is not None:
            df = subsample(df, float(fraction), seed=gen.seed)
        df.attrs["synthetic"] = dict(gen.last_meta)
    elif src in ("lending_club", "lc", "kaggle"):
        unknown = set(kw) - _LC_KEYS
        if unknown:
            raise DataSourceError(f"unsupported arguments for source='lending_club': {sorted(unknown)}")
        loader = LendingClubLoader(
            kw.pop("path", None), cfg, cache=kw.pop("cache", True), cache_dir=kw.pop("cache_dir", None)
        )
        df = loader.load(**kw)
    else:
        raise DataSourceError(f"unknown source {source!r}; expected one of {SOURCES}")
    if with_outcomes:
        df = derive_outcomes(df, arm_space, segmenter, cfg)
    return df


__all__ = [
    "ALL_RETAINED_COLUMNS",
    "CANONICAL_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "DATE_COLUMNS",
    "DEFAULT_FILENAME",
    "DEFAULT_SHIFT",
    "DEFAULT_SHIFT_AT",
    "ID_COLUMN",
    "INTERIM_COLUMNS",
    "KAGGLE_DATASET",
    "NULL_POLICY",
    "NUMERIC_COLUMNS",
    "OUTCOME_COLUMNS",
    "POLICY_COLUMNS",
    "PRE_DECISION_COLUMNS",
    "RETAINED_VARIABLES",
    "SOURCES",
    "STATUS_CURRENT",
    "STATUS_DEFAULTED",
    "STATUS_DELINQUENT",
    "STATUS_FULLY_PAID",
    "STATUS_RESOLVED",
    "STATUS_UNRESOLVED",
    "STRING_DTYPE",
    "TERMINAL_COLUMNS",
    "ArmSpace",
    "BorrowerContext",
    "LeakageGuard",
    "LendingClubLoader",
    "LendingPolicy",
    "LoanStatePanel",
    "ReferenceRates",
    "Segmenter",
    "SyntheticLendingClub",
    "Variable",
    "add_months",
    "annuity_installment",
    "canonicalize",
    "default_lending_club_path",
    "derive_outcomes",
    "generate_synthetic",
    "interim_state",
    "load_loans",
    "months_between",
    "paths_to_counts",
    "recovery_rate",
    "row_segments",
    "sample_bucket",
    "subsample",
    "summarize",
    "temporal_split",
    "validate_loan_table",
]
