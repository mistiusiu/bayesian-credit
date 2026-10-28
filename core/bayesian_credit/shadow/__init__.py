"""Counterfactual shadow lending engine (SPEC §2.10; proposal G.4, Table 8)."""

from __future__ import annotations

from .engine import (
    ARM_TABLE_COLUMNS,
    OUTCOME_CLASSES,
    PER_LOAN_COLUMNS,
    PortfolioSummary,
    ShadowLendingEngine,
    ShadowReport,
    eligibility_matrix,
)

__all__ = [
    "ARM_TABLE_COLUMNS",
    "OUTCOME_CLASSES",
    "PER_LOAN_COLUMNS",
    "PortfolioSummary",
    "ShadowLendingEngine",
    "ShadowReport",
    "eligibility_matrix",
]
