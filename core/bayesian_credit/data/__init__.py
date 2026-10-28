"""Data layer (SPEC §1, Appendix F): schema & leakage guard, arm space and segmentation."""

from __future__ import annotations

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
