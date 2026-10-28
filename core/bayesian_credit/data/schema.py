"""Retained variables, their lifecycle stage, and the leakage guard (Appendix F).

Table 3 of the proposal classifies every retained Lending Club variable by the stage at
which it becomes available. That classification is the data-leakage firewall (F.3): the
contextual bandit decides using *pre-decision* variables only; interim variables enter
when they would have been observed; terminal variables only update posteriors after the
outcome has occurred.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..exceptions import LeakageError
from ..types import Stage


@dataclass(frozen=True)
class Variable:
    name: str
    stage: Stage
    primary_use: str
    kind: str  # "numeric" | "categorical" | "date"


# --- Table 3: Retained Variables and Their Availability in the Credit Decision Process ---
RETAINED_VARIABLES: tuple[Variable, ...] = (
    Variable("annual_inc", Stage.PRE_DECISION, "Borrower context / risk estimation", "numeric"),
    Variable("annual_inc_joint", Stage.PRE_DECISION, "Joint-borrower risk estimation", "numeric"),
    Variable("application_type", Stage.PRE_DECISION, "Borrower context", "categorical"),
    Variable("dti", Stage.PRE_DECISION, "Risk estimation", "numeric"),
    Variable("dti_joint", Stage.PRE_DECISION, "Joint-borrower risk estimation", "numeric"),
    Variable("emp_length", Stage.PRE_DECISION, "Borrower context", "numeric"),
    Variable("fico_range_low", Stage.PRE_DECISION, "Credit-risk estimation", "numeric"),
    Variable("fico_range_high", Stage.PRE_DECISION, "Credit-risk estimation", "numeric"),
    Variable("home_ownership", Stage.PRE_DECISION, "Borrower context", "categorical"),
    Variable("purpose", Stage.PRE_DECISION, "Loan context", "categorical"),
    Variable("verification_status", Stage.PRE_DECISION, "Information-quality assessment", "categorical"),
    Variable("verified_status_joint", Stage.PRE_DECISION, "Joint-borrower information quality", "categorical"),
    Variable("delinq_2yrs", Stage.PRE_DECISION, "Historical credit risk", "numeric"),
    Variable("delinq_amnt", Stage.PRE_DECISION, "Credit risk", "numeric"),
    Variable("inq_last_6mths", Stage.PRE_DECISION, "Credit-seeking behavior", "numeric"),
    Variable("inq_last_12m", Stage.PRE_DECISION, "Credit-seeking behavior", "numeric"),
    Variable("open_acc", Stage.PRE_DECISION, "Credit profile", "numeric"),
    Variable("total_acc", Stage.PRE_DECISION, "Credit history", "numeric"),
    Variable("revol_bal", Stage.PRE_DECISION, "Credit exposure", "numeric"),
    Variable("revol_util", Stage.PRE_DECISION, "Credit utilization", "numeric"),
    Variable("installment", Stage.PRE_DECISION, "Repayment burden", "numeric"),
    Variable("mort_acc", Stage.PRE_DECISION, "Credit profile", "numeric"),
    Variable("pub_rec", Stage.PRE_DECISION, "Credit risk", "numeric"),
    Variable("pub_rec_bankruptcies", Stage.PRE_DECISION, "Credit risk", "numeric"),
    Variable("collections_12_mths_ex_med", Stage.PRE_DECISION, "Credit risk", "numeric"),
    Variable("num_accts_ever_120_pd", Stage.PRE_DECISION, "Historical severe delinquency", "numeric"),
    Variable("pct_tl_nvr_dlq", Stage.PRE_DECISION, "Credit behavior", "numeric"),
    Variable("num_sats", Stage.PRE_DECISION, "Credit profile", "numeric"),
    Variable("tot_coll_amt", Stage.PRE_DECISION, "Historical credit risk", "numeric"),
    Variable("tot_cur_bal", Stage.PRE_DECISION, "Current credit exposure", "numeric"),
    Variable("tot_hi_cred_lim", Stage.PRE_DECISION, "Credit capacity", "numeric"),
    Variable("total_bal_ex_mort", Stage.PRE_DECISION, "Debt exposure", "numeric"),
    Variable("total_bc_limit", Stage.PRE_DECISION, "Revolving credit capacity", "numeric"),
    Variable("bc_util", Stage.PRE_DECISION, "Bankcard utilization", "numeric"),
    Variable("all_util", Stage.PRE_DECISION, "Overall credit utilization", "numeric"),
    Variable("grade", Stage.PRE_DECISION, "Credit-risk context", "categorical"),
    Variable("sub_grade", Stage.PRE_DECISION, "Credit-risk context", "categorical"),
    Variable("loan_amnt", Stage.POLICY, "Lending-policy arm", "numeric"),
    Variable("int_rate", Stage.POLICY, "Lending-policy arm", "numeric"),
    Variable("term", Stage.POLICY, "Lending-policy characteristic", "numeric"),
    Variable("funded_amnt", Stage.DISBURSEMENT, "Actual loan exposure", "numeric"),
    Variable("issue_d", Stage.DISBURSEMENT, "Loan origination and time index", "date"),
    Variable("last_fico_range_low", Stage.INTERIM, "Updated borrower risk", "numeric"),
    Variable("last_fico_range_high", Stage.INTERIM, "Updated borrower risk", "numeric"),
    Variable("last_pymnt_amnt", Stage.INTERIM, "Interim repayment reward", "numeric"),
    Variable("last_pymnt_d", Stage.INTERIM, "Repayment timing", "date"),
    Variable("out_prncp", Stage.INTERIM, "Remaining exposure", "numeric"),
    Variable("out_prncp_inv", Stage.INTERIM, "Remaining investor exposure", "numeric"),
    Variable("num_tl_30dpd", Stage.INTERIM, "Current delinquency", "numeric"),
    Variable("num_tl_90g_dpd_24m", Stage.INTERIM, "Evolving credit risk", "numeric"),
    Variable("mths_since_last_delinq", Stage.INTERIM, "Evolving delinquency risk", "numeric"),
    Variable("next_pymnt_d", Stage.INTERIM, "Scheduled repayment information", "date"),
    Variable("loan_status", Stage.TERMINAL, "Default and credit-state outcome", "categorical"),
    Variable("recoveries", Stage.TERMINAL, "Recovery estimation", "numeric"),
    Variable("collection_recovery_fee", Stage.TERMINAL, "Recovery/revenue calculation", "numeric"),
    Variable("total_pymnt", Stage.TERMINAL, "Total repayment", "numeric"),
    Variable("total_pymnt_inv", Stage.TERMINAL, "Investor repayment", "numeric"),
    Variable("total_rec_prncp", Stage.TERMINAL, "Principal recovered", "numeric"),
    Variable("total_rec_int", Stage.TERMINAL, "Interest revenue", "numeric"),
    Variable("total_rec_late_fee", Stage.TERMINAL, "Late-fee revenue", "numeric"),
)

VARIABLES_BY_NAME: dict[str, Variable] = {v.name: v for v in RETAINED_VARIABLES}

#: Row key. The LC listing id is kept only to join rows/paths; it is never a feature.
ID_COLUMN = "loan_id"


def columns_for(*stages: Stage) -> tuple[str, ...]:
    wanted = set(stages)
    return tuple(v.name for v in RETAINED_VARIABLES if v.stage in wanted)


PRE_DECISION_COLUMNS = columns_for(Stage.PRE_DECISION)
POLICY_COLUMNS = columns_for(Stage.POLICY)
DISBURSEMENT_COLUMNS = columns_for(Stage.DISBURSEMENT)
INTERIM_COLUMNS = columns_for(Stage.INTERIM)
TERMINAL_COLUMNS = columns_for(Stage.TERMINAL)
ALL_RETAINED_COLUMNS = tuple(v.name for v in RETAINED_VARIABLES)

CATEGORICAL_COLUMNS = tuple(v.name for v in RETAINED_VARIABLES if v.kind == "categorical")
DATE_COLUMNS = tuple(v.name for v in RETAINED_VARIABLES if v.kind == "date")
NUMERIC_COLUMNS = tuple(v.name for v in RETAINED_VARIABLES if v.kind == "numeric")

#: Lending Club stores these as percent strings ("12.99%") -> stored as fractions.
PERCENT_STRING_COLUMNS = ("int_rate", "revol_util")
#: Stored numerically in 0-100 by LC -> divided by 100 to fractions.
PERCENT_NUMERIC_COLUMNS = ("bc_util", "all_util", "pct_tl_nvr_dlq")

#: Derived pre-decision features (pure functions of PRE_DECISION columns) that may also
#: appear in a BorrowerContext.
DERIVED_CONTEXT_COLUMNS = (
    "purpose_group",
    "fico_band",
    "dti_band",
    "income_band",
    "emp_band",
    "fico_mid",
)

#: Derived outcome columns produced by ``data.outcomes.derive_outcomes`` (SPEC §1.2).
OUTCOME_COLUMNS = (
    "y_resolved",
    "y_default",
    "y_recovery_rate",
    "y_realized_revenue",
    "y_realized_loss",
    "y_months_on_book",
    "y_resolution_date",
    "y_historical_arm",
    "y_group",
)

# --- loan_status vocabulary (Lending Club accepted file) ---
STATUS_FULLY_PAID = frozenset(
    {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
)
STATUS_DEFAULTED = frozenset(
    {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
)
STATUS_DELINQUENT = frozenset({"In Grace Period", "Late (16-30 days)", "Late (31-120 days)"})
STATUS_CURRENT = frozenset({"Current"})
STATUS_RESOLVED = STATUS_FULLY_PAID | STATUS_DEFAULTED
STATUS_UNRESOLVED = STATUS_DELINQUENT | STATUS_CURRENT


class LeakageGuard:
    """Enforces Appendix F.3: no future information at an earlier decision stage."""

    @staticmethod
    def stage_of(column: str) -> Stage | None:
        if column in VARIABLES_BY_NAME:
            return VARIABLES_BY_NAME[column].stage
        if column in DERIVED_CONTEXT_COLUMNS:
            return Stage.PRE_DECISION
        if column in OUTCOME_COLUMNS:
            # y_group / y_historical_arm are functions of pre-decision + policy columns;
            # everything else encodes terminal information.
            if column == "y_group":
                return Stage.PRE_DECISION
            if column == "y_historical_arm":
                return Stage.POLICY
            return Stage.TERMINAL
        return None

    @classmethod
    def assert_allowed(cls, columns: Iterable[str], max_stage: Stage) -> None:
        """Raise :class:`LeakageError` if any column belongs to a stage after ``max_stage``."""
        offending: list[str] = []
        for col in columns:
            if col == ID_COLUMN:
                continue
            stage = cls.stage_of(col)
            if stage is None:
                offending.append(f"{col} (unknown column)")
            elif stage.order > max_stage.order:
                offending.append(f"{col} ({stage.value})")
        if offending:
            raise LeakageError(
                f"columns not available at stage '{max_stage.value}': {', '.join(sorted(offending))}"
            )

    @classmethod
    def context_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Project a LoanTable onto its pre-decision (context) columns only."""
        keep = [
            c
            for c in df.columns
            if c == ID_COLUMN or cls.stage_of(c) == Stage.PRE_DECISION
        ]
        return df.loc[:, keep]


@dataclass(frozen=True)
class BorrowerContext:
    """X_i — what the decision layer is allowed to see about borrower i.

    Construction validates every feature key through :class:`LeakageGuard`; passing an
    interim or terminal column raises :class:`LeakageError`.
    """

    loan_id: Any
    features: Mapping[str, Any] = field(default_factory=dict)
    issue_date: pd.Timestamp | None = None  # the decision epoch (time index only)

    def __post_init__(self) -> None:
        LeakageGuard.assert_allowed(self.features.keys(), Stage.PRE_DECISION)

    def __getitem__(self, key: str) -> Any:
        return self.features[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.features.get(key, default)

    @classmethod
    def from_row(cls, row: Mapping[str, Any] | pd.Series) -> BorrowerContext:
        """Build a context from a LoanTable row, silently dropping non-context columns.

        Use this at the boundary (runner / API); it is the one place that is allowed to
        *see* later-stage columns because it discards them.
        """
        feats = {
            k: v
            for k, v in dict(row).items()
            if LeakageGuard.stage_of(k) == Stage.PRE_DECISION
        }
        issue = row.get("issue_d") if hasattr(row, "get") else None
        ts = pd.Timestamp(issue) if issue is not None and not pd.isna(issue) else None
        return cls(loan_id=row.get(ID_COLUMN) if hasattr(row, "get") else None, features=feats, issue_date=ts)
