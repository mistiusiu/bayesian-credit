"""Table 3 stage classification, leakage guard and BorrowerContext (SPEC §1.1, Appendix F.3)."""

import itertools

import pandas as pd
import pytest

from bayesian_credit.data.schema import (
    ALL_RETAINED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    DERIVED_CONTEXT_COLUMNS,
    DISBURSEMENT_COLUMNS,
    ID_COLUMN,
    INTERIM_COLUMNS,
    NUMERIC_COLUMNS,
    OUTCOME_COLUMNS,
    PERCENT_NUMERIC_COLUMNS,
    PERCENT_STRING_COLUMNS,
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
    VARIABLES_BY_NAME,
    BorrowerContext,
    LeakageGuard,
    columns_for,
)
from bayesian_credit.exceptions import LeakageError
from bayesian_credit.types import Stage

# Table 3 of the proposal, verbatim (SPEC §1.1)
TABLE3 = {
    Stage.PRE_DECISION: {
        "annual_inc", "annual_inc_joint", "application_type", "dti", "dti_joint", "emp_length",
        "fico_range_low", "fico_range_high", "home_ownership", "purpose", "verification_status",
        "verified_status_joint", "delinq_2yrs", "delinq_amnt", "inq_last_6mths", "inq_last_12m",
        "open_acc", "total_acc", "revol_bal", "revol_util", "installment", "mort_acc", "pub_rec",
        "pub_rec_bankruptcies", "collections_12_mths_ex_med", "num_accts_ever_120_pd",
        "pct_tl_nvr_dlq", "num_sats", "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim",
        "total_bal_ex_mort", "total_bc_limit", "bc_util", "all_util", "grade", "sub_grade",
    },
    Stage.POLICY: {"loan_amnt", "int_rate", "term"},
    Stage.DISBURSEMENT: {"funded_amnt", "issue_d"},
    Stage.INTERIM: {
        "last_fico_range_low", "last_fico_range_high", "last_pymnt_amnt", "last_pymnt_d", "out_prncp",
        "out_prncp_inv", "num_tl_30dpd", "num_tl_90g_dpd_24m", "mths_since_last_delinq", "next_pymnt_d",
    },
    Stage.TERMINAL: {
        "loan_status", "recoveries", "collection_recovery_fee", "total_pymnt", "total_pymnt_inv",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    },
}
STAGE_TUPLES = {
    Stage.PRE_DECISION: PRE_DECISION_COLUMNS,
    Stage.POLICY: POLICY_COLUMNS,
    Stage.DISBURSEMENT: DISBURSEMENT_COLUMNS,
    Stage.INTERIM: INTERIM_COLUMNS,
    Stage.TERMINAL: TERMINAL_COLUMNS,
}
LC_STATUSES = {
    "Fully Paid", "Current", "Charged Off", "Late (31-120 days)", "In Grace Period",
    "Late (16-30 days)", "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off", "Default",
}


class TestTable3:
    def test_each_stage_matches_table3_exactly(self):
        for stage, expected in TABLE3.items():
            assert set(STAGE_TUPLES[stage]) == expected, stage
            assert set(columns_for(stage)) == expected

    def test_stage_sets_are_pairwise_disjoint(self):
        for a, b in itertools.combinations(STAGE_TUPLES.values(), 2):
            assert not set(a) & set(b)

    def test_stage_sets_cover_all_retained_variables(self):
        union = set().union(*map(set, STAGE_TUPLES.values()))
        assert union == set(ALL_RETAINED_COLUMNS)
        assert len(ALL_RETAINED_COLUMNS) == len(set(ALL_RETAINED_COLUMNS)) == 60
        assert [len(STAGE_TUPLES[s]) for s in Stage] == [37, 3, 2, 10, 8]

    def test_every_variable_has_exactly_one_stage_and_lookup_is_consistent(self):
        assert len(VARIABLES_BY_NAME) == len(RETAINED_VARIABLES)
        for v in RETAINED_VARIABLES:
            assert VARIABLES_BY_NAME[v.name] is v
            assert isinstance(v.stage, Stage)
            assert v.kind in {"numeric", "categorical", "date"}

    def test_kinds(self):
        assert set(DATE_COLUMNS) == {"issue_d", "last_pymnt_d", "next_pymnt_d"}
        assert set(CATEGORICAL_COLUMNS) == {
            "grade", "sub_grade", "home_ownership", "purpose", "verification_status",
            "verified_status_joint", "application_type", "loan_status",
        }
        assert set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS) | set(DATE_COLUMNS) == set(ALL_RETAINED_COLUMNS)
        assert set(PERCENT_STRING_COLUMNS) | set(PERCENT_NUMERIC_COLUMNS) <= set(NUMERIC_COLUMNS)

    def test_data_minimisation_no_identifiers_or_free_text(self):
        forbidden = {"member_id", "url", "desc", "emp_title", "zip_code", "addr_state", "title", "id"}
        assert not forbidden & set(ALL_RETAINED_COLUMNS)
        assert ID_COLUMN == "loan_id" and ID_COLUMN not in ALL_RETAINED_COLUMNS

    def test_stage_order_is_the_lending_life_cycle(self):
        assert [s.order for s in Stage] == [0, 1, 2, 3, 4]
        assert Stage.PRE_DECISION.order < Stage.POLICY.order < Stage.TERMINAL.order


class TestStatusVocabulary:
    def test_resolved_and_unresolved_partition_lc_statuses(self):
        assert not STATUS_RESOLVED & STATUS_UNRESOLVED
        assert STATUS_RESOLVED | STATUS_UNRESOLVED == LC_STATUSES
        assert STATUS_RESOLVED == STATUS_FULLY_PAID | STATUS_DEFAULTED
        assert STATUS_UNRESOLVED == STATUS_DELINQUENT | STATUS_CURRENT
        assert not STATUS_FULLY_PAID & STATUS_DEFAULTED

    def test_policy_variants_and_default(self):
        assert "Does not meet the credit policy. Status:Fully Paid" in STATUS_FULLY_PAID
        assert "Does not meet the credit policy. Status:Charged Off" in STATUS_DEFAULTED
        assert "Default" in STATUS_DEFAULTED


class TestLeakageGuard:
    def test_stage_of(self):
        assert LeakageGuard.stage_of("fico_range_low") == Stage.PRE_DECISION
        assert LeakageGuard.stage_of("loan_amnt") == Stage.POLICY
        assert LeakageGuard.stage_of("issue_d") == Stage.DISBURSEMENT
        assert LeakageGuard.stage_of("out_prncp") == Stage.INTERIM
        assert LeakageGuard.stage_of("loan_status") == Stage.TERMINAL
        assert LeakageGuard.stage_of("not_a_column") is None

    def test_derived_and_outcome_columns(self):
        for col in DERIVED_CONTEXT_COLUMNS:
            assert LeakageGuard.stage_of(col) == Stage.PRE_DECISION
        assert LeakageGuard.stage_of("y_group") == Stage.PRE_DECISION
        assert LeakageGuard.stage_of("y_historical_arm") == Stage.POLICY
        for col in set(OUTCOME_COLUMNS) - {"y_group", "y_historical_arm"}:
            assert LeakageGuard.stage_of(col) == Stage.TERMINAL

    def test_pre_decision_columns_allowed_at_decision_time(self):
        LeakageGuard.assert_allowed([*PRE_DECISION_COLUMNS, ID_COLUMN, "fico_band", "y_group"], Stage.PRE_DECISION)

    @pytest.mark.parametrize(
        "column", ["loan_status", "recoveries", "total_pymnt", "total_rec_late_fee", "y_default", "y_recovery_rate"]
    )
    def test_terminal_columns_raise_at_decision_time(self, column):
        with pytest.raises(LeakageError, match=column):
            LeakageGuard.assert_allowed(["fico_range_low", column], Stage.PRE_DECISION)

    @pytest.mark.parametrize("column", ["out_prncp", "last_pymnt_d", "loan_amnt", "funded_amnt", "issue_d"])
    def test_later_stage_columns_raise(self, column):
        with pytest.raises(LeakageError):
            LeakageGuard.assert_allowed([column], Stage.PRE_DECISION)

    def test_terminal_columns_raise_at_interim_stage(self):
        LeakageGuard.assert_allowed(["out_prncp", "loan_amnt", "issue_d", "dti"], Stage.INTERIM)
        with pytest.raises(LeakageError):
            LeakageGuard.assert_allowed(["out_prncp", "loan_status"], Stage.INTERIM)

    def test_everything_allowed_at_terminal_stage(self):
        LeakageGuard.assert_allowed(ALL_RETAINED_COLUMNS + OUTCOME_COLUMNS, Stage.TERMINAL)

    def test_unknown_columns_are_rejected(self):
        with pytest.raises(LeakageError, match="unknown column"):
            LeakageGuard.assert_allowed(["emp_title"], Stage.TERMINAL)

    def test_error_lists_all_offending_columns(self):
        with pytest.raises(LeakageError) as exc:
            LeakageGuard.assert_allowed(["recoveries", "dti", "loan_status"], Stage.PRE_DECISION)
        msg = str(exc.value)
        assert "loan_status" in msg and "recoveries" in msg and "dti" not in msg

    def test_context_frame_keeps_only_context(self):
        df = pd.DataFrame(
            {"loan_id": [1], "dti": [10.0], "grade": ["A"], "loan_amnt": [1000.0], "loan_status": ["Current"],
             "out_prncp": [5.0], "fico_band": ["700-739"], "y_default": [0.0]}
        )
        out = LeakageGuard.context_frame(df)
        assert list(out.columns) == ["loan_id", "dti", "grade", "fico_band"]


class TestBorrowerContext:
    def test_construction_with_terminal_feature_raises(self):
        with pytest.raises(LeakageError):
            BorrowerContext(loan_id=1, features={"dti": 12.0, "loan_status": "Charged Off"})
        with pytest.raises(LeakageError):
            BorrowerContext(loan_id=1, features={"int_rate": 0.1})

    def test_from_row_drops_later_stage_columns(self):
        row = pd.Series(
            {
                "loan_id": 7, "dti": 12.5, "grade": "B", "fico_range_low": 700.0, "loan_amnt": 10_000.0,
                "int_rate": 0.12, "issue_d": pd.Timestamp("2016-03-01"), "out_prncp": 500.0,
                "loan_status": "Charged Off", "recoveries": 100.0, "y_default": 1.0, "y_group": "B|other|RENT",
            }
        )
        ctx = BorrowerContext.from_row(row)
        assert ctx.loan_id == 7
        assert ctx.issue_date == pd.Timestamp("2016-03-01")
        assert set(ctx.features) == {"dti", "grade", "fico_range_low", "y_group"}
        assert ctx["dti"] == 12.5 and ctx.get("grade") == "B" and ctx.get("missing", 0) == 0
        for leaked in ("loan_status", "recoveries", "out_prncp", "loan_amnt", "y_default"):
            assert leaked not in ctx.features

    def test_from_row_with_mapping_and_missing_issue_date(self):
        ctx = BorrowerContext.from_row({"loan_id": 3, "annual_inc": 50_000.0, "issue_d": pd.NaT, "total_pymnt": 1.0})
        assert ctx.issue_date is None
        assert dict(ctx.features) == {"annual_inc": 50_000.0}
