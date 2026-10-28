"""Borrower segmentation - the group g of the hierarchical model (SPEC §1.4)."""

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import SegmentationConfig
from bayesian_credit.data.schema import BorrowerContext
from bayesian_credit.data.segmentation import (
    DIMENSIONS,
    SEPARATOR,
    Segmenter,
    dti_band,
    emp_band,
    fico_band,
    income_band,
    purpose_group,
)
from bayesian_credit.exceptions import ConfigurationError


def _frame(**cols):
    return pd.DataFrame(cols)


class TestKeys:
    def test_default_dimensions_and_key_format(self, segmenter):
        assert segmenter.dimensions == ("grade", "purpose_group", "home_ownership")
        df = _frame(grade=["B", "A"], purpose=["debt_consolidation", "wedding"], home_ownership=["RENT", "mortgage"])
        keys = segmenter.assign(df)
        assert keys.name == "y_group"
        assert list(keys) == ["B|debt_consolidation|RENT", "A|other|MORTGAGE"]
        assert SEPARATOR == "|"

    def test_assign_preserves_index(self, segmenter):
        df = _frame(grade=["C"], purpose=["credit_card"], home_ownership=["OWN"])
        df.index = [42]
        assert segmenter.assign(df).index.tolist() == [42]

    def test_purpose_group(self):
        s = purpose_group(pd.Series(["credit_card", "DEBT_CONSOLIDATION", "car", None]))
        assert list(s) == ["credit_card", "debt_consolidation", "other", "other"]

    def test_home_ownership_collapses_rare_levels(self):
        seg = Segmenter(["home_ownership"])
        df = _frame(home_ownership=["ANY", "NONE", "OTHER", "own", None])
        assert list(seg.assign(df)) == ["OTHER", "OTHER", "OTHER", "OWN", "OTHER"]

    def test_missing_source_column_gives_unknown(self):
        seg = Segmenter(["grade", "verification_status"])
        assert list(seg.assign(_frame(grade=["A"]))) == ["A|unknown"]

    def test_key_for_context_matches_assign(self, segmenter):
        feats = {"grade": "D", "purpose": "credit_card", "home_ownership": "OWN", "dti": 12.0}
        ctx = BorrowerContext(loan_id=1, features=feats)
        assert segmenter.key_for(ctx) == segmenter.key_for(feats) == "D|credit_card|OWN"
        assert segmenter.key_for(ctx) == segmenter.assign(pd.DataFrame([feats])).iloc[0]


class TestBands:
    def test_fico_band_edges(self):
        out = fico_band(pd.Series([659, 660, 699, 700, 739, 740, 779, 780, 845, np.nan]))
        assert list(out) == ["<660", "660-699", "660-699", "700-739", "700-739", "740-779", "740-779",
                             "780+", "780+", "unknown"]

    def test_dti_band_edges(self):
        assert list(dti_band(pd.Series([0, 9.99, 10, 20, 29.9, 30, 45]))) == [
            "<10", "<10", "10-20", "20-30", "20-30", "30+", "30+"]

    def test_income_band_edges(self):
        assert list(income_band(pd.Series([10_000, 40_000, 74_999, 75_000, 120_000]))) == [
            "<40k", "40-75k", "40-75k", "75-120k", "120k+"]

    def test_emp_band_edges(self):
        assert list(emp_band(pd.Series([0, 1, 2, 5, 6, 9, 10, np.nan]))) == [
            "<2", "<2", "2-5", "2-5", "6-9", "6-9", "10+", "unknown"]

    def test_derived_dimensions_from_frame(self):
        seg = Segmenter(["fico_band", "dti_band", "income_band", "emp_band"])
        df = _frame(fico_range_low=[705.0], dti=[31.0], annual_inc=[50_000.0], emp_length=[3.0])
        derived = seg.derive(df)
        assert list(derived.columns) == list(seg.dimensions)
        assert seg.assign(df).iloc[0] == "700-739|30+|40-75k|2-5"

    def test_every_dimension_derives_from_pre_decision_column(self):
        for name, (src, fn) in DIMENSIONS.items():
            seg = Segmenter([name])
            assert seg.dimensions == (name,) and callable(fn) and isinstance(src, str)


class TestRefineCoarsen:
    def test_refine_appends_next_unused_dimension(self, segmenter):
        finer = segmenter.refine()
        assert finer.dimensions == (*segmenter.dimensions, "fico_band")
        assert finer.refine().dimensions[-1] == "dti_band"
        assert segmenter.dimensions == ("grade", "purpose_group", "home_ownership")  # immutable

    def test_refine_skips_dimensions_already_used(self):
        seg = Segmenter(["fico_band"], refinement_order=("fico_band", "dti_band"))
        assert seg.refine().dimensions == ("fico_band", "dti_band")

    def test_refine_exhausted_returns_self(self):
        seg = Segmenter(["grade"], refinement_order=("grade",))
        assert seg.refine() is seg

    def test_coarsen_is_inverse_of_refine(self, segmenter):
        assert segmenter.refine().coarsen() == segmenter
        assert segmenter.coarsen().dimensions == ("grade", "purpose_group")

    def test_coarsen_never_below_one_dimension(self):
        seg = Segmenter(["grade"])
        assert seg.coarsen() is seg

    def test_refined_keys_split_groups(self, segmenter):
        df = _frame(grade=["A", "A"], purpose=["credit_card"] * 2, home_ownership=["RENT"] * 2,
                    fico_range_low=[690.0, 750.0])
        assert segmenter.assign(df).nunique() == 1
        assert segmenter.refine().assign(df).nunique() == 2


class TestValidation:
    def test_unknown_dimension_raises(self):
        with pytest.raises(ConfigurationError, match="unsupported"):
            Segmenter(["grade", "zip_code"])

    def test_empty_dimensions_raise(self):
        with pytest.raises(ConfigurationError):
            Segmenter([])

    def test_round_trip_equality_hash(self, segmenter):
        clone = Segmenter.from_dict(segmenter.to_dict())
        assert clone == segmenter and hash(clone) == hash(segmenter)
        assert Segmenter.from_config(SegmentationConfig()) == segmenter
        assert "grade" in repr(segmenter)
