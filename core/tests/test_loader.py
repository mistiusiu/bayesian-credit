"""LendingClubLoader: DuckDB projection/parsing, sampling, cache (SPEC §1.2-§1.3)."""

import csv
import json
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import MCTMConfig
from bayesian_credit.data.loader import (
    CANONICAL_COLUMNS,
    HASH_BUCKETS,
    STRING_DTYPE,
    LendingClubLoader,
    _hash_columns_sql,
    canonicalize,
    default_lending_club_path,
    sample_bucket,
    subsample,
    validate_loan_table,
)
from bayesian_credit.data.outcomes import derive_outcomes
from bayesian_credit.data.panel import LoanStatePanel
from bayesian_credit.data.schema import (
    ALL_RETAINED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    PERCENT_NUMERIC_COLUMNS,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED,
)
from bayesian_credit.data.synthetic import generate
from bayesian_credit.exceptions import DataSourceError

FORBIDDEN = ["member_id", "emp_title", "url", "desc", "zip_code", "addr_state", "title"]


# ---------------------------------------------------------------------------------------
# helpers: write LC-formatted CSVs
# ---------------------------------------------------------------------------------------
def _emp_text(v):
    if pd.isna(v):
        return ""
    v = int(v)
    return {0: "< 1 year", 1: "1 year", 10: "10+ years"}.get(v, f"{v} years")


def write_lc_csv(df: pd.DataFrame, path, *, percent_sign: bool = True) -> None:
    """Inverse of the loader's parsing: canonical LoanTable -> Kaggle-style CSV."""
    out = {"id": df["loan_id"].astype(str)}
    for col in FORBIDDEN:
        out[col] = pd.Series(["free text, with comma"] * len(df))
    for col in ALL_RETAINED_COLUMNS:
        src = "verification_status_joint" if col == "verified_status_joint" else col
        s = df[col]
        if col in ("int_rate", "revol_util"):
            fmt = "{:.2f}%" if percent_sign else "{:.2f}"
            txt = (s * 100).map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
        elif col in PERCENT_NUMERIC_COLUMNS:
            txt = (s * 100).map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
        elif col == "term":
            txt = s.map(lambda v: f" {int(v)} months")
        elif col == "emp_length":
            txt = s.map(_emp_text)
        elif col in DATE_COLUMNS:
            txt = s.map(lambda v: "" if pd.isna(v) else v.strftime("%b-%Y"))
        elif col in CATEGORICAL_COLUMNS:
            txt = s.astype(object).map(lambda v: "" if pd.isna(v) else str(v))
        else:
            txt = s.map(lambda v: "" if pd.isna(v) else repr(float(v)))
        out[src] = txt
    out["policy_code"] = pd.Series(["1.0"] * len(df))
    pd.DataFrame(out).to_csv(path, index=False)


TINY_ROWS = [
    # id, loan_amnt, term, int_rate, grade, sub_grade, emp_length, home, issue, status, revol_util,
    # bc_util, all_util, pct, last_pymnt_d, next_pymnt_d, policy_code, app_type, vsj, fee
    dict(id="101", loan_amnt="3600.0", term=" 36 months", int_rate="13.99", grade="C", sub_grade="C4",
         emp_length="10+ years", home_ownership="MORTGAGE", issue_d="Dec-2015", loan_status="Fully Paid",
         revol_util="29.7", bc_util="37.2", all_util="34.0", pct_tl_nvr_dlq="76.9", last_pymnt_d="Jan-2019",
         next_pymnt_d="", policy_code="1.0", application_type="Individual", verification_status_joint="",
         total_rec_late_fee="0.0"),
    dict(id="102", loan_amnt="24700.0", term=" 60 months", int_rate="22.45%", grade="F", sub_grade="F1",
         emp_length="< 1 year", home_ownership="ANY", issue_d="Jan-2016", loan_status="Charged Off",
         revol_util="64.5%", bc_util="77.5", all_util="", pct_tl_nvr_dlq="", last_pymnt_d="Jul-2016",
         next_pymnt_d="", policy_code="1.0", application_type="Joint App", verification_status_joint="Verified",
         total_rec_late_fee="15.0"),
    dict(id="103", loan_amnt="1000.0", term="36 months", int_rate=" 7.50 ", grade="A", sub_grade="A3",
         emp_length="n/a", home_ownership="NONE", issue_d="Mar-2018", loan_status="Current",
         revol_util="", bc_util="", all_util="", pct_tl_nvr_dlq="", last_pymnt_d="Mar-2019",
         next_pymnt_d="Apr-2019", policy_code="1.0", application_type="Individual", verification_status_joint="",
         total_rec_late_fee="0.0"),
    dict(id="104", loan_amnt="5000.0", term=" 36 months", int_rate="10.00", grade="B", sub_grade="B1",
         emp_length="2 years", home_ownership="RENT", issue_d="Jan-2016", loan_status="Fully Paid",
         policy_code="2.0", application_type="Individual"),  # policy_code != 1 -> dropped
    dict(id="105", loan_amnt="5000.0", term=" 36 months", int_rate="", grade="B", sub_grade="B1",
         issue_d="Jan-2016", loan_status="Current", policy_code="1.0"),  # missing int_rate -> dropped
    dict(id="Total amount funded in policy code 1: 123456", policy_code=""),  # LC footer junk -> dropped
    dict(),  # all-NULL junk row (the Kaggle file ends with 33 of these) -> dropped
]
TINY_HEADER = [
    "id", "member_id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status",
    "url", "desc", "purpose", "zip_code", "addr_state", "dti", "revol_util", "bc_util", "all_util",
    "pct_tl_nvr_dlq", "last_pymnt_d", "next_pymnt_d", "policy_code", "application_type",
    "verification_status_joint", "total_rec_late_fee",
]


@pytest.fixture
def tiny_csv(tmp_path):
    path = tmp_path / "tiny_lc.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TINY_HEADER)
        writer.writeheader()
        for row in TINY_ROWS:
            full = {k: "" for k in TINY_HEADER}
            full.update(row)
            if row.get("loan_amnt"):
                full.update(funded_amnt=row["loan_amnt"], installment="123.03", annual_inc="55000.0",
                            verification_status="Not Verified", purpose="debt_consolidation", dti="5.91",
                            emp_title="Teacher, part-time", desc='says "hello", twice', url="https://lc/x",
                            zip_code="190xx", addr_state="PA", member_id="999")
            writer.writerow(full)
    return path


@pytest.fixture(scope="module")
def synth_table():
    return generate(1_500, seed=11)


@pytest.fixture
def synth_csv(tmp_path, synth_table):
    path = tmp_path / "synth_lc.csv"
    write_lc_csv(synth_table, path)
    return path


# ---------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------
class TestParsing:
    def test_tiny_csv_rows_cleaned(self, tiny_csv):
        df = LendingClubLoader(tiny_csv, cache=False).load()
        assert df["loan_id"].tolist() == [101, 102, 103]
        assert list(df.columns) == list(CANONICAL_COLUMNS)
        validate_loan_table(df)

    def test_tiny_csv_formats(self, tiny_csv):
        df = LendingClubLoader(tiny_csv, cache=False).load()
        assert df["int_rate"].tolist() == pytest.approx([0.1399, 0.2245, 0.075])
        assert df["term"].tolist() == [36, 60, 36]
        assert df["term"].dtype == np.int64
        np.testing.assert_array_equal(df["emp_length"].to_numpy(), [10.0, 0.0, np.nan])
        assert df["home_ownership"].tolist() == ["MORTGAGE", "OTHER", "OTHER"]
        np.testing.assert_allclose(df["revol_util"], [0.297, 0.645, np.nan])
        np.testing.assert_allclose(df["bc_util"], [0.372, 0.775, np.nan])
        np.testing.assert_allclose(df["all_util"], [0.34, np.nan, np.nan])
        np.testing.assert_allclose(df["pct_tl_nvr_dlq"], [0.769, np.nan, np.nan])
        assert df["issue_d"].tolist() == [pd.Timestamp("2015-12-01"), pd.Timestamp("2016-01-01"), pd.Timestamp("2018-03-01")]
        assert df["last_pymnt_d"].tolist() == [pd.Timestamp("2019-01-01"), pd.Timestamp("2016-07-01"), pd.Timestamp("2019-03-01")]
        assert df["next_pymnt_d"].isna().tolist() == [True, True, False]
        assert df["total_rec_late_fee"].tolist() == [0.0, 15.0, 0.0]

    def test_tiny_csv_renames_and_minimises(self, tiny_csv):
        df = LendingClubLoader(tiny_csv, cache=False).load()
        assert "verification_status_joint" not in df.columns
        assert df["verified_status_joint"].isna().tolist() == [True, False, True]
        assert df.loc[1, "verified_status_joint"] == "Verified"
        assert not set(FORBIDDEN) & set(df.columns)
        assert "id" not in df.columns and "policy_code" not in df.columns
        # retained columns absent from the file come back all-missing with canonical dtype
        assert df["mort_acc"].isna().all() and df["mort_acc"].dtype == np.float64

    def test_dtypes_are_canonical(self, tiny_csv):
        df = LendingClubLoader(tiny_csv, cache=False).load()
        for col in CATEGORICAL_COLUMNS:
            assert df[col].dtype == STRING_DTYPE, col
        for col in DATE_COLUMNS:
            assert df[col].dtype == np.dtype("datetime64[ns]"), col
        assert df["loan_id"].dtype == np.int64

    @pytest.mark.parametrize("percent_sign", [True, False])
    def test_round_trip_synthetic_through_lc_format(self, tmp_path, synth_table, percent_sign):
        path = tmp_path / "rt.csv"
        write_lc_csv(synth_table, path, percent_sign=percent_sign)
        loaded = LendingClubLoader(path, cache=False).load(order="file")
        assert len(loaded) == len(synth_table)
        expected = synth_table.set_index("loan_id").loc[loaded["loan_id"]].reset_index()
        for col in CANONICAL_COLUMNS:
            a, b = loaded[col], expected[col]
            if col in CATEGORICAL_COLUMNS:
                assert a.fillna("<NA>").tolist() == b.fillna("<NA>").tolist(), col
            elif col in DATE_COLUMNS or col in ("loan_id", "term"):
                assert a.equals(b), col
            else:
                np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-9, err_msg=col)

    def test_column_expression_handles_missing_source(self):
        assert LendingClubLoader.column_expression("grade", None) == "CAST(NULL AS VARCHAR)"
        assert "TIMESTAMP_NS" in LendingClubLoader.column_expression("issue_d", None)


# ---------------------------------------------------------------------------------------
# selection: order / max_rows / sampling / issue window / columns
# ---------------------------------------------------------------------------------------
class TestSelection:
    def test_issue_order_returns_earliest_loans(self, synth_csv, synth_table):
        df = LendingClubLoader(synth_csv, cache=False).load(max_rows=100)
        expected = synth_table.sort_values(["issue_d", "loan_id"]).head(100)
        assert df["loan_id"].tolist() == expected["loan_id"].tolist()

    def test_file_order_keeps_file_order(self, synth_csv, synth_table):
        df = LendingClubLoader(synth_csv, cache=False).load(max_rows=50, order="file")
        assert df["loan_id"].tolist() == synth_table["loan_id"].head(50).tolist()

    def test_random_order_is_reproducible_and_seeded(self, synth_csv):
        loader = LendingClubLoader(synth_csv, cache=False)
        a = loader.load(max_rows=200, order="random", seed=1)
        b = loader.load(max_rows=200, order="random", seed=1)
        c = loader.load(max_rows=200, order="random", seed=2)
        assert a.equals(b)
        assert set(a["loan_id"]) != set(c["loan_id"])

    def test_sample_fraction_deterministic_and_matches_python_rule(self, synth_csv, synth_table):
        loader = LendingClubLoader(synth_csv, cache=False)
        s1 = loader.load(sample_fraction=0.3, seed=7)
        s2 = loader.load(sample_fraction=0.3, seed=7)
        assert s1.equals(s2)
        expected = synth_table.loc[sample_bucket(synth_table["loan_id"], 7) < 0.3 * HASH_BUCKETS, "loan_id"]
        assert sorted(s1["loan_id"]) == sorted(expected)
        assert 0.2 < len(s1) / len(synth_table) < 0.4
        assert set(s1["loan_id"]) != set(loader.load(sample_fraction=0.3, seed=8)["loan_id"])
        assert sorted(subsample(synth_table, 0.3, seed=7)["loan_id"]) == sorted(expected)

    def test_sample_then_limit(self, synth_csv):
        df = LendingClubLoader(synth_csv, cache=False).load(sample_fraction=0.5, max_rows=10, seed=3)
        assert len(df) == 10

    def test_issue_window_is_inclusive(self, synth_csv, synth_table):
        df = LendingClubLoader(synth_csv, cache=False).load(issue_from="2015-01", issue_to="2015-12")
        assert df["issue_d"].min() >= pd.Timestamp("2015-01-01")
        assert df["issue_d"].max() <= pd.Timestamp("2015-12-01")
        expected = synth_table[(synth_table.issue_d >= "2015-01-01") & (synth_table.issue_d <= "2015-12-01")]
        assert len(df) == len(expected)

    def test_columns_subset(self, synth_csv):
        df = LendingClubLoader(synth_csv, cache=False).load(columns=["grade", "dti", "loan_status"], max_rows=5)
        assert list(df.columns) == ["loan_id", "dti", "grade", "loan_status"]

    @pytest.mark.parametrize("bad", ["member_id", "emp_title", "zip_code", "y_default"])
    def test_non_retained_columns_refused(self, synth_csv, bad):
        with pytest.raises(DataSourceError, match="retained"):
            LendingClubLoader(synth_csv, cache=False).load(columns=["dti", bad])

    @pytest.mark.parametrize(
        "kwargs",
        [dict(order="newest"), dict(sample_fraction=0.0), dict(sample_fraction=1.5), dict(max_rows=0),
         dict(issue_from="not a date")],
    )
    def test_invalid_arguments(self, synth_csv, kwargs):
        with pytest.raises(DataSourceError):
            LendingClubLoader(synth_csv, cache=False).load(**kwargs)

    def test_sql_hash_equals_numpy_hash(self):
        ids = np.array([0, 1, 54_734, 68_407_277, 145_647_287, 2**32 + 5, 9_000_000_000], dtype=np.int64)
        for seed in (0, 42, 2**40 + 3):
            con = duckdb.connect()
            got = con.execute(
                f"SELECT __bucket FROM (SELECT *, {_hash_columns_sql(seed)} FROM (SELECT unnest($ids) AS loan_id))",
                {"ids": ids.tolist()},
            ).fetchnumpy()["__bucket"]
            np.testing.assert_array_equal(got, sample_bucket(ids, seed))


# ---------------------------------------------------------------------------------------
# errors, cache, description
# ---------------------------------------------------------------------------------------
class TestCacheAndErrors:
    def test_missing_file_message_points_to_download_script(self, tmp_path):
        loader = LendingClubLoader(tmp_path / "nope.csv.gz")
        with pytest.raises(DataSourceError, match="download_data.sh"):
            loader.load()
        info = loader.describe_source()
        assert info["exists"] is False and "hint" in info

    def test_build_cache_and_load_from_parquet(self, tmp_path, synth_csv):
        loader = LendingClubLoader(synth_csv, cache=True, cache_dir=tmp_path / "cache")
        direct = LendingClubLoader(synth_csv, cache=False).load(sample_fraction=0.4, seed=5)
        assert not loader.cache_is_fresh()
        cached = loader.load(sample_fraction=0.4, seed=5)  # builds the cache on first use
        assert loader.cache_is_fresh() and loader.last_source == "parquet"
        assert loader.cache_path.parent == tmp_path / "cache"
        meta = json.loads(loader.cache_path.with_suffix(".json").read_text())
        assert meta["n_rows"] == len(LendingClubLoader(synth_csv, cache=False).load())
        assert cached.equals(direct)
        # file-order + limit reads the CSV directly (early exit) even when caching is on
        loader.load(max_rows=5, order="file")
        assert loader.last_source == "csv"

    def test_cache_goes_stale_when_source_changes(self, tmp_path, synth_csv):
        loader = LendingClubLoader(synth_csv, cache_dir=tmp_path / "c")
        loader.build_cache()
        assert loader.cache_is_fresh()
        st = synth_csv.stat()
        os.utime(synth_csv, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert not loader.cache_is_fresh()

    def test_parquet_round_trip_keeps_outcome_dtypes(self, tmp_path, synth_table):
        derived = derive_outcomes(synth_table)
        path = LendingClubLoader.to_parquet(derived, tmp_path / "t.parquet")
        back = LendingClubLoader.from_parquet(path)
        assert list(back.columns) == list(derived.columns)
        pd.testing.assert_frame_equal(back, derived)
        sub = LendingClubLoader.from_parquet(path, columns=["grade"])
        assert list(sub.columns) == ["loan_id", "grade"]
        with pytest.raises(DataSourceError):
            LendingClubLoader.from_parquet(tmp_path / "missing.parquet")

    def test_describe_source(self, tiny_csv):
        info = LendingClubLoader(tiny_csv, cache=False).describe_source()
        assert info["exists"] and info["n_rows_raw"] == len(TINY_ROWS)
        assert info["renamed"] == {"id": "loan_id", "verification_status_joint": "verified_status_joint"}
        assert info["retained_present"]["int_rate"] and not info["retained_present"]["mort_acc"]
        assert "mort_acc" in info["missing_retained"]
        assert info["n_source_columns"] == len(TINY_HEADER)


class TestCanonicalHelpers:
    def test_canonicalize_converts_and_orders(self, synth_table):
        messy = synth_table.iloc[::-1, ::-1].copy()
        messy["term"] = messy["term"].astype(str)
        messy["grade"] = messy["grade"].astype(object)
        messy["issue_d"] = messy["issue_d"].astype("datetime64[us]")
        out = canonicalize(messy)
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        validate_loan_table(out)

    def test_canonicalize_strictness(self, synth_table):
        partial = synth_table.drop(columns=["mort_acc", "grade"])
        with pytest.raises(DataSourceError, match="missing"):
            canonicalize(partial)
        filled = canonicalize(partial, strict=False)
        assert filled["mort_acc"].isna().all() and filled["grade"].dtype == STRING_DTYPE
        validate_loan_table(filled)

    def test_validate_rejects_percent_rates_and_duplicates(self, synth_table):
        bad = synth_table.copy()
        bad["int_rate"] = bad["int_rate"] * 100
        with pytest.raises(DataSourceError, match="fraction"):
            validate_loan_table(bad)
        dup = pd.concat([synth_table.head(2), synth_table.head(1)], ignore_index=True)
        with pytest.raises(DataSourceError, match="unique"):
            validate_loan_table(dup)


# ---------------------------------------------------------------------------------------
# the real Kaggle file (skipped when absent)
# ---------------------------------------------------------------------------------------
REAL_PATH = default_lending_club_path()
needs_real_file = pytest.mark.skipif(not REAL_PATH.is_file(), reason=f"Kaggle file not found at {REAL_PATH}")


@needs_real_file
class TestRealFile:
    def test_load_first_20000_rows(self):
        loader = LendingClubLoader(REAL_PATH, cache=False)
        df = loader.load(max_rows=20_000, order="file")
        assert len(df) == 20_000
        validate_loan_table(df)
        assert set(df["loan_status"]) <= STATUS_RESOLVED | STATUS_UNRESOLVED
        assert df["int_rate"].between(0.05, 0.31).all()
        assert set(df["term"]) <= {36, 60}
        assert set(df["home_ownership"]) <= {"MORTGAGE", "RENT", "OWN", "OTHER"}
        assert df["verified_status_joint"].notna().any()
        assert df["emp_length"].isna().any()
        assert df["emp_length"].dropna().between(0, 10).all()
        derived = derive_outcomes(df)
        assert derived["y_resolved"].any()
        panel = LoanStatePanel.from_table(derived, MCTMConfig())
        assert panel.counts.sum() == panel.n_loan_months - panel.n_loans

    def test_describe_source_header(self):
        info = LendingClubLoader(REAL_PATH, cache=False).describe_source(count_rows=False)
        assert info["n_source_columns"] == 151
        assert not info["missing_retained"]
        assert info["renamed"] == {"id": "loan_id", "verification_status_joint": "verified_status_joint"}

    def test_cached_earliest_200k(self):
        loader = LendingClubLoader(REAL_PATH)
        if not loader.cache_is_fresh():
            pytest.skip("parquet cache not built (run LendingClubLoader().build_cache())")
        df = loader.load(max_rows=200_000)
        assert len(df) == 200_000 and df["issue_d"].is_monotonic_increasing
        assert df["issue_d"].min() == pd.Timestamp("2007-06-01")
