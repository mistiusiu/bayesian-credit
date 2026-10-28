"""Lending Club loader: Kaggle CSV(.gz) -> canonical ``LoanTable`` (SPEC §1.2-§1.3, Appendix F).

The whole projection / parsing step runs inside DuckDB SQL, so the 1.6 GB (374 MB gzipped)
Kaggle export is never materialised as 151 string columns in pandas:

* **data minimisation (F.4)** - only the Table 3 retained variables plus the row key
  ``loan_id`` (LC ``id``) are ever selected; free text, URLs, zip/state, member_id are not
  read into memory at all;
* **parsing** - ``int_rate`` / ``revol_util`` percent strings (``'13.99'`` or ``'13.99%'``)
  -> fractions; ``bc_util`` / ``all_util`` / ``pct_tl_nvr_dlq`` (0-100) -> fractions;
  ``term`` (``' 36 months'``) -> int; ``emp_length`` (``'10+ years'``, ``'< 1 year'``) ->
  float years; ``'Dec-2015'`` dates -> ``datetime64[ns]``; ``home_ownership`` ANY/NONE ->
  OTHER; ``verification_status_joint`` -> ``verified_status_joint`` (Table 3 name);
* **cleaning** - rows with a NULL ``loan_status`` / ``issue_d`` / ``loan_amnt`` /
  ``int_rate`` / ``term`` / ``id`` are dropped (the Kaggle file ends with 33 all-NULL junk
  rows), as are rows whose ``policy_code`` is present and != 1;
* **reproducible subsets** - ``sample_fraction`` keeps rows whose hash bucket
  ``lowbias32(loan_id + seed * 2654435769 mod 2^32) mod 10000`` falls below
  ``fraction * 10000``. The hash is plain integer arithmetic evaluated identically in SQL
  and numpy (:func:`sample_bucket`), so the same ``(fraction, seed)`` selects the same
  loans on every machine and DuckDB version (DuckDB's own ``hash()`` is version-specific).
* **ordering** - ``max_rows`` keeps the earliest loans (``order="issue_d"``), a seeded
  random subset (``"random"``) or the first rows of the file (``"file"``, early exit).

A parquet cache of the full canonical table (``build_cache``) makes every later load a
sub-second ``read_parquet`` query instead of a CSV scan.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from ..config import DataConfig
from ..exceptions import DataSourceError
from .schema import (
    ALL_RETAINED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    ID_COLUMN,
    OUTCOME_COLUMNS,
    PERCENT_NUMERIC_COLUMNS,
    PERCENT_STRING_COLUMNS,
)

#: Bump when the parsing rules change so stale parquet caches are rebuilt.
LOADER_VERSION = 3

#: File name of the Kaggle export (dataset ``wordsforthewise/lending-club``, CC0).
DEFAULT_FILENAME = "accepted_2007_to_2018Q4.csv.gz"
KAGGLE_DATASET = "wordsforthewise/lending-club"

#: Column order of a canonical LoanTable: the row key, then Table 3 in proposal order.
CANONICAL_COLUMNS: tuple[str, ...] = (ID_COLUMN, *ALL_RETAINED_COLUMNS)

#: pandas 3 default string dtype (``str``): missing values are ``NaN`` and comparisons
#: return plain ``bool`` arrays, so boolean masks never contain ``pd.NA``.
STRING_DTYPE = pd.StringDtype(na_value=np.nan)

#: Integer columns of the canonical table (everything else numeric is float64).
INTEGER_COLUMNS: tuple[str, ...] = ("term",)

#: Canonical name -> candidate source header names (first match wins).
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    ID_COLUMN: ("id", ID_COLUMN),
    "verified_status_joint": ("verification_status_joint", "verified_status_joint"),
}

#: Rows missing any of these are junk and dropped.
REQUIRED_NON_NULL: tuple[str, ...] = (ID_COLUMN, "loan_status", "issue_d", "loan_amnt", "int_rate", "term")

#: Accepted date layouts (the Kaggle export uses ``Dec-2015``).
DATE_FORMATS: tuple[str, ...] = ("%b-%Y", "%b-%y", "%Y-%m-%d")

OrderMode = Literal["issue_d", "random", "file"]


# ---------------------------------------------------------------------------------------
# canonical table helpers (shared with the synthetic generator)
# ---------------------------------------------------------------------------------------
def _coerce_datetime(values: pd.Series) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _has_canonical_dtype(col: str, dtype: Any) -> bool:
    if col == ID_COLUMN or col in INTEGER_COLUMNS:
        return dtype == np.dtype("int64")
    if col in DATE_COLUMNS:
        return dtype == np.dtype("datetime64[ns]")
    if col in CATEGORICAL_COLUMNS:
        return dtype == STRING_DTYPE
    return dtype == np.dtype("float64")


def _canonical_series(col: str, values: pd.Series) -> pd.Series:
    if col == ID_COLUMN or col in INTEGER_COLUMNS:
        if values.isna().any():
            raise DataSourceError(f"{col} contains missing values")
        return pd.to_numeric(values, errors="raise").astype("int64")
    if col in DATE_COLUMNS:
        return _coerce_datetime(values)
    if col in CATEGORICAL_COLUMNS:
        return values.astype(STRING_DTYPE)
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _empty_series(col: str, n: int) -> pd.Series:
    if col in DATE_COLUMNS:
        return pd.Series(pd.NaT, index=pd.RangeIndex(n), dtype="datetime64[ns]")
    if col in CATEGORICAL_COLUMNS:
        return pd.Series(np.nan, index=pd.RangeIndex(n), dtype=STRING_DTYPE)
    if col == ID_COLUMN or col in INTEGER_COLUMNS:
        raise DataSourceError(f"required column {col} is missing")
    return pd.Series(np.nan, index=pd.RangeIndex(n), dtype="float64")


def _outcome_series(col: str, values: pd.Series) -> pd.Series:
    if col == "y_resolution_date":
        return values if values.dtype == np.dtype("datetime64[ns]") else _coerce_datetime(values)
    if col == "y_group":
        return values if values.dtype == STRING_DTYPE else values.astype(STRING_DTYPE)
    if col == "y_resolved":
        return values.astype(bool)
    if col == "y_historical_arm":
        return values.astype("int64")
    return pd.to_numeric(values, errors="coerce").astype("float64")


def canonicalize(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    strict: bool = True,
    keep_extra: bool = False,
) -> pd.DataFrame:
    """Return ``df`` with canonical column order and dtypes (SPEC §1.2).

    ``columns`` restricts the retained variables (``loan_id`` is always first). With
    ``strict`` a missing retained column raises :class:`DataSourceError`; otherwise it is
    added as all-missing. ``keep_extra`` appends non-retained columns (e.g. the ``y_*``
    outcome columns) after the canonical ones, restoring outcome dtypes. Columns that
    already have their canonical dtype are not copied.
    """
    wanted = _resolve_columns(columns)
    missing = [c for c in wanted if c not in df.columns]
    if missing and strict:
        raise DataSourceError(f"LoanTable is missing columns: {missing}")
    extra = [c for c in df.columns if c not in wanted] if keep_extra else []
    present = [c for c in wanted if c in df.columns]
    order = present + extra
    out = df if list(df.columns) == order else df.loc[:, order]
    if not isinstance(out.index, pd.RangeIndex) or out.index.start != 0 or out.index.step != 1:
        out = out.reset_index(drop=True)
    else:
        out = out.copy(deep=False)
    n = len(out)
    for col in present:
        if not _has_canonical_dtype(col, out[col].dtype):
            out[col] = _canonical_series(col, out[col])
    for col in extra:
        if col in OUTCOME_COLUMNS:
            out[col] = _outcome_series(col, out[col])
    if missing:
        for col in missing:
            out[col] = _empty_series(col, n)
        out = out.loc[:, wanted + extra]
    return out


def validate_loan_table(df: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    """Raise :class:`DataSourceError` unless ``df`` is a canonical LoanTable."""
    wanted = _resolve_columns(columns)
    problems: list[str] = []
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    for col in wanted:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if col == ID_COLUMN or col in INTEGER_COLUMNS:
            ok = pd.api.types.is_integer_dtype(dtype)
        elif col in DATE_COLUMNS:
            ok = dtype == np.dtype("datetime64[ns]")
        elif col in CATEGORICAL_COLUMNS:
            ok = isinstance(dtype, pd.StringDtype)
        else:
            ok = dtype == np.dtype("float64")
        if not ok:
            problems.append(f"{col} has dtype {dtype}")
    if ID_COLUMN in df.columns and df[ID_COLUMN].duplicated().any():
        problems.append("loan_id is not unique")
    for col in REQUIRED_NON_NULL:
        if col in df.columns and df[col].isna().any():
            problems.append(f"{col} has missing values")
    if "int_rate" in df.columns and len(df) and float(df["int_rate"].max()) >= 1.0:
        problems.append("int_rate must be a fraction (0.1299), not a percent")
    if problems:
        raise DataSourceError("not a canonical LoanTable: " + "; ".join(problems))


def _resolve_columns(columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        return list(CANONICAL_COLUMNS)
    requested = list(dict.fromkeys(columns))
    unknown = [c for c in requested if c not in CANONICAL_COLUMNS]
    if unknown:
        raise DataSourceError(
            f"{unknown} are not retained variables (Table 3); data minimisation (Appendix F.4) "
            "forbids loading other Lending Club columns"
        )
    keep = set(requested) | {ID_COLUMN}
    return [c for c in CANONICAL_COLUMNS if c in keep]


#: Number of hash buckets used for deterministic sub-sampling.
HASH_BUCKETS = 10_000
_U32 = 4_294_967_296
_GOLDEN32 = 2_654_435_769  # 2^32 / phi, decorrelates seeds


def sample_bucket(loan_ids: np.ndarray | pd.Series | Sequence[int], seed: int = 42) -> np.ndarray:
    """Deterministic hash bucket in ``[0, HASH_BUCKETS)`` per loan id.

    ``lowbias32`` integer hash (C. Wellons) of ``(loan_id + seed * 2654435769) mod 2^32``.
    It is evaluated with identical arithmetic in DuckDB SQL by the loader, so
    ``sample_bucket(ids, seed) < fraction * HASH_BUCKETS`` selects exactly the loans that
    ``LendingClubLoader.load(sample_fraction=fraction, seed=seed)`` returns - on any machine
    and any DuckDB version.
    """
    ids = np.asarray(loan_ids, dtype=np.int64).astype(np.uint64)
    u32 = np.uint64(_U32)
    x = (ids % u32 + np.uint64(int(seed) % _U32) * np.uint64(_GOLDEN32)) % u32
    x = ((x ^ (x >> np.uint64(16))) * np.uint64(0x7FEB352D)) % u32
    x = ((x ^ (x >> np.uint64(15))) * np.uint64(0x846CA68B)) % u32
    x = x ^ (x >> np.uint64(16))
    return (x % np.uint64(HASH_BUCKETS)).astype(np.int64)


def _hash_columns_sql(seed: int) -> str:
    """SQL twin of :func:`sample_bucket` (UBIGINT arithmetic, products < 2^64)."""
    s = int(seed) % _U32
    return (
        f"((CAST(loan_id AS UBIGINT) % {_U32}) + CAST({s} AS UBIGINT) * {_GOLDEN32}) % {_U32} AS __x0, "
        f"(xor(__x0, __x0 >> 16) * {0x7FEB352D}) % {_U32} AS __x2, "
        f"(xor(__x2, __x2 >> 15) * {0x846CA68B}) % {_U32} AS __x4, "
        f"CAST(xor(__x4, __x4 >> 16) AS BIGINT) AS __hash, "
        f"CAST(xor(__x4, __x4 >> 16) % {HASH_BUCKETS} AS BIGINT) AS __bucket"
    )


def subsample(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Deterministic per-loan subsample of an in-memory LoanTable (same rule as the loader)."""
    if not 0 < fraction <= 1:
        raise DataSourceError("fraction must be in (0, 1]")
    keep = sample_bucket(df[ID_COLUMN].to_numpy(), seed) < int(round(fraction * HASH_BUCKETS))
    return df.loc[keep].reset_index(drop=True)


def default_lending_club_path() -> Path:
    """``$LENDING_CLUB_CSV`` or ``<repo>/data/raw/accepted_2007_to_2018Q4.csv.gz``."""
    env = os.environ.get("LENDING_CLUB_CSV")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "raw" / DEFAULT_FILENAME
        if candidate.exists():
            return candidate
    repo_guess = here.parents[3] if len(here.parents) > 3 else Path.cwd()
    if (repo_guess / "data").is_dir():
        return repo_guess / "data" / "raw" / DEFAULT_FILENAME
    return Path.cwd() / "data" / "raw" / DEFAULT_FILENAME


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _to_timestamp(value: Any, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataSourceError(f"{name}={value!r} is not a date") from exc
    if ts is pd.NaT:
        return None
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# ---------------------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------------------
class LendingClubLoader:
    """Reads the Kaggle Lending Club export with DuckDB into a canonical ``LoanTable``.

    Implements the data source of SPEC §1.3 / Appendix F.1: projection onto the Table 3
    retained variables (F.4 data minimisation), parsing to canonical dtypes (§1.2), junk-row
    and ``policy_code`` filtering, deterministic sub-sampling and a parquet cache.

    Parameters
    ----------
    path:
        CSV or CSV.GZ file; default :func:`default_lending_club_path`.
    data_cfg:
        Snapshot / charge-off conventions (kept for downstream outcome derivation).
    cache:
        If true, full-table queries go through a parquet cache of the canonical table,
        building it on first use (one CSV scan, ~10-30 s for the Kaggle file).
    cache_dir:
        Where the cache lives. Default: ``data/processed`` when the CSV sits in
        ``data/raw``, else the CSV's directory.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        data_cfg: DataConfig | None = None,
        *,
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else default_lending_club_path()
        self.data_cfg = data_cfg or DataConfig()
        self.cache = cache
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._header: list[str] | None = None
        self.last_query: str | None = None
        self.last_load_seconds: float | None = None
        self.last_source: str | None = None

    # ----- file / header ------------------------------------------------------------
    def _require_file(self) -> None:
        if not self.path.is_file():
            raise DataSourceError(
                f"Lending Club file not found: {self.path}. Download it with "
                f"`scripts/download_data.sh` (Kaggle dataset {KAGGLE_DATASET}, file "
                f"{DEFAULT_FILENAME}) or pass path=... / set $LENDING_CLUB_CSV. "
                "Use source='synthetic' to run without it."
            )

    @staticmethod
    def _connect() -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=":memory:")

    def _csv_source(self) -> str:
        return "read_csv($path, all_varchar=true, header=true, sample_size=20000)"

    def source_columns(self) -> list[str]:
        """Header of the source file (sniffed from the first rows only)."""
        self._require_file()
        if self._header is None:
            con = self._connect()
            try:
                rows = con.execute(
                    f"DESCRIBE SELECT * FROM {self._csv_source()}", {"path": str(self.path)}
                ).fetchall()
            except duckdb.Error as exc:
                raise DataSourceError(f"cannot read {self.path}: {exc}") from exc
            finally:
                con.close()
            self._header = [r[0] for r in rows]
        return list(self._header)

    def column_mapping(self) -> dict[str, str | None]:
        """Canonical column -> source header name (``None`` when absent from the file)."""
        header = set(self.source_columns())
        mapping: dict[str, str | None] = {}
        for col in CANONICAL_COLUMNS:
            candidates = SOURCE_ALIASES.get(col, (col,))
            mapping[col] = next((c for c in candidates if c in header), None)
        return mapping

    # ----- SQL ----------------------------------------------------------------------
    @staticmethod
    def column_expression(col: str, source: str | None) -> str:
        """DuckDB expression parsing raw VARCHAR ``source`` into canonical ``col``."""
        if source is None:
            if col in CATEGORICAL_COLUMNS:
                return "CAST(NULL AS VARCHAR)"
            if col in DATE_COLUMNS:
                return "CAST(NULL AS TIMESTAMP_NS)"
            if col == ID_COLUMN:
                return "CAST(NULL AS BIGINT)"
            if col in INTEGER_COLUMNS:
                return "CAST(NULL AS BIGINT)"
            return "CAST(NULL AS DOUBLE)"
        raw = f"NULLIF(trim({_quote_ident(source)}), '')"
        if col == ID_COLUMN:
            return f"TRY_CAST(TRY_CAST({raw} AS DOUBLE) AS BIGINT)"
        if col in PERCENT_STRING_COLUMNS:
            return f"TRY_CAST(trim(replace({raw}, '%', '')) AS DOUBLE) / 100.0"
        if col in PERCENT_NUMERIC_COLUMNS:
            return f"TRY_CAST({raw} AS DOUBLE) / 100.0"
        if col == "term":
            return f"TRY_CAST(regexp_extract({raw}, '(\\d+)', 1) AS BIGINT)"
        if col == "emp_length":
            # NB: no least()/greatest() here - DuckDB's versions ignore NULLs.
            return (
                f"CASE WHEN {raw} IS NULL THEN NULL WHEN {raw} LIKE '<%' THEN 0.0 "
                f"ELSE TRY_CAST(regexp_extract({raw}, '(\\d+)', 1) AS DOUBLE) END"
            )
        if col in DATE_COLUMNS:
            parts = ", ".join(f"try_strptime({raw}, '{fmt}')" for fmt in DATE_FORMATS)
            return f"CAST(COALESCE({parts}) AS TIMESTAMP_NS)"
        if col == "home_ownership":
            return f"CASE WHEN upper({raw}) IN ('ANY', 'NONE') THEN 'OTHER' ELSE upper({raw}) END"
        if col in CATEGORICAL_COLUMNS:
            return raw
        return f"TRY_CAST({raw} AS DOUBLE)"

    def _parse_select(self) -> str:
        """SELECT that turns the raw CSV into canonical, cleaned rows (all columns)."""
        mapping = self.column_mapping()
        exprs = [
            f"{self.column_expression(col, mapping[col])} AS {_quote_ident(col)}"
            for col in CANONICAL_COLUMNS
        ]
        has_policy = "policy_code" in set(self.source_columns())
        policy = (
            "TRY_CAST(NULLIF(trim(\"policy_code\"), '') AS DOUBLE)" if has_policy else "CAST(NULL AS DOUBLE)"
        )
        exprs.append(f"{policy} AS __policy_code")
        not_null = " AND ".join(f"{_quote_ident(c)} IS NOT NULL" for c in REQUIRED_NON_NULL)
        return (
            f"SELECT * EXCLUDE (__policy_code) FROM (SELECT {', '.join(exprs)} FROM {self._csv_source()}) "
            f"WHERE {not_null} AND (__policy_code IS NULL OR __policy_code = 1)"
        )

    def build_query(
        self,
        *,
        max_rows: int | None = None,
        sample_fraction: float | None = None,
        seed: int = 42,
        issue_from: Any = None,
        issue_to: Any = None,
        columns: Sequence[str] | None = None,
        order: OrderMode = "issue_d",
        source: Literal["csv", "parquet"] = "csv",
    ) -> tuple[str, dict[str, Any]]:
        """SQL + parameters for :meth:`load` (exposed for inspection and tests)."""
        if order not in ("issue_d", "random", "file"):
            raise DataSourceError(f"order must be 'issue_d', 'random' or 'file', got {order!r}")
        if max_rows is not None and (int(max_rows) != max_rows or max_rows < 1):
            raise DataSourceError("max_rows must be a positive integer")
        if sample_fraction is not None and not 0 < float(sample_fraction) <= 1:
            raise DataSourceError("sample_fraction must be in (0, 1]")
        cols = _resolve_columns(columns)
        params: dict[str, Any] = {}
        if source == "parquet":
            base = "SELECT * FROM read_parquet($cache)"
            params["cache"] = str(self.cache_path)
        else:
            base = self._parse_select()
            params["path"] = str(self.path)
        sampling = sample_fraction is not None and float(sample_fraction) < 1
        need_hash = sampling or order == "random"
        if need_hash:
            base = f"SELECT *, {_hash_columns_sql(seed)} FROM ({base}) AS raw"
        where: list[str] = []
        lo = _to_timestamp(issue_from, "issue_from")
        hi = _to_timestamp(issue_to, "issue_to")
        if lo is not None:
            where.append("issue_d >= $issue_from")
            params["issue_from"] = lo.to_pydatetime()
        if hi is not None:
            where.append("issue_d <= $issue_to")
            params["issue_to"] = hi.to_pydatetime()
        if sampling:
            where.append("__bucket < $threshold")
            params["threshold"] = int(round(float(sample_fraction) * HASH_BUCKETS))
        select_cols = ", ".join(_quote_ident(c) for c in cols)
        sql = f"SELECT {select_cols} FROM ({base}) AS src"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if order == "issue_d":
            sql += " ORDER BY issue_d, loan_id"
        elif order == "random":
            sql += " ORDER BY __hash, loan_id"
        if max_rows is not None:
            sql += f" LIMIT {int(max_rows)}"
        return sql, params

    # ----- loading --------------------------------------------------------------------
    def load(
        self,
        max_rows: int | None = None,
        sample_fraction: float | None = None,
        seed: int = 42,
        issue_from: Any = None,
        issue_to: Any = None,
        columns: Sequence[str] | None = None,
        order: OrderMode = "issue_d",
        use_cache: bool | None = None,
    ) -> pd.DataFrame:
        """Load a canonical LoanTable.

        ``order`` decides which rows ``max_rows`` keeps: ``"issue_d"`` (default) sorts by
        ``(issue_d, loan_id)`` and so returns the *earliest* loans - the ones most likely to
        be resolved by the 2019-03 snapshot, but concentrated in 2007-2013 (the first 200k
        loans end around late 2013). ``"random"`` orders by the seeded MD5 hash (a
        reproducible random subset over all years); ``"file"`` keeps file order and stops
        reading as soon as ``max_rows`` rows qualify (fastest, never builds the cache).
        ``sample_fraction`` (deterministic per ``seed``) and ``issue_from`` / ``issue_to``
        (inclusive) are applied before ``max_rows``.
        """
        use_cache = self.cache if use_cache is None else use_cache
        self._require_file()
        source: Literal["csv", "parquet"] = "csv"
        if use_cache and not (order == "file" and max_rows is not None):
            if not self.cache_is_fresh():
                try:
                    self.build_cache()
                except (OSError, duckdb.Error, DataSourceError) as exc:
                    warnings.warn(f"parquet cache unavailable ({exc}); reading the CSV directly", stacklevel=2)
            if self.cache_is_fresh():
                source = "parquet"
        sql, params = self.build_query(
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            issue_from=issue_from,
            issue_to=issue_to,
            columns=columns,
            order=order,
            source=source,
        )
        t0 = time.perf_counter()
        con = self._connect()
        try:
            raw = con.execute(sql, params).to_arrow_table().to_pandas()
        except duckdb.Error as exc:
            raise DataSourceError(f"failed to load {self.path}: {exc}") from exc
        finally:
            con.close()
        self.last_query = sql
        self.last_source = source
        out = canonicalize(raw, columns)
        self.last_load_seconds = time.perf_counter() - t0
        return out

    # ----- parquet cache ----------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return self._cache_dir
        parent = self.path.parent
        processed = parent.parent / "processed"
        if parent.name == "raw" and processed.is_dir():
            return processed
        return parent

    @property
    def cache_path(self) -> Path:
        name = self.path.name
        for suffix in (".gz", ".csv"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return self.cache_dir / f"{name}.canonical.v{LOADER_VERSION}.parquet"

    @property
    def _cache_meta_path(self) -> Path:
        return self.cache_path.with_suffix(".json")

    def _source_signature(self) -> dict[str, Any]:
        st = self.path.stat()
        return {
            "source": str(self.path.resolve()),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "loader_version": LOADER_VERSION,
        }

    def cache_is_fresh(self) -> bool:
        """True when the parquet cache exists and was built from the current source file."""
        if not (self.cache_path.is_file() and self._cache_meta_path.is_file()):
            return False
        try:
            meta = json.loads(self._cache_meta_path.read_text())
            sig = self._source_signature()
        except (OSError, ValueError):
            return False
        return all(meta.get(k) == v for k, v in sig.items())

    def build_cache(self, overwrite: bool = False) -> Path:
        """Parse the full CSV once into a zstd parquet file of the canonical table."""
        self._require_file()
        if self.cache_is_fresh() and not overwrite:
            return self.cache_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # unique temp names + atomic os.replace: concurrent builders never see partial files
        tmp = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        t0 = time.perf_counter()
        con = self._connect()
        try:
            con.execute(
                f"COPY ({self._parse_select()} ORDER BY issue_d, loan_id) TO $out "
                "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880)",
                {"path": str(self.path), "out": str(tmp)},
            )
            n_rows = con.execute("SELECT count(*) FROM read_parquet($out)", {"out": str(tmp)}).fetchone()[0]
        except duckdb.Error as exc:
            tmp.unlink(missing_ok=True)
            raise DataSourceError(f"failed to build parquet cache from {self.path}: {exc}") from exc
        finally:
            con.close()
        os.replace(tmp, self.cache_path)
        meta = {
            **self._source_signature(),
            "n_rows": int(n_rows),
            "build_seconds": round(time.perf_counter() - t0, 2),
            "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        meta_tmp = tmp.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps(meta, indent=2))
        os.replace(meta_tmp, self._cache_meta_path)
        return self.cache_path

    @staticmethod
    def to_parquet(df: pd.DataFrame, path: str | os.PathLike[str]) -> Path:
        """Write a (canonical, optionally outcome-augmented) LoanTable to parquet."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, engine="pyarrow", compression="zstd")
        return out

    @staticmethod
    def from_parquet(path: str | os.PathLike[str], columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Read a parquet LoanTable back with canonical dtypes (``y_*`` columns kept)."""
        p = Path(path)
        if not p.is_file():
            raise DataSourceError(f"parquet file not found: {p}")
        df = pd.read_parquet(p, engine="pyarrow")
        if columns is None:
            retained = [c for c in CANONICAL_COLUMNS if c in df.columns]
            return canonicalize(df, retained, keep_extra=True)
        return canonicalize(df, columns)

    # ----- description ------------------------------------------------------------------
    def describe_source(self, count_rows: bool = True) -> dict[str, Any]:
        """Quick facts about the source: existence, size, retained-column presence, rows.

        Column presence comes from the sniffed header (fast). ``n_rows_raw`` is a
        ``count(*)`` over the file (one decompression pass, a few seconds); when a fresh
        cache exists ``n_rows_canonical`` (after cleaning) is read from parquet metadata.
        """
        info: dict[str, Any] = {
            "path": str(self.path),
            "exists": self.path.is_file(),
            "size_bytes": None,
            "n_source_columns": None,
            "retained_present": {},
            "missing_retained": [],
            "renamed": {},
            "n_rows_raw": None,
            "n_rows_canonical": None,
            "cache_path": str(self.cache_path),
            "cached": False,
        }
        if not info["exists"]:
            info["hint"] = "run scripts/download_data.sh or use source='synthetic'"
            return info
        info["size_bytes"] = self.path.stat().st_size
        header = self.source_columns()
        mapping = self.column_mapping()
        info["n_source_columns"] = len(header)
        info["retained_present"] = {c: mapping[c] is not None for c in CANONICAL_COLUMNS}
        info["missing_retained"] = [c for c in CANONICAL_COLUMNS if mapping[c] is None]
        info["renamed"] = {src: c for c, src in mapping.items() if src is not None and src != c}
        info["cached"] = self.cache_is_fresh()
        con = self._connect()
        try:
            if info["cached"]:
                info["n_rows_canonical"] = int(
                    con.execute(
                        "SELECT count(*) FROM read_parquet($p)", {"p": str(self.cache_path)}
                    ).fetchone()[0]
                )
            if count_rows:
                info["n_rows_raw"] = int(
                    con.execute(f"SELECT count(*) FROM {self._csv_source()}", {"path": str(self.path)}).fetchone()[0]
                )
        except duckdb.Error as exc:
            raise DataSourceError(f"cannot describe {self.path}: {exc}") from exc
        finally:
            con.close()
        return info
