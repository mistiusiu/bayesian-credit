#!/usr/bin/env python
"""Generate a synthetic Lending Club LoanTable (SPEC §1.3) and write it to parquet.

Examples
--------
    python scripts/make_synthetic.py --n 50000 --out data/processed/synthetic_50k.parquet
    python scripts/make_synthetic.py --n 200000 --seed 7 --shift            # default shift at 2016-06
    python scripts/make_synthetic.py --shift '{"fico_range_low": -25, "dti": 5}' --shift-at 2017-01
    python scripts/make_synthetic.py --n 20000 --with-outcomes --out data/processed/synth_y.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import bayesian_credit  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e core`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from bayesian_credit.config import DataConfig
from bayesian_credit.data import (
    DEFAULT_SHIFT,
    DEFAULT_SHIFT_AT,
    LendingClubLoader,
    SyntheticLendingClub,
    derive_outcomes,
    summarize,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000, help="number of loans (default 100000)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic.parquet"), help="output parquet path")
    ap.add_argument("--start", default="2012-01", help="first issue month (default 2012-01)")
    ap.add_argument("--end", default="2018-12", help="last issue month (default 2018-12)")
    ap.add_argument("--snapshot", default=DataConfig().snapshot_date, help="observation date (default 2019-03-31)")
    ap.add_argument(
        "--shift",
        nargs="?",
        const="default",
        default=None,
        help=f"distribution shift: bare flag = {json.dumps(DEFAULT_SHIFT)}, or a JSON object of feature -> delta "
        "(plus optional grade_mix / latent)",
    )
    ap.add_argument("--shift-at", default=None, help=f"first affected issue month (default {DEFAULT_SHIFT_AT} with --shift)")
    ap.add_argument("--with-outcomes", action="store_true", help="append the y_* outcome columns")
    ap.add_argument("--quiet", action="store_true", help="do not print the JSON summary")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shift = None
    if args.shift is not None:
        shift = dict(DEFAULT_SHIFT) if args.shift == "default" else json.loads(args.shift)
        if not isinstance(shift, dict):
            raise SystemExit("--shift must be a JSON object")
    shift_at = args.shift_at or (DEFAULT_SHIFT_AT if shift is not None else None)

    gen = SyntheticLendingClub(seed=args.seed)
    t0 = time.perf_counter()
    df = gen.generate(args.n, start=args.start, end=args.end, snapshot=args.snapshot, shift_at=shift_at, shift=shift)
    gen_seconds = time.perf_counter() - t0
    if args.with_outcomes:
        df = derive_outcomes(df, cfg=DataConfig(snapshot_date=args.snapshot))
    out = LendingClubLoader.to_parquet(df, args.out)

    if not args.quiet:
        summary = summarize(df, DataConfig(snapshot_date=args.snapshot))
        report = {
            "out": str(out),
            "rows": len(df),
            "columns": df.shape[1],
            "size_mb": round(out.stat().st_size / 1e6, 2),
            "generate_seconds": round(gen_seconds, 2),
            "generator": gen.last_meta,
            "summary": {k: summary[k] for k in ("issue_min", "issue_max", "status_counts", "resolved_share",
                                                "default_rate", "mean_recovery", "default_rate_by_grade")},
        }
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
