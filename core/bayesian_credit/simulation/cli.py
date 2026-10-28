"""Command-line entry points used by the Django API's subprocess jobs (SPEC §3).

    python -m bayesian_credit.simulation.cli fit    --dataset <parquet> --config <json> --out <fit_dir> [--train-end YYYY-MM-DD]
    python -m bayesian_credit.simulation.cli run    --run-dir <dir>        # config.json must exist
    python -m bayesian_credit.simulation.cli resume --run-dir <dir>        # continue from the latest checkpoint
    python -m bayesian_credit.simulation.cli shadow --fit-dir <dir> --dataset <parquet> --out <dir> [--max-loans N --seed S]
    python -m bayesian_credit.simulation.cli report --run-dir <dir>

Every command prints one JSON object on stdout (the API stores it in ``Job.log_tail``) and
exits 0 on success (also for a cancelled run), 1 on failure, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..config import FrameworkConfig
from ..exceptions import BayesianCreditError
from .events import jsonable


def _config_from_file(path: str | None) -> FrameworkConfig:
    if not path:
        return FrameworkConfig()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "framework" in data:
        data = data["framework"]
    return FrameworkConfig.from_dict(data)


def _print(obj: Any) -> None:
    print(json.dumps(jsonable(obj), indent=2, allow_nan=False))


def cmd_fit(args: argparse.Namespace) -> int:
    from ..pipeline import fit_models, load_dataset

    t0 = time.perf_counter()
    cfg = _config_from_file(args.config)
    df = load_dataset(args.dataset)
    fitted = fit_models(df, cfg, args.train_end or cfg.train_end, evaluate=not args.no_evaluate)
    out = fitted.save(args.out)
    m = fitted.metrics
    _print(
        {
            "fit_dir": str(out),
            "train_end": fitted.train_end,
            "seconds": round(time.perf_counter() - t0, 3),
            "metrics": {k: m.get(k) for k in ("n_train", "n_train_resolved", "n_holdout", "fit_seconds", "brier", "auc", "ece",
                                               "log_loss", "recovery_mae", "recovery_rmse", "transition_log_likelihood", "n_test")},
        }
    )
    return 0


def _result_dict(res: Any) -> dict[str, Any]:
    return {
        "status": res.status,
        "run_dir": str(res.run_dir),
        "epochs_completed": res.epochs_completed,
        "n_epochs": res.n_epochs,
        "n_decisions": res.n_decisions,
        "wall_seconds": res.wall_seconds,
        "last_epoch": res.last_epoch,
    }


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import SimulationRunner

    runner = SimulationRunner.from_run_dir(args.run_dir, poll_interval=args.poll_interval)
    res = runner.run()
    _print(_result_dict(res))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from .runner import SimulationRunner

    res = SimulationRunner.resume(args.run_dir, poll_interval=args.poll_interval)
    _print(_result_dict(res))
    return 0


def cmd_shadow(args: argparse.Namespace) -> int:
    from ..pipeline import FittedModels, load_dataset, run_shadow_portfolio
    from .store import atomic_write_json, atomic_write_parquet

    t0 = time.perf_counter()
    fitted = FittedModels.load(args.fit_dir)
    df = load_dataset(args.dataset)
    per_loan, summary = run_shadow_portfolio(
        fitted, df, max_loans=args.max_loans, seed=args.seed, n_samples=args.n_samples, holdout_only=not args.all_loans
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(out / "per_loan.parquet", per_loan, fsync=False)
    summ = summary.to_dict()
    atomic_write_json(out / "summary.json", summ, fsync=False)
    _print(
        {
            "out": str(out),
            "seconds": round(time.perf_counter() - t0, 3),
            "summary": {k: summ[k] for k in ("n_loans", "n_lendable", "share_lendable", "expected_profit_historical",
                                              "expected_profit_shadow", "uplift", "uplift_pct", "counts")},
        }
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ..pipeline import build_report

    rep = build_report(args.run_dir)
    _print({"report": str(Path(args.run_dir) / "report.json"), "status": rep["status"], "policy_summary": rep["policy_summary"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bayesian_credit.simulation.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="fit the Two-Stage Beta-Beta model + MCTM and save the fit")
    p.add_argument("--dataset", required=True, help="LoanTable parquet (from prepare_dataset)")
    p.add_argument("--config", default=None, help="FrameworkConfig JSON (or {'framework': {...}})")
    p.add_argument("--out", required=True, help="output fit directory")
    p.add_argument("--train-end", default=None, help="fit on issue_d < train_end (default: config.train_end)")
    p.add_argument("--no-evaluate", action="store_true", help="skip the holdout predictive evaluation")
    p.set_defaults(func=cmd_fit)

    for name, func, helptext in (("run", cmd_run, "run a prepared run directory"), ("resume", cmd_resume, "resume from the latest checkpoint")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--poll-interval", type=float, default=1.0, help="seconds between control/ack polls")
        p.set_defaults(func=func)

    p = sub.add_parser("shadow", help="shadow-engine portfolio evaluation")
    p.add_argument("--fit-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-loans", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--all-loans", action="store_true", help="include training-window loans (default: holdout only)")
    p.set_defaults(func=cmd_shadow)

    p = sub.add_parser("report", help="(re)build report.json of a run directory")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (BayesianCreditError, FileNotFoundError, ValueError) as exc:
        _print({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)})
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
