#!/usr/bin/env python
"""End-to-end pipeline: ingest → fit → predictive evaluation → shadow portfolio →
replay simulation (Thompson Sampling vs baselines) → kill-and-resume check → Table 11 report.

Examples
--------
    python scripts/run_pipeline.py --source synthetic                       # 200k synthetic loans (with a 2016-06 shift)
    python scripts/run_pipeline.py --source synthetic --n 50000 --max-loans 10000 --no-shift
    python scripts/run_pipeline.py --source lending_club --max-rows 300000 --max-loans 50000

Outputs (``--out``, default ``data/runs/<timestamp>``)::

    dataset.parquet  fit/  shadow/{per_loan.parquet,summary.json}  simulation/ (run directory)
    resilience/  report.json  summary.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import bayesian_credit  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e core`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from bayesian_credit.config import FrameworkConfig, SimulationConfig
from bayesian_credit.data import DEFAULT_SHIFT, DEFAULT_SHIFT_AT
from bayesian_credit.pipeline import (
    build_report,
    fit_models,
    peak_rss_mb,
    prepare_dataset,
    resilience_check,
    run_shadow_portfolio,
    run_simulation,
)
from bayesian_credit.simulation.store import RunDirectory, atomic_write_json

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("synthetic", "lending_club"), default="synthetic")
    ap.add_argument("--n", type=int, default=200_000, help="synthetic: loans to generate (default 200000)")
    ap.add_argument("--max-rows", type=int, default=200_000, help="lending_club: random subset size (default 200000)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-end", default="2015-01-01")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--max-loans", type=int, default=50_000, help="loans streamed through the simulation")
    ap.add_argument("--policies", default="thompson,thompson_se,epsilon_greedy,ucb1,historical,random")
    ap.add_argument("--feedback-mode", choices=("simulated", "replay", "hybrid"), default="simulated")
    ap.add_argument("--breaker-mode", choices=("auto_ack", "pause", "off"), default="auto_ack")
    ap.add_argument("--auto-action", choices=("resume", "resegment", "rollback"), default=None,
                    help="action applied by auto_ack (default: PSIConfig.auto_action = resume)")
    ap.add_argument("--interim-months", type=int, default=None, help="default: SimulationConfig.interim_months (12)")
    ap.add_argument("--checkpoint-every", type=int, default=None, help="default: SimulationConfig.checkpoint_every (6)")
    ap.add_argument("--risk-aversion", type=float, default=None, help="λ in mean − λ·std (default: RewardConfig, 0)")
    ap.add_argument("--shadow-max-loans", type=int, default=200_000, help="holdout loans scored by the shadow engine")
    ap.add_argument("--no-shift", action="store_true", help="synthetic: no distribution shift")
    ap.add_argument("--no-resilience-check", action="store_true", help="skip the kill-and-resume determinism check")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default data/runs/<timestamp>)")
    return ap.parse_args(argv)


def _fmt_money(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.1f}k"
    return f"${v:,.0f}"


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x: Any, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100 * float(x):.{nd}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _dist(d: dict[str, Any] | None) -> str:
    if not d:
        return "n/a"
    return ", ".join(f"{k} {_pct(v, 0)}" for k, v in d.items())


def risk_probe(fitted: Any) -> dict[str, Any]:
    """p(default) of the busiest grade-A group (training data) at 7 % vs 28 % and 36 vs 60 m."""
    hm = fitted.two_stage.default_model
    groups = [g for g in hm.groups if g.split("|")[0] == "A"]
    if not groups:
        return {}
    g = max(groups, key=lambda k: sum(hm.group_counts(k)))
    space = fitted.arm_space
    out: dict[str, Any] = {"group": g, "group_mean": hm.group_posterior(g).mean}
    for label, (amount, rate, term) in {
        "10k@7%/36m": (10_000, 0.07, 36),
        "10k@28%/36m": (10_000, 0.28, 36),
        "10k@7%/60m": (10_000, 0.07, 60),
        "40k@28%/60m": (40_000, 0.28, 60),
    }.items():
        arm = space.nearest(amount, rate, term)
        est = fitted.two_stage.risk(g, arm)
        out[label] = {"p": est.p_default.mean, "n": est.n_default_obs, "tilt": hm.prior_tilt(g, arm)}
    return out


def summarise(report: dict[str, Any], timings: dict[str, float], ds: dict[str, Any], args: argparse.Namespace,
              out: Path, fit_metrics: dict[str, Any], probe: dict[str, Any] | None = None) -> str:
    dims = report["dimensions"]
    lines: list[str] = []
    src = f"synthetic n={args.n:,}" if args.source == "synthetic" else f"lending_club max_rows={args.max_rows:,}"
    lines.append(f"Bayesian Credit pipeline - {src}, seed {args.seed}")
    lines.append("=" * 100)
    lines.append(
        f"data      {ds.get('n', 0):,} loans {ds.get('issue_min')}..{ds.get('issue_max')}; resolved {_pct(ds.get('resolved_share'))}; "
        f"default rate (resolved) {_pct(ds.get('default_rate'))}; mean recovery {_fmt(ds.get('mean_recovery'), 3)}"
    )
    lines.append(
        f"fit       issue_d < {fit_metrics.get('train_end')}: {fit_metrics.get('n_train', 0):,} loans "
        f"({fit_metrics.get('n_train_resolved', 0):,} resolved), {fit_metrics.get('hierarchy', {}).get('n_groups')} groups, "
        f"{fit_metrics.get('hierarchy', {}).get('n_leaves')} leaves; fit {fit_metrics.get('fit_seconds')} s"
    )
    pe = fit_metrics.get("policy_effect") or {}
    if pe:
        beta = pe.get("beta") or {}
        se = pe.get("std_errors") or {}
        coef = ", ".join(f"{k} {v:+.3f}" + (f" (se {se[k]:.3f})" if se.get(k) is not None else "") for k, v in beta.items())
        lines.append(
            f"policy    leaf-prior tilt {pe.get('status')}: beta {coef or 'n/a'}; n_fit {pe.get('n_fit') or 0:,} "
            f"({pe.get('sample')}), clip +/-{pe.get('max_abs_logit_shift')}"
        )
    if probe:
        parts = [f"{k} p={v['p']:.3f} (tilt {v['tilt']:+.2f}, n={v['n']})" for k, v in probe.items() if isinstance(v, dict)]
        lines.append(f"          grade-A check {probe['group']} (group mean {probe['group_mean']:.3f}): " + "; ".join(parts))
    ir = fit_metrics.get("interest_realization") or {}
    if ir:
        grades = ", ".join(
            f"{g} " + "/".join(f"{_fmt(v, 2)}" for _, v in sorted(ts.items())) for g, ts in sorted((ir.get("by_grade_term") or {}).items())
        )
        lines.append(
            f"prepay    realised share of scheduled interest ({ir.get('prepayment_adjustment')}; {ir.get('sample')}, n={ir.get('n_fit') or 0:,}): "
            + ", ".join(f"{t}m {_fmt(v, 3)}" for t, v in sorted((ir.get("by_term") or {}).items()))
            + f"; by grade 36m/60m: {grades}"
        )
    gr = fit_metrics.get("guardrails") or {}
    if gr:
        ref = fit_metrics.get("reference_rates") or {}
        by_grade = ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted((ref.get("by_grade") or {}).items()))
        lines.append(
            f"guards    payment/income <= {gr.get('max_payment_to_income')}, amount/income <= {gr.get('max_amount_to_income')}, "
            f"total DTI <= {gr.get('max_total_dti')}, rate within +/-{gr.get('rate_corridor')} of the sub-grade reference "
            f"(grade means: {by_grade or 'n/a'})"
        )
    dp = dims["default_prediction"]
    if dp.get("available"):
        p, s = dp["primary"], dp["secondary"]
        lines.append(
            f"default   primary (resolved, term elapsed; n={p.get('n', 0):,}, base rate {_pct(p.get('base_rate'))}): "
            f"Brier {_fmt(p.get('brier'))} (climatology {_fmt(p.get('brier_climatology'))}), AUC {_fmt(p.get('auc'), 3)}, "
            f"ECE {_fmt(p.get('ece'))}, log loss {_fmt(p.get('log_loss'))}"
        )
        lines.append(
            f"          secondary (all resolved; n={s.get('n', 0):,}, base rate {_pct(s.get('base_rate'))}): "
            f"Brier {_fmt(s.get('brier'))}, AUC {_fmt(s.get('auc'), 3)}, ECE {_fmt(s.get('ece'))}, log loss {_fmt(s.get('log_loss'))}"
        )
        vint = dp.get("by_vintage") or {}
        if vint:
            parts = [f"{y}: AUC {_fmt(v.get('auc'), 3)} n={v.get('n', 0):,}" for y, v in sorted(vint.items())]
            lines.append("          by vintage (all resolved): " + "; ".join(parts))
    rp = dims["recovery_prediction"]
    if rp.get("available"):
        r1, r2 = rp.get("primary", {}), rp.get("secondary", {})
        rd = fit_metrics.get("recovery") or {}
        src = ", ".join(f"{k} {rd.get('source_' + k, 0)}" for k in ("leaf", "group_term", "group", "global", "prior"))
        lines.append(
            f"recovery  primary MAE {_fmt(r1.get('mae'))} RMSE {_fmt(r1.get('rmse'))} (n={r1.get('n', 0):,}); "
            f"all resolved MAE {_fmt(r2.get('mae'))} RMSE {_fmt(r2.get('rmse'))} (n={r2.get('n', 0):,}); "
            f"leaf sources: {src}"
        )
    cs = dims["credit_state_modeling"]
    if cs.get("available"):
        h = (cs.get("holdout") or {}).get("all", {})
        lines.append(
            f"MCTM      holdout log-lik/transition {_fmt(h.get('log_likelihood_per_transition'))} over {h.get('n_transitions', 0):,.0f} "
            f"transitions; row L1 error C/D {h.get('row_error')}"
        )
    ce = dims["counterfactual_evaluation"]
    if ce.get("available"):
        c = ce.get("counts", {})
        lines.append(
            f"shadow    {ce.get('n_loans', 0):,} holdout loans in {ce.get('seconds')} s: expected profit historical "
            f"{_fmt_money(ce.get('expected_profit_historical'))} vs shadow {_fmt_money(ce.get('expected_profit_shadow'))} "
            f"(uplift {_fmt_money(ce.get('uplift'))}, {_fmt(ce.get('uplift_pct'), 1)}%); lendable {_pct(ce.get('share_lendable'))}; "
            f"realized historical {_fmt_money(ce.get('realized_profit_historical'))} on {ce.get('n_realized', 0):,} resolved"
        )
        lines.append(
            f"          classes: declined {c.get('declined', 0):,}, rescued {c.get('rescued', 0):,}, upsized {c.get('upsized', 0):,}, "
            f"downsized {c.get('downsized', 0):,}, repriced down {c.get('repriced_down', 0):,}, repriced up {c.get('repriced_up', 0):,}, "
            f"unchanged {c.get('unchanged', 0):,}"
        )
        dec = ce.get("uplift_decomposition") or {}
        if dec:
            parts = []
            for key, label in (("repricing_only", "repricing only (historical term & amount)"),
                               ("repricing_and_resizing", "+ resizing (historical term)"), ("unrestricted", "+ term choice")):
                v = dec.get(key) or {}
                parts.append(f"{label} {_fmt_money(v.get('uplift'))} ({_fmt(v.get('uplift_pct'), 1)}%)")
            lines.append("          uplift decomposition: " + "; ".join(parts))
        mix = ce.get("recommendation_mix") or {}
        if mix:
            rec, hist, ref = mix.get("recommended") or {}, mix.get("historical") or {}, mix.get("reference") or {}
            lines.append(
                f"          mix (recommended vs historical): 36m {_pct(mix.get('share_36m_recommended'))} vs "
                f"{_pct(mix.get('share_36m_historical'))}; mean rate {_pct(mix.get('mean_rate_recommended'), 2)} vs "
                f"{_pct(mix.get('mean_rate_historical'), 2)} (reference {_pct(ref.get('mean_reference_rate'), 2)}, "
                f"recommended - reference {_pct(ref.get('mean_rate_minus_reference'), 2)}, above reference "
                f"{_pct(ref.get('share_above_reference'))}); mean amount {_fmt_money(mix.get('mean_amount_recommended'))} vs "
                f"{_fmt_money(mix.get('mean_amount_historical'))}; at max rate {_pct(mix.get('share_at_max_rate_recommended'))}, "
                f"at max amount {_pct(mix.get('share_at_max_amount_recommended'))}"
            )
            lines.append(f"          rate buckets   recommended: {_dist(rec.get('by_rate'))} | historical: {_dist(hist.get('by_rate'))}")
            lines.append(f"          amount buckets recommended: {_dist(rec.get('by_amount'))} | historical: {_dist(hist.get('by_amount'))}")
        sweep = ce.get("by_risk_aversion") or {}
        if sweep:
            lines.append(
                "          risk appetite (lambda in mean - lambda*std): "
                + "; ".join(
                    f"lambda={v.get('risk_aversion')}: uplift {_fmt(v.get('uplift_pct'), 1)}%, 36m {_pct(v.get('share_36m_recommended'))}, "
                    f"amount {_fmt_money(v.get('mean_amount_recommended'))}, rate {_pct(v.get('mean_rate_recommended'), 1)}, "
                    f"decline {_pct(v.get('decline_share'))}"
                    for v in sweep.values()
                )
            )
        incl = ce.get("inclusion") or []
        if incl:
            lines.append(
                "          lendable by grade: "
                + ", ".join(f"{r.get('grade')} {_pct(r.get('share_lendable'))}" for r in incl)
                + "; shadow vs historical mean rate by grade: "
                + ", ".join(f"{r.get('grade')} {_pct(r.get('mean_rate_shadow'), 1)}/{_pct(r.get('mean_rate_historical'), 1)}" for r in incl)
            )
    cf = dims["computational_feasibility"]
    lines.append(
        f"simulate  {cf.get('n_epochs')} epochs, {cf.get('n_loans_streamed') or 0:,} loans, {cf.get('n_decisions', 0):,} decisions in "
        f"{_fmt(cf.get('simulation_wall_seconds'), 1)} s ({_fmt(cf.get('decisions_per_second'), 0)} decisions/s); "
        f"feedback={args.feedback_mode}, breaker={args.breaker_mode}"
    )
    pp = dims["policy_performance"].get("by_policy", {})
    bp = dims["bandit_performance"].get("by_policy", {})
    header = f"          {'policy':<16}{'cum. expected':>15}{'uplift vs hist':>16}{'cum. regret':>14}{'avg regret':>12}{'decline':>9}{'fallback':>9}{'realized':>13}{'top-3 share':>12}"
    lines.append(header)
    for name, row in pp.items():
        b = bp.get(name, {})
        lines.append(
            f"          {name:<16}{_fmt_money(row.get('total_expected_reward')):>15}{_fmt_money(row.get('uplift_vs_historical')):>16}"
            f"{_fmt_money(b.get('final_cumulative_regret')):>14}{_fmt_money(b.get('average_regret')):>12}"
            f"{_pct(row.get('decline_rate')):>9}{_pct(row.get('fallback_rate')):>9}{_fmt_money(row.get('total_realized_reward')):>13}"
            f"{_pct(b.get('selection_concentration_top3')):>12}"
        )
    lines.append("          selection mix, last 6 epochs (lending decisions): 36m share / mean rate / mean amount / at max rate / at max amount; top-3 arms")
    for name in pp:
        b = bp.get(name, {})
        m = b.get("selection_mix") or {}
        tops = ", ".join(x for x in (b.get("top_arm_labels_recent") or []) if x)
        lines.append(
            f"          {name:<16}{_pct(m.get('share_36m')):>7} {_pct(m.get('mean_rate'), 1):>7} {_fmt_money(m.get('mean_amount')):>8} "
            f"{_pct(m.get('share_at_max_rate')):>7} {_pct(m.get('share_at_max_amount')):>7}   {tops}"
        )
    dm = dims["distributional_monitoring"]
    lines.append(
        f"sentinel  max aggregate PSI {_fmt(dm.get('max_aggregate_psi'), 3)} (first breaker-level epoch {dm.get('first_breaker_epoch')}); "
        f"epochs normal/watch/breaker {dm.get('epochs_normal')}/{dm.get('epochs_watch')}/{dm.get('epochs_breaker_level')}; "
        f"open {dm.get('epochs_breaker_open')}; trips {dm.get('n_trips')}; actions {dm.get('actions')}"
    )
    feats = dm.get("per_feature") or {}
    if feats:
        lines.append("          max PSI by feature: " + ", ".join(f"{k} {_fmt(v.get('max_psi'), 3)}" for k, v in feats.items()))
    ir = dims["infrastructure_resilience"]
    rt = ir.get("resume_test") or {}
    if rt:
        lines.append(
            f"resilience kill after epoch index {rt.get('killed_after_epoch_index')} + resume from checkpoint -> decisions identical "
            f"{rt.get('decisions_identical')}, timeseries identical {rt.get('timeseries_identical')} (recovery {rt.get('recovery_seconds')} s)"
        )
    lines.append(f"          checkpoints written {ir.get('checkpoints_written')}, restarts {ir.get('restarts')}")
    t_parts = ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
    lines.append(f"timings   {t_parts}; total {sum(timings.values()):.1f}s; peak RSS {peak_rss_mb():.0f} MB")
    lines.append(f"outputs   {out}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    out = args.out or (REPO / "data" / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out = out if out.is_absolute() else (Path.cwd() / out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = FrameworkConfig(train_end=args.train_end)
    if args.risk_aversion is not None:
        cfg.reward.risk_aversion = float(args.risk_aversion)
    if args.auto_action is not None:
        cfg.psi.auto_action = args.auto_action
    cfg.validate()
    sim_kwargs: dict[str, Any] = {
        "start": args.start,
        "end": args.end,
        "max_loans": args.max_loans,
        "seed": args.seed,
        "feedback_mode": args.feedback_mode,
        "policies": tuple(p.strip() for p in args.policies.split(",") if p.strip()),
        "breaker_mode": args.breaker_mode,
    }
    if args.interim_months is not None:
        sim_kwargs["interim_months"] = args.interim_months
    if args.checkpoint_every is not None:
        sim_kwargs["checkpoint_every"] = args.checkpoint_every
    sim_cfg = SimulationConfig(**sim_kwargs)
    sim_cfg.validate()
    timings: dict[str, float] = {}

    # 1. ingest ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    if args.source == "synthetic":
        kw: dict[str, Any] = {"n": args.n, "seed": args.seed}
        if not args.no_shift:
            kw.update(shift_at=DEFAULT_SHIFT_AT, shift=dict(DEFAULT_SHIFT))
    else:
        kw = {"max_rows": args.max_rows, "seed": args.seed, "order": "random"}
    df, ds = prepare_dataset(args.source, out / "dataset.parquet", cfg=cfg, **kw)
    timings["ingest"] = time.perf_counter() - t0
    print(f"[ingest] {len(df):,} loans in {timings['ingest']:.1f}s -> {ds['path']}", flush=True)

    # 2. fit (+ holdout predictive evaluation) ---------------------------------------------
    t0 = time.perf_counter()
    fitted = fit_models(df, cfg, args.train_end)
    fit_dir = fitted.save(out / "fit")
    timings["fit"] = fitted.metrics.get("fit_seconds", 0.0)
    timings["predictive"] = fitted.metrics.get("evaluate_seconds", 0.0)
    timings["fit_io"] = max(time.perf_counter() - t0 - timings["fit"] - timings["predictive"], 0.0)
    print(f"[fit] {fitted.metrics['n_train']:,} training loans; AUC(primary) {_fmt(fitted.metrics.get('auc'), 3)} "
          f"Brier {_fmt(fitted.metrics.get('brier'))} in {time.perf_counter() - t0:.1f}s", flush=True)

    # 3. shadow portfolio --------------------------------------------------------------------
    t0 = time.perf_counter()
    per_loan, portfolio = run_shadow_portfolio(fitted, df, max_loans=args.shadow_max_loans, seed=args.seed)
    timings["shadow"] = time.perf_counter() - t0
    shadow_dir = out / "shadow"
    shadow_dir.mkdir()
    per_loan.to_parquet(shadow_dir / "per_loan.parquet", index=False)
    atomic_write_json(shadow_dir / "summary.json", portfolio.to_dict(), fsync=False)
    print(f"[shadow] {portfolio.n_loans:,} loans in {timings['shadow']:.1f}s; uplift {_fmt_money(portfolio.uplift)}", flush=True)

    # 4. simulation ----------------------------------------------------------------------------
    t0 = time.perf_counter()
    run_dir = out / "simulation"
    last = {"t": time.perf_counter()}

    def progress(info: dict[str, Any]) -> None:
        if info["epoch_index"] % 6 == 5 or info["epoch_index"] == info["n_epochs"] - 1:
            now = time.perf_counter()
            print(f"[simulate] epoch {info['epoch']} ({info['epoch_index'] + 1}/{info['n_epochs']}) "
                  f"breaker={info['breaker_state']} +{now - last['t']:.1f}s", flush=True)
            last["t"] = now

    res = run_simulation(
        df, fitted, sim_cfg, run_dir, dataset_path=ds["path"], dataset_info={k: v for k, v in ds.items() if k != "params"},
        fit_dir=fit_dir, progress_callback=progress,
    )
    timings["simulation"] = time.perf_counter() - t0
    print(f"[simulate] {res.status}: {res.n_decisions:,} decisions in {timings['simulation']:.1f}s", flush=True)
    atomic_write_json(run_dir / "shadow_summary.json", portfolio.to_dict(), fsync=False)

    # 5. kill-and-resume determinism check (G.6) ---------------------------------------------
    resilience = None
    if not args.no_resilience_check:
        t0 = time.perf_counter()
        resilience = resilience_check(df, fitted, sim_cfg, out / "resilience", dataset_path=ds["path"], fit_dir=fit_dir)
        timings["resilience"] = time.perf_counter() - t0
        atomic_write_json(run_dir / "resilience.json", resilience, fsync=False)
        print(f"[resilience] passed={resilience['passed']} in {timings['resilience']:.1f}s", flush=True)

    # 6. report ----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    timings["report"] = 0.0
    report = build_report(
        run_dir, fitted=fitted, shadow_summary=portfolio, timings=timings, resilience=resilience, dataset_summary=ds
    )
    timings["report"] = time.perf_counter() - t0
    report = build_report(
        run_dir, fitted=fitted, shadow_summary=portfolio, timings=timings, resilience=resilience, dataset_summary=ds
    )
    shutil.copyfile(RunDirectory(run_dir).report_path, out / "report.json")
    text = summarise(report, timings, ds, args, out, fitted.metrics, risk_probe(fitted))
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    print(f"\n(total wall time {time.perf_counter() - t_start:.1f}s)")
    with open(out / "timings.json", "w", encoding="utf-8") as fh:
        json.dump({k: round(v, 3) for k, v in timings.items()}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
