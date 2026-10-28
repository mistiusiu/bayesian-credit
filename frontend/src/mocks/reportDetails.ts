/** EvaluationReport.details mock: bandit selection mix (last 6 epochs) and the counterfactual block. */
import type { BanditPolicyDetail, CounterfactualDetail, FitMetrics, PortfolioResult, ReportDetails } from '@/api/types';
import { selectionMix } from './portfolio';
import type { SimView } from './simViews';
import { round } from './stats';

const WINDOW = 6;

function sum(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s;
}

/** Epoch from which per-decision regret stays within ±10% of its final level (null if only at the end). */
function convergence(per: number[]): number | null {
  if (per.length < 6) return null;
  const tail = sum(per.slice(-3)) / 3;
  const tol = 0.1 * Math.abs(tail) + 1;
  for (let e = 0; e < per.length - 3; e++) if (per.slice(e).every((x) => Math.abs(x - tail) <= tol)) return e;
  return null;
}

export function banditDetails(v: SimView): NonNullable<ReportDetails['bandit_performance']> {
  const s = v.series;
  const arms = v.model.grid.arms;
  const K = arms.length;
  const from = Math.max(0, v.done - WINDOW);
  const maxRate = Math.max(...v.model.grid.rates);
  const maxAmount = Math.max(...v.model.grid.amounts);
  const byPolicy: Record<string, BanditPolicyDetail> = {};
  for (const p of v.config.policies) {
    const ps = s.policies.get(p);
    if (!ps) continue;
    const recent = new Float64Array(K);
    const ever = new Float64Array(K);
    for (let e = 0; e < v.done; e++)
      for (let a = 0; a < K; a++) {
        const x = s.n[e] * ps.shares[e][a];
        ever[a] += x;
        if (e >= from) recent[a] += x;
      }
    const total = sum(recent);
    const top = [...recent.keys()].sort((a, b) => recent[b] - recent[a]).slice(0, 3);
    const regret = ps.regret.slice(0, v.done);
    const per = regret.map((r, e) => r / Math.max(s.n[e], 1));
    const cum = sum(regret);
    const conv = convergence(per);
    byPolicy[p] = {
      final_cumulative_regret: round(cum, 2),
      average_regret: round(cum / Math.max(sum(s.n.slice(0, v.done)), 1), 2),
      final_epoch_avg_regret: per.length ? round(per[per.length - 1], 2) : null,
      convergence_epoch: conv === null ? null : s.epochs[conv],
      converged: conv !== null,
      selection_concentration_top3: total > 0 ? round(top.reduce((a, k) => a + recent[k], 0) / total, 6) : null,
      top_arms_recent: top,
      top_arm_labels_recent: top.map((k) => arms[k].label),
      n_distinct_arms: [...ever].filter((x) => x >= 0.5).length,
      selection_mix: total > 0 ? selectionMix(arms.map((arm, k) => ({ arm, weight: recent[k] })).filter((x) => x.weight > 0), maxRate, maxAmount) : null,
    };
  }
  return {
    available: v.done > 0,
    by_policy: byPolicy,
    concentration_window_epochs: s.epochs.slice(from, v.done),
    note: 'regret = oracle − expected under the hindsight model (pseudo-regret); concentration = share of the top-3 arms in the last 6 epochs; selection_mix = the same window\'s lending decisions by term / rate bucket / amount bucket',
  };
}

export function counterfactualDetails(portfolio: Omit<PortfolioResult, 'job'> | null, metrics: FitMetrics): CounterfactualDetail {
  const sm = portfolio?.summary;
  if (!portfolio || !sm) return { available: false, note: 'shadow portfolio not run' };
  const flags = Object.fromEntries(Object.entries(sm.counts).filter(([k]) => k !== 'unchanged').map(([k, c]) => [`n_${k}`, c]));
  // like the server, `details` passes the core's blocks through unconverted: every uplift_pct is in percent
  const pct = <T extends { uplift_pct: number }>(b: T): T => ({ ...b, uplift_pct: round(b.uplift_pct * 100, 4) });
  const d = sm.uplift_decomposition;
  const sweep = sm.by_risk_aversion;
  return {
    available: true,
    n_loans: sm.n_loans,
    n_lendable: sm.n_lendable,
    share_lendable: sm.share_lendable,
    expected_profit_historical: sm.expected_profit_historical,
    expected_profit_shadow: sm.expected_profit_shadow,
    uplift: sm.uplift,
    uplift_pct: round(sm.uplift_pct * 100, 4),
    uplift_risk_adjusted: sm.uplift_risk_adjusted,
    realized_profit_historical: sm.realized_profit_historical,
    n_realized: sm.n_realized,
    counts: { ...sm.counts, ...flags },
    top_loans: portfolio.top_loans.slice(0, 10),
    recommendation_mix: sm.recommendation_mix ?? null,
    uplift_decomposition: d ? { ...d, repricing_only: pct(d.repricing_only), repricing_and_resizing: pct(d.repricing_and_resizing), unrestricted: pct(d.unrestricted) } : null,
    by_risk_aversion: sweep ? Object.fromEntries(Object.entries(sweep).map(([k, v]) => [k, pct(v)])) : null,
    eligibility: sm.eligibility ?? null,
    interest_realization: metrics.interest_realization ?? null,
  };
}
