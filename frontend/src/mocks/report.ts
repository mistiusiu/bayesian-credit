/** EvaluationReport mock: the nine Table 11 dimensions + runtime / feasibility. */
import type {
  EvaluationReport,
  ReportDetails,
  FitConfig,
  FitMetrics,
  PolicyHeadline,
  PolicyName,
  ReportDimension,
  ReportMetric,
  RuntimeStats,
  SentinelTimeline,
  SimulationConfig,
  Timeseries,
} from '@/api/types';
import { leadPolicy, policyBase } from '@/lib/domain';
import { round } from './stats';

const metric = (m: Partial<ReportMetric> & Pick<ReportMetric, 'key' | 'label' | 'value' | 'unit'>): ReportMetric => ({
  better: null,
  baseline: null,
  baseline_label: null,
  by_policy: null,
  note: null,
  ...m,
});

const byPolicy = (h: PolicyHeadline[], f: (p: PolicyHeadline) => number) => {
  const out: Partial<Record<PolicyName, number>> = {};
  for (const p of h) out[p.policy] = round(f(p), 4);
  return out;
};

export function buildReport(args: {
  simulation: number;
  fit: number;
  dataset: number;
  final: boolean;
  metrics: FitMetrics;
  headline: PolicyHeadline[];
  timeseries: Timeseries;
  sentinel: SentinelTimeline;
  runtime: RuntimeStats;
  fitConfig: FitConfig;
  simConfig: SimulationConfig;
  generatedAt: string;
  topShare: Partial<Record<PolicyName, number>>;
  details: ReportDetails;
}): EvaluationReport {
  const { metrics: m, headline: h, sentinel: s } = args;
  const leadName = leadPolicy(h.map((p) => p.policy));
  const lead = h.find((p) => p.policy === leadName) ?? h[0];
  const hist = h.find((p) => policyBase(p.policy) === 'historical') ?? null;
  const tsRows = args.timeseries.rows.filter((r) => r.policy === lead?.policy);
  const finalRolling = tsRows[tsRows.length - 1]?.rolling_reward ?? 0;
  const convergeIdx = tsRows.findIndex((r) => r.rolling_reward >= 0.95 * finalRolling);
  const trips = s.transitions.filter((t) => t.to_state === 'open');
  const closes = s.transitions.filter((t) => t.to_state === 'closed');
  const epochIdx = (e: string) => s.aggregate.findIndex((a) => a.epoch === e);
  const timeToClose = trips.length && closes.length ? epochIdx(closes[0].epoch) - epochIdx(trips[0].epoch) : null;
  const logLossBase = -(m.base_rate_test * Math.log(m.base_rate_train) + (1 - m.base_rate_test) * Math.log(1 - m.base_rate_train));

  const dims: ReportDimension[] = [
    {
      key: 'default_prediction',
      label: 'Default prediction',
      axis: 'predictive',
      metrics: [
        metric({ key: 'brier', label: 'Brier score', value: m.brier, unit: 'ratio', better: 'lower', baseline: m.brier_baseline, baseline_label: 'training base rate' }),
        metric({ key: 'auc', label: 'ROC-AUC', value: m.auc, unit: 'ratio', better: 'higher', baseline: 0.5, baseline_label: 'random ranking' }),
        metric({ key: 'ece', label: 'Calibration error (ECE)', value: m.ece, unit: 'fraction', better: 'lower', note: '10 prediction-decile bins' }),
        metric({ key: 'log_loss', label: 'Log loss', value: m.log_loss, unit: 'nats', better: 'lower', baseline: round(logLossBase, 4), baseline_label: 'training base rate' }),
      ],
    },
    {
      key: 'recovery_prediction',
      label: 'Recovery prediction',
      axis: 'predictive',
      metrics: [
        metric({ key: 'recovery_mae', label: 'MAE', value: m.recovery_mae, unit: 'ratio', better: 'lower', note: 'recovery rate on the principal basis' }),
        metric({ key: 'recovery_rmse', label: 'RMSE', value: m.recovery_rmse, unit: 'ratio', better: 'lower' }),
      ],
    },
    {
      key: 'credit_state_modeling',
      label: 'Credit-state modeling',
      axis: 'predictive',
      metrics: [
        metric({ key: 'transition_log_lik', label: 'Transition log-likelihood per loan-month', value: m.transition_log_lik_per_obs, unit: 'nats', better: 'higher' }),
        metric({ key: 'transition_prediction_error', label: 'Transition prediction error', value: m.transition_prediction_error, unit: 'ratio', better: 'lower', note: 'mean absolute row error, transient states' }),
      ],
    },
    {
      key: 'policy_performance',
      label: 'Policy performance',
      axis: 'decision',
      metrics: [
        metric({ key: 'cum_reward', label: 'Cumulative risk-adjusted reward', value: lead?.cum_reward ?? null, unit: 'usd', better: 'higher', baseline: hist?.cum_reward ?? null, baseline_label: 'historical policy', by_policy: byPolicy(h, (p) => p.cum_reward) }),
        metric({ key: 'mean_reward', label: 'Risk-adjusted profitability', value: lead?.mean_reward ?? null, unit: 'usd_per_decision', better: 'higher', baseline: hist?.mean_reward ?? null, baseline_label: 'historical policy', by_policy: byPolicy(h, (p) => p.mean_reward) }),
        metric({ key: 'cum_realized', label: 'Cumulative realized reward', value: lead?.cum_realized ?? null, unit: 'usd', better: 'higher', baseline: hist?.cum_realized ?? null, baseline_label: 'historical policy', by_policy: byPolicy(h, (p) => p.cum_realized) }),
      ],
    },
    {
      key: 'bandit_performance',
      label: 'Bandit performance',
      axis: 'decision',
      metrics: [
        metric({ key: 'cum_regret', label: 'Cumulative regret', value: lead?.cum_regret ?? null, unit: 'usd', better: 'lower', baseline: hist?.cum_regret ?? null, baseline_label: 'historical policy', by_policy: byPolicy(h, (p) => p.cum_regret) }),
        metric({ key: 'convergence_epochs', label: 'Reward convergence (epochs to 95% of final rolling reward)', value: convergeIdx >= 0 ? convergeIdx + 1 : null, unit: 'count', better: 'lower' }),
        metric({ key: 'top5_share', label: 'Policy selection frequency (top-5 arm share, final epoch)', value: lead ? (args.topShare[lead.policy] ?? null) : null, unit: 'fraction', by_policy: args.topShare }),
        metric({ key: 'decline_rate', label: 'Decline rate', value: lead?.decline_rate ?? null, unit: 'fraction', by_policy: byPolicy(h, (p) => p.decline_rate) }),
      ],
    },
    {
      key: 'counterfactual_evaluation',
      label: 'Counterfactual evaluation',
      axis: 'decision',
      metrics: [
        metric({ key: 'uplift_vs_historical', label: 'Uplift vs historical policy', value: lead && hist ? round(lead.cum_reward - hist.cum_reward, 2) : null, unit: 'usd', better: 'higher' }),
        metric({ key: 'uplift_per_decision', label: 'Uplift per decision', value: lead && hist ? round(lead.mean_reward - hist.mean_reward, 2) : null, unit: 'usd_per_decision', better: 'higher' }),
        metric({ key: 'uplift_pct', label: 'Relative uplift', value: lead && hist ? round((lead.cum_reward - hist.cum_reward) / Math.abs(hist.cum_reward || 1), 4) : null, unit: 'fraction', better: 'higher' }),
      ],
    },
    {
      key: 'distributional_monitoring',
      label: 'Distributional monitoring',
      axis: 'adaptability',
      metrics: [
        metric({ key: 'psi_max', label: 'Maximum aggregate PSI', value: s.aggregate.length ? Math.max(...s.aggregate.map((a) => a.aggregate)) : null, unit: 'ratio', better: 'lower', baseline: s.thresholds.breaker, baseline_label: 'breaker threshold' }),
        metric({ key: 'watch_epochs', label: 'Epochs at watch level', value: s.shift_counts.watch, unit: 'count' }),
        metric({ key: 'breaker_epochs', label: 'Epochs at breaker level', value: s.shift_counts.breaker, unit: 'count' }),
        metric({ key: 'breaker_trips', label: 'Detected shifts (breaker trips)', value: trips.length, unit: 'count' }),
        metric({ key: 'time_to_close', label: 'Trip to close (epochs)', value: timeToClose, unit: 'count', better: 'lower' }),
      ],
    },
    {
      key: 'computational_feasibility',
      label: 'Computational feasibility',
      axis: 'feasibility',
      metrics: [
        metric({ key: 'cpu_seconds', label: 'CPU time', value: args.runtime.cpu_seconds, unit: 'seconds', better: 'lower' }),
        metric({ key: 'cpu_utilization', label: 'CPU utilisation (cores)', value: args.runtime.cpu_utilization, unit: 'ratio' }),
        metric({ key: 'peak_memory', label: 'Peak memory', value: args.runtime.peak_memory_mb, unit: 'megabytes', better: 'lower' }),
        metric({ key: 'wall_seconds', label: 'Execution time', value: args.runtime.wall_seconds, unit: 'seconds', better: 'lower' }),
      ],
    },
    {
      key: 'infrastructure_resilience',
      label: 'Infrastructure resilience',
      axis: 'feasibility',
      metrics: [
        metric({ key: 'recovery_seconds', label: 'Recovery time', value: args.runtime.recovery_seconds, unit: 'seconds', better: 'lower' }),
        metric({ key: 'failed_workloads', label: 'Failed workloads', value: args.runtime.failed_workloads, unit: 'count', better: 'lower' }),
        metric({ key: 'restarts', label: 'Successful state recoveries', value: args.runtime.restarts, unit: 'count', note: 'resume-from-checkpoint restarts with deterministic replay' }),
      ],
    },
  ];
  return {
    simulation: args.simulation,
    fit: args.fit,
    dataset: args.dataset,
    generated_at: args.generatedAt,
    status: args.final ? 'final' : 'partial',
    dimensions: dims,
    runtime: args.runtime,
    policy_summary: h,
    config: { fit: args.fitConfig, simulation: args.simConfig },
    details: args.details,
  };
}
