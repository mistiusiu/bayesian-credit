import type { Fit, FitMetrics as Metrics } from '@/api/types';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';

/** Table 11 predictive dimensions for the selected fit (holdout = issue_d ≥ train_end, matured, resolved). */
export function FitMetrics({ fit, metrics: m }: { fit: Fit; metrics: Metrics }) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 2xl:grid-cols-8">
        <StatTile label="Brier score" value={fmt.num(m.brier, 4)} delta={{ text: `${fmt.numSigned(m.brier - m.brier_baseline, 4)} vs base`, good: m.brier < m.brier_baseline }} />
        <StatTile label="ROC-AUC" value={fmt.num(m.auc, 3)} hint="0.5 = random" />
        <StatTile label="Calibration error" value={fmt.pct(m.ece, 2)} hint="ECE, 10 deciles" />
        <StatTile label="Log loss" value={fmt.num(m.log_loss, 4)} hint="nats" />
        <StatTile label="Recovery MAE" value={fmt.num(m.recovery_mae, 3)} hint="rate, principal basis" />
        <StatTile label="Recovery RMSE" value={fmt.num(m.recovery_rmse, 3)} />
        <StatTile label="Transition log-lik" value={fmt.num(m.transition_log_lik_per_obs, 4)} hint="nats / loan-month" />
        <StatTile label="Transition error" value={fmt.num(m.transition_prediction_error, 4)} hint="mean |ΔP|, C & D rows" />
      </div>
      <p className="num text-2xs text-muted">
        Train n {fmt.int(m.n_train)} ({fmt.int(m.n_defaults_train)} defaults, base rate {fmt.pct(m.base_rate_train)}) · holdout n {fmt.int(m.n_test)} (base rate {fmt.pct(m.base_rate_test)}) · fit time {fmt.duration(m.fit_seconds)} · train_end {fit.config.train_end} · κ group {String(fit.config.hierarchy.kappa_group)} · κ leaf {String(fit.config.hierarchy.kappa_leaf)}
      </p>
    </div>
  );
}
