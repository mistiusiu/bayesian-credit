/** Rows for the "selection mix, last 6 epochs" table from `EvaluationReport.details.bandit_performance`. */
import type { EvaluationReport, PolicyName } from '@/api/types';
import type { SelectionMixRow } from '@/components/counterfactual/SelectionMixTable';
import { policyLabel } from './domain';
import { policyColor } from './palette';

export function selectionMixRows(report: EvaluationReport, order: readonly PolicyName[]): SelectionMixRow[] {
  const bp = report.details?.bandit_performance;
  if (!bp?.available) return [];
  const keys = Object.keys(bp.by_policy);
  const sorted = [...order.filter((p) => keys.includes(p)), ...keys.filter((k) => !order.includes(k))];
  return sorted.map((p) => {
    const d = bp.by_policy[p];
    return {
      key: p,
      label: policyLabel(p),
      color: policyColor(p),
      mix: d.selection_mix,
      top3: d.selection_concentration_top3,
      topLabels: d.top_arm_labels_recent,
      distinct: d.n_distinct_arms,
    };
  });
}

/** "Jul 2018 – Dec 2018" for the concentration window. */
export function windowLabel(report: EvaluationReport, fmtEpoch: (e: string) => string): string | null {
  const w = report.details?.bandit_performance?.concentration_window_epochs ?? [];
  return w.length ? `${fmtEpoch(w[0])} – ${fmtEpoch(w[w.length - 1])}` : null;
}
