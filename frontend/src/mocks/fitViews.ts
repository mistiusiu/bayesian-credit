/** Endpoint payloads for a fit: metrics, hierarchy rows, risk estimate, calibration, arms. */
import type {
  Calibration,
  FitArms,
  FitMetrics,
  HierarchyLevel,
  HierarchyRow,
  RecoverySummary,
  RiskEstimate,
} from '@/api/types';
import { groupDims } from '@/lib/domain';
import { type LoanIndex, isDefault } from './loans';
import {
  type FitModel,
  type Node,
  type RecoveryNode,
  groupArmReward,
  leafPosterior,
  leafRecovery,
  nodeHdi,
  riskAdjusted,
} from './riskModel';
import { betaHdi, betaPdfGrid, betaStd, round } from './stats';
import { GRADE_LETTERS } from './vocab';

export interface TestEval {
  predictions: Float64Array;
  outcomes: Uint8Array;
  recovery: { predicted: number; actual: number; grade: number }[];
}

const evalCache = new WeakMap<FitModel, TestEval>();

/** Posterior-mean predictions for the holdout cohort. */
export function testEval(model: FitModel, idx: LoanIndex): TestEval {
  const hit = evalCache.get(model);
  if (hit) return hit;
  const predictions = new Float64Array(model.test.length);
  const outcomes = new Uint8Array(model.test.length);
  const recovery: TestEval['recovery'] = [];
  model.test.forEach((i, k) => {
    const group = model.groupOf(i);
    const arm = model.armOf(idx.amount[i], idx.rate[i], idx.term60[i] ? 60 : 36);
    const post = leafPosterior(model, group, arm);
    predictions[k] = post.alpha / (post.alpha + post.beta);
    const def = isDefault(idx, i);
    outcomes[k] = def ? 1 : 0;
    if (def) recovery.push({ predicted: leafRecovery(model, group, arm).mean, actual: idx.recov[i], grade: idx.grade[i] });
  });
  const ev = { predictions, outcomes, recovery };
  evalCache.set(model, ev);
  return ev;
}

function aucAndRoc(pred: Float64Array, y: Uint8Array) {
  const order = Array.from(pred.keys()).sort((a, b) => pred[b] - pred[a]);
  const P = y.reduce((a, b) => a + b, 0);
  const Nn = y.length - P;
  const roc: { fpr: number; tpr: number; threshold: number }[] = [{ fpr: 0, tpr: 0, threshold: 1 }];
  let tp = 0;
  let fp = 0;
  let auc = 0;
  let prevF = 0;
  let prevT = 0;
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    if (y[i]) tp++;
    else fp++;
    const last = k === order.length - 1 || pred[order[k + 1]] !== pred[i];
    if (last) {
      const f = fp / Nn;
      const t = tp / P;
      auc += ((f - prevF) * (t + prevT)) / 2;
      prevF = f;
      prevT = t;
      roc.push({ fpr: f, tpr: t, threshold: pred[i] });
    }
  }
  const stride = Math.max(1, Math.ceil(roc.length / 160));
  const thinned = roc.filter((_, k) => k % stride === 0 || k === roc.length - 1);
  return { auc, roc: thinned.map((r) => ({ fpr: round(r.fpr, 4), tpr: round(r.tpr, 4), threshold: round(r.threshold, 4) })) };
}

function reliability(pred: Float64Array, y: Uint8Array) {
  const order = Array.from(pred.keys()).sort((a, b) => pred[a] - pred[b]);
  const bins = [];
  let ece = 0;
  for (let b = 0; b < 10; b++) {
    const lo = Math.floor((b * order.length) / 10);
    const hi = Math.floor(((b + 1) * order.length) / 10);
    let sp = 0;
    let sy = 0;
    for (let k = lo; k < hi; k++) {
      sp += pred[order[k]];
      sy += y[order[k]];
    }
    const n = hi - lo;
    const mp = sp / Math.max(n, 1);
    const ob = sy / Math.max(n, 1);
    ece += (n / order.length) * Math.abs(mp - ob);
    bins.push({
      bin: b,
      lo: round(pred[order[lo]], 4),
      hi: round(pred[order[Math.max(lo, hi - 1)]], 4),
      n,
      mean_predicted: round(mp, 4),
      observed_rate: round(ob, 4),
    });
  }
  return { bins, ece };
}

const metricsCache = new WeakMap<FitModel, ReturnType<typeof computeMetrics>>();

export function predictiveMetrics(model: FitModel, idx: LoanIndex) {
  let m = metricsCache.get(model);
  if (!m) {
    m = computeMetrics(model, idx);
    metricsCache.set(model, m);
  }
  return m;
}

function computeMetrics(model: FitModel, idx: LoanIndex) {
  const ev = testEval(model, idx);
  const n = ev.predictions.length;
  let brier = 0;
  let ll = 0;
  let ySum = 0;
  for (let k = 0; k < n; k++) {
    const p = Math.min(Math.max(ev.predictions[k], 1e-6), 1 - 1e-6);
    const y = ev.outcomes[k];
    brier += (p - y) ** 2;
    ll += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    ySum += y;
  }
  const baseTrain = model.global.d / Math.max(model.global.n, 1);
  let brierBase = 0;
  for (let k = 0; k < n; k++) brierBase += (baseTrain - ev.outcomes[k]) ** 2;
  const { auc, roc } = aucAndRoc(ev.predictions, ev.outcomes);
  const { bins, ece } = reliability(ev.predictions, ev.outcomes);
  const recErr = ev.recovery.map((r) => r.actual - r.predicted);
  const mae = recErr.reduce((a, e) => a + Math.abs(e), 0) / Math.max(recErr.length, 1);
  const rmse = Math.sqrt(recErr.reduce((a, e) => a + e * e, 0) / Math.max(recErr.length, 1));
  return { n, brier: brier / n, brierBase: brierBase / n, logLoss: ll / n, auc, roc, bins, ece, baseTest: ySum / n, baseTrain, mae, rmse };
}

export function fitMetrics(model: FitModel, idx: LoanIndex, mctm: { ll: number; n: number; rowErr: number[] }, fitSeconds: number): FitMetrics {
  const m = predictiveMetrics(model, idx);
  const transient = mctm.rowErr.slice(0, 2);
  return {
    brier: round(m.brier, 5),
    brier_baseline: round(m.brierBase, 5),
    auc: round(m.auc, 4),
    ece: round(m.ece, 4),
    log_loss: round(m.logLoss, 4),
    recovery_mae: round(m.mae, 4),
    recovery_rmse: round(m.rmse, 4),
    transition_log_lik: round(mctm.ll, 1),
    transition_log_lik_per_obs: round(mctm.ll / mctm.n, 5),
    transition_prediction_error: round(transient.reduce((a, b) => a + b, 0) / transient.length, 5),
    n_train: model.train.length,
    n_test: m.n,
    n_defaults_train: model.global.d,
    base_rate_train: round(m.baseTrain, 4),
    base_rate_test: round(m.baseTest, 4),
    fit_seconds: fitSeconds,
    ...fitExtras(model),
  };
}

/** policy_effect · interest_realization · guardrails · reference_rates (API §13.1). */
export function fitExtras(model: FitModel) {
  const pe = model.policyEffect;
  const named = (vals: number[]) => Object.fromEntries(pe.features.map((f, j) => [f, round(vals[j], 6)]));
  const r = model.realization;
  const a = model.config.arm_space;
  return {
    policy_effect: {
      status: pe.status,
      model_status: pe.status === 'disabled' ? 'not_fitted' : pe.status,
      installed: pe.status === 'fitted',
      type: pe.status === 'fitted' ? 'PolicyEffectTilt' : null,
      reason: pe.reason,
      beta: pe.status === 'disabled' ? {} : named(pe.beta),
      std_errors: pe.status === 'disabled' ? {} : named(pe.se),
      n_fit: pe.nFit,
      sample: pe.status === 'disabled' ? null : pe.sample,
      converged: pe.converged,
      n_iter: pe.nIter,
      max_abs_logit_shift: pe.maxShift,
    },
    interest_realization: {
      global: round(r.global, 6),
      by_term: Object.fromEntries(Object.entries(r.byTerm).map(([k, v]) => [k, round(v, 6)])),
      by_grade_term: Object.fromEntries(
        Object.entries(r.byGradeTerm)
          .sort(([x], [y]) => x.localeCompare(y))
          .map(([g, t]) => [g, Object.fromEntries(Object.entries(t).map(([k, v]) => [k, round(v, 6)]))]),
      ),
      n_fit: r.nFit,
      sample: r.sample,
      min_count: 30,
      clip: [0.3, 1] as [number, number],
      prepayment_adjustment: model.config.reward.prepayment_adjustment,
    },
    guardrails: { max_payment_to_income: a.max_payment_to_income, rate_corridor: a.rate_corridor, max_amount_to_income: a.max_amount_to_income, max_total_dti: a.max_total_dti },
    reference_rates: {
      n_fit: model.refRates.nFit,
      by_grade: Object.fromEntries(Object.entries(model.refRates.byGrade).map(([g, v]) => [g, round(v, 6)])),
      global: round(model.refRates.global, 6),
    },
  };
}

const levelOf = (r: RecoveryNode): HierarchyLevel => (r.source === 'group_term' ? 'group' : r.source);

const recoverySummary = (r: RecoveryNode): RecoverySummary => ({
  mean: round(r.mean, 4),
  alpha: round(r.alpha, 3),
  beta: round(r.beta, 3),
  n_obs: r.n,
  source: levelOf(r),
  core_source: r.source,
});

function row(level: HierarchyLevel, node: Node, extra: Partial<HierarchyRow>, rec: RecoveryNode | null): HierarchyRow {
  const [lo, hi] = nodeHdi(node);
  const mean = node.alpha / (node.alpha + node.beta);
  return {
    level,
    key: node.key,
    parent: null,
    group: null,
    arm: null,
    dims: {},
    alpha: round(node.alpha, 3),
    beta: round(node.beta, 3),
    n_obs: node.n,
    defaults: node.d,
    mean: round(mean, 5),
    hdi_low: round(lo, 5),
    hdi_high: round(hi, 5),
    prior_mean: node.priorMean === null ? null : round(node.priorMean, 5),
    kappa: node.kappa === null ? null : round(node.kappa, 2),
    shrinkage_weight: node.kappa === null ? 1 : round(node.n / (node.n + node.kappa), 4),
    empirical_rate: node.n > 0 ? round(node.d / node.n, 5) : null,
    recovery: rec ? recoverySummary(rec) : null,
    n_children: 0,
    prior_tilt_logit: level === 'leaf' ? round(node.tilt, 4) : level === 'group' ? 0 : null,
    ...extra,
  };
}

export function hierarchyRows(model: FitModel, level: string | null, group: string | null): HierarchyRow[] {
  const globalRow = row('global', model.global, { n_children: model.groups.size }, model.globalRecovery);
  const groupRows = () =>
    [...model.groups.values()]
      .sort((a, b) => a.key.localeCompare(b.key))
      .map((g) =>
        row('group', g.node, { parent: 'global', group: g.key, dims: groupDims(g.key, model.dims), n_children: g.leaves.size }, g.recovery),
      );
  if (level === 'global') return [globalRow];
  if (level === 'group') return groupRows();
  if (level === 'leaf') {
    const g = group ? model.groups.get(group) : undefined;
    if (!g) return [];
    return [...g.leaves.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([arm, node]) =>
        row(
          'leaf',
          node,
          { parent: g.key, group: g.key, arm, dims: { ...groupDims(g.key, model.dims), arm: model.grid.arms[arm].label } },
          leafRecovery(model, g.key, arm),
        ),
      );
  }
  return [globalRow, ...groupRows()];
}

export function riskEstimate(model: FitModel, fitId: number, group: string, arm: number | null): RiskEstimate {
  const g = model.groups.get(group);
  const node = arm === null ? g?.node ?? model.global : leafPosterior(model, group, arm);
  const rec = arm === null ? g?.recovery ?? model.globalRecovery : leafRecovery(model, group, arm);
  const level: HierarchyLevel = arm === null ? (g ? 'group' : 'global') : node.n > 0 ? 'leaf' : 'group';
  const [lo, hi] = node.hdi ?? betaHdi(node.alpha, node.beta);
  const [rlo, rhi] = betaHdi(rec.alpha, rec.beta);
  const mean = node.alpha / (node.alpha + node.beta);
  const priorMean = node.priorMean ?? mean;
  const kappa = node.kappa ?? model.kappaLeaf;
  const policy = arm === null ? null : model.grid.arms[arm];
  const reward = arm === null ? null : groupArmReward(model, group, arm);
  return {
    fit: fitId,
    group,
    arm,
    policy: policy && !policy.is_null ? { amount: policy.amount, rate: policy.rate, term: policy.term } : null,
    label: policy ? policy.label : 'group posterior',
    p_default: {
      alpha: round(node.alpha, 3),
      beta: round(node.beta, 3),
      mean: round(mean, 5),
      std: round(betaStd(node.alpha, node.beta), 5),
      hdi_low: round(lo, 5),
      hdi_high: round(hi, 5),
      level,
      n_obs: node.n,
      defaults: node.d,
      shrinkage_weight: round(node.n / (node.n + kappa), 4),
    },
    recovery: {
      alpha: round(rec.alpha, 3),
      beta: round(rec.beta, 3),
      mean: round(rec.mean, 4),
      std: round(Math.sqrt(rec.variance), 4),
      hdi_low: round(rlo, 4),
      hdi_high: round(rhi, 4),
      level: levelOf(rec),
      n_obs: rec.n,
      core_source: rec.source,
    },
    expected_loss_fraction: round(mean * (1 - rec.mean), 5),
    reward:
      reward && policy && !policy.is_null
        ? {
            mean: round(reward.mean, 2),
            std: round(reward.std, 2),
            revenue: round(reward.revenue, 2),
            expected_loss: round(reward.expectedLoss, 2),
            funding_cost: round(reward.funding, 2),
            risk_adjusted: round(riskAdjusted(model, reward), 2),
            realization: round(reward.realization, 6),
            gross_interest: round(reward.grossInterest, 2),
          }
        : null,
    pdf: {
      p_default: betaPdfGrid(node.alpha, node.beta),
      p_default_prior: betaPdfGrid(Math.max(kappa * priorMean, 0.05), Math.max(kappa * (1 - priorMean), 0.05)),
      recovery: betaPdfGrid(rec.alpha, rec.beta),
    },
  };
}

export function calibration(model: FitModel, idx: LoanIndex, fitId: number): Calibration {
  const m = predictiveMetrics(model, idx);
  const ev = testEval(model, idx);
  const hist = new Array<number>(20).fill(0);
  const byGrade = GRADE_LETTERS.map(() => ({ s: 0, n: 0 }));
  for (const r of ev.recovery) {
    hist[Math.min(19, Math.floor(r.actual * 20))]++;
    byGrade[r.grade].s += r.actual;
    byGrade[r.grade].n++;
  }
  const g = model.globalRecovery;
  return {
    fit: fitId,
    n_test: m.n,
    base_rate: round(m.baseTest, 4),
    brier: round(m.brier, 5),
    auc: round(m.auc, 4),
    ece: round(m.ece, 4),
    log_loss: round(m.logLoss, 4),
    bins: m.bins,
    roc: m.roc,
    recovery: {
      alpha: round(g.alpha, 3),
      beta: round(g.beta, 3),
      mean: round(g.mean, 4),
      n_defaults: g.n,
      histogram: hist.map((count, k) => ({ lo: k / 20, hi: (k + 1) / 20, count })),
      pdf: betaPdfGrid(g.alpha, g.beta),
      mae: round(m.mae, 4),
      rmse: round(m.rmse, 4),
      by_grade: GRADE_LETTERS.map((grade, k) => ({ grade, mean: round(byGrade[k].s / Math.max(byGrade[k].n, 1), 4), n: byGrade[k].n })),
    },
  };
}

export function fitArms(model: FitModel, fitId: number): FitArms {
  const total = model.historicalCounts.reduce((a, b) => a + b, 0);
  const groups = [...model.groups.values()];
  return {
    fit: fitId,
    amounts: model.grid.amounts,
    rates: model.grid.rates,
    terms: model.grid.terms,
    arms: model.grid.arms.map((a) => {
      let mean = 0;
      let variance = 0;
      let pd = 0;
      let rec = 0;
      for (const g of groups) {
        const r = groupArmReward(model, g.key, a.arm);
        const post = leafPosterior(model, g.key, a.arm);
        mean += g.weight * r.mean;
        variance += g.weight * (r.std * r.std + r.mean * r.mean);
        pd += g.weight * (post.alpha / (post.alpha + post.beta));
        rec += g.weight * leafRecovery(model, g.key, a.arm).mean;
      }
      const std = Math.sqrt(Math.max(0, variance - mean * mean));
      return {
        ...a,
        expected_reward: a.is_null ? 0 : round(mean, 2),
        reward_std: a.is_null ? 0 : round(std, 2),
        p_default: a.is_null ? 0 : round(pd, 5),
        expected_recovery: a.is_null ? 0 : round(rec, 4),
        historical_count: model.historicalCounts[a.arm],
        historical_share: round(model.historicalCounts[a.arm] / Math.max(total, 1), 5),
      };
    }),
  };
}
