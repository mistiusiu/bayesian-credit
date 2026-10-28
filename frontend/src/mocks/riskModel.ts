/**
 * Mock of the Two-Stage Beta-Beta model (SPEC §2.2–2.4, docs/MATH.md §2.6–2.9) fitted on the
 * loan index: hierarchical Beta-Bernoulli default (global → group → leaf) whose leaf prior mean
 * is tilted by a fitted within-group policy effect; Beta method-of-moments recovery with a
 * leaf → (group, term) → group → global fallback; realised-interest factors ρ(grade, term);
 * reference rates for the rate corridor; and the four lending guardrails.
 */
import type { FitConfig, SegmentationDimension } from '@/api/types';
import { buildArmGrid, type ArmGrid } from '@/lib/domain';
import { epochDiff } from '@/lib/epoch';
import { installment, interestThrough, rewardMoments, type RewardMoments } from './finance';
import { type LoanIndex, isDefault, isResolved, termOf } from './loans';
import { ALL_FEATURES, fitOffsetLogit, type PolicyEffectFit } from './policyEffect';
import { betaHdi, betaHdiApprox, clamp } from './stats';
import { GRADE_LETTERS, HOME_KEYS, SNAPSHOT, ST, SUB_GRADES, VERIF_KEYS, purposeGroupOf } from './vocab';

export interface Node {
  key: string;
  alpha: number;
  beta: number;
  n: number;
  d: number;
  priorMean: number | null;
  kappa: number | null;
  hdi: [number, number] | null;
  /** Logit tilt of the prior mean (leaves only). */
  tilt: number;
}

export type RecoverySource = 'leaf' | 'group_term' | 'group' | 'global';

export interface RecoveryNode {
  alpha: number;
  beta: number;
  mean: number;
  variance: number;
  n: number;
  source: RecoverySource;
}

export interface GroupInfo {
  key: string;
  node: Node;
  grade: string | null;
  /** Feature means over all resolved training rows of the group (tilt centring). */
  xbar: number[];
  medianIncome: number;
  meanRate: number;
  weight: number;
  recovery: RecoveryNode;
  termRecovery: Map<number, RecoveryNode>;
  leaves: Map<number, Node>;
  leafRecovery: Map<number, RecoveryNode>;
}

export interface RefRates {
  bySub: Record<string, number>;
  byGrade: Record<string, number>;
  global: number;
  nFit: number;
}

export interface RealizationTable {
  enabled: boolean;
  global: number;
  byTerm: Record<string, number>;
  byGradeTerm: Record<string, Record<string, number>>;
  nFit: number;
  sample: string;
}

export interface FitModel {
  config: FitConfig;
  grid: ArmGrid;
  dims: SegmentationDimension[];
  global: Node;
  globalRecovery: RecoveryNode;
  groups: Map<string, GroupInfo>;
  kappaGroup: number;
  kappaLeaf: number;
  train: number[];
  test: number[];
  groupOf: (i: number) => string;
  armOf: (amount: number, rate: number, term: number) => number;
  historicalCounts: number[];
  policyEffect: PolicyEffectFit;
  refRates: RefRates;
  realization: RealizationTable;
  globalXbar: number[];
  globalMedianIncome: number;
}

const band = (v: number, edges: number[], labels: string[]) => {
  for (let k = 0; k < edges.length; k++) if (v < edges[k]) return labels[k];
  return labels[labels.length - 1];
};

export function dimensionValue(idx: LoanIndex, i: number, dim: SegmentationDimension): string {
  switch (dim) {
    case 'grade':
      return GRADE_LETTERS[idx.grade[i]];
    case 'sub_grade':
      return SUB_GRADES[idx.sub[i]];
    case 'purpose_group':
      return purposeGroupOf(idx.purpose[i]);
    case 'home_ownership':
      return HOME_KEYS[idx.home[i]];
    case 'fico_band':
      return band(idx.fico[i], [660, 700, 740, 780], ['<660', '660-699', '700-739', '740-779', '780+']);
    case 'dti_band':
      return band(idx.dti[i], [10, 20, 30], ['<10', '10-20', '20-30', '30+']);
    case 'income_band':
      return band(idx.income[i], [40000, 75000, 120000], ['<40k', '40-75k', '75-120k', '120k+']);
    case 'emp_band':
      return idx.emp[i] === 255 ? 'unknown' : band(idx.emp[i], [2, 6, 10], ['<2', '2-5', '6-9', '10+']);
    case 'verification_status':
      return VERIF_KEYS[idx.verif[i]];
    case 'application_type':
      return idx.joint[i] ? 'Joint App' : 'Individual';
  }
}

function makeArmLookup(grid: ArmGrid) {
  const offset = grid.arms[0]?.is_null ? 1 : 0;
  const nA = grid.amounts.length;
  const nR = grid.rates.length;
  const nearestIdx = (arr: number[], v: number) => {
    let best = 0;
    for (let k = 1; k < arr.length; k++) if (Math.abs(arr[k] - v) < Math.abs(arr[best] - v)) best = k;
    return best;
  };
  return (amount: number, rate: number, term: number) => {
    let t = grid.terms.indexOf(term);
    if (t < 0) t = nearestIdx(grid.terms, term);
    return offset + t * nA * nR + nearestIdx(grid.amounts, amount) * nR + nearestIdx(grid.rates, rate);
  };
}

function momKappa(rates: { mean: number; n: number }[], m: number): number {
  const usable = rates.filter((r) => r.n >= 30);
  if (usable.length < 3) return 20;
  const w = usable.reduce((a, r) => a + r.n, 0);
  const v = usable.reduce((a, r) => a + r.n * (r.mean - m) ** 2, 0) / w;
  if (v <= 0) return 1000;
  return clamp((m * (1 - m)) / v - 1, 1, 1000);
}

function recoveryMom(values: number[], minObs: number, fallback: RecoveryNode | null, source: RecoverySource): RecoveryNode {
  if (values.length < minObs && fallback) return fallback;
  const n = values.length;
  const mu = clamp(values.reduce((a, b) => a + b, 0) / Math.max(n, 1), 1e-4, 1 - 1e-4);
  const variance = values.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(n - 1, 1);
  let conc = variance < 1e-8 ? 1000 : (mu * (1 - mu)) / variance - 1;
  if (!(conc > 0) && fallback) return fallback;
  conc = clamp(conc, 0.5, 1000);
  return { alpha: mu * conc, beta: (1 - mu) * conc, mean: mu, variance: (mu * (1 - mu)) / (conc + 1), n, source };
}

const logit = (p: number) => Math.log(p / (1 - p));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const median = (a: number[]) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
};

/** Policy features of training row i: [int_rate, 1[60m], PTI (capped), log amount]. */
function rowFeatures(idx: LoanIndex, i: number, ptiCap: number): number[] {
  const T = termOf(idx, i);
  const inst = installment(idx.amount[i], idx.rate[i], T);
  return [idx.rate[i], T === 60 ? 1 : 0, clamp((12 * inst) / Math.max(idx.income[i], 1), 0, ptiCap), Math.log(idx.amount[i])];
}

export function buildFitModel(idx: LoanIndex, config: FitConfig): FitModel {
  const grid = buildArmGrid(config.arm_space);
  const dims = config.segmentation.dimensions;
  const armOf = makeArmLookup(grid);
  const groupOf = (i: number) => dims.map((d) => dimensionValue(idx, i, d)).join('|');
  const cfgPE = config.hierarchy.policy_effect_cfg;
  const train: number[] = [];
  const test: number[] = [];
  const subRates = new Map<number, { s: number; n: number }>();
  for (let i = 0; i < idx.n; i++) {
    const epoch = idx.epochs[idx.issue[i]];
    if (epoch < config.train_end) {
      const r = subRates.get(idx.sub[i]) ?? { s: 0, n: 0 };
      r.s += idx.rate[i];
      r.n++;
      subRates.set(idx.sub[i], r);
    }
    if (!isResolved(idx, i)) continue;
    if (epoch < config.train_end) train.push(i);
    else if (epochDiff(epoch, SNAPSHOT) >= termOf(idx, i)) test.push(i);
  }

  // reference rates: sub-grade → grade → global mean training int_rate
  const bySub: Record<string, number> = {};
  const gradeAcc = GRADE_LETTERS.map(() => ({ s: 0, n: 0 }));
  let allS = 0;
  let allN = 0;
  for (const [sub, r] of subRates) {
    bySub[SUB_GRADES[sub]] = r.s / r.n;
    gradeAcc[Math.floor(sub / 5)].s += r.s;
    gradeAcc[Math.floor(sub / 5)].n += r.n;
    allS += r.s;
    allN += r.n;
  }
  const byGrade: Record<string, number> = {};
  GRADE_LETTERS.forEach((g, k) => {
    if (gradeAcc[k].n) byGrade[g] = gradeAcc[k].s / gradeAcc[k].n;
  });
  const refRates: RefRates = { bySub, byGrade, global: allS / Math.max(allN, 1), nFit: allN };

  // realised share of scheduled interest, fully-paid training loans (elapsed if ≥ 1000)
  const paid = train.filter((i) => idx.status[i] === ST.FULLY_PAID);
  const paidElapsed = paid.filter((i) => epochDiff(idx.epochs[idx.issue[i]], SNAPSHOT) >= termOf(idx, i));
  const rhoRows = paidElapsed.length >= 1000 ? paidElapsed : paid;
  const cell = new Map<string, { s: number; n: number }>();
  const add = (key: string, v: number) => {
    const c = cell.get(key) ?? { s: 0, n: 0 };
    c.s += v;
    c.n++;
    cell.set(key, c);
  };
  for (const i of rhoRows) {
    const T = termOf(idx, i);
    const total = interestThrough(idx.amount[i], idx.rate[i], T, T);
    const rho = clamp(interestThrough(idx.amount[i], idx.rate[i], T, idx.payMonth[i]) / Math.max(total, 1e-9), 0, 1);
    add(`${GRADE_LETTERS[idx.grade[i]]}|${T}`, rho);
    add(`*|${T}`, rho);
    add('*|*', rho);
  }
  const clipRho = (v: number) => clamp(v, 0.3, 1);
  const byGradeTerm: Record<string, Record<string, number>> = {};
  const byTerm: Record<string, number> = {};
  for (const [key, c] of cell) {
    const [g, t] = key.split('|');
    if (g === '*' && t !== '*') byTerm[t] = clipRho(c.s / c.n);
    else if (g !== '*' && c.n >= 30) (byGradeTerm[g] ??= {})[t] = clipRho(c.s / c.n);
  }
  const rhoAll = cell.get('*|*');
  const realization: RealizationTable = {
    enabled: config.reward.prepayment_adjustment === 'empirical',
    global: rhoAll ? clipRho(rhoAll.s / rhoAll.n) : 1,
    byTerm,
    byGradeTerm,
    nFit: rhoRows.length,
    sample: rhoRows === paidElapsed ? 'fully_paid_elapsed' : 'fully_paid',
  };

  const { alpha0, beta0 } = config.hierarchy;
  let N = 0;
  let D = 0;
  const raw = new Map<
    string,
    { n: number; d: number; rate: number; inc: number[]; x: number[]; rec: number[]; recT: Map<number, number[]>; leaves: Map<number, { n: number; d: number; rec: number[] }> }
  >();
  const historicalCounts = new Array<number>(grid.arms.length).fill(0);
  const allRec: number[] = [];
  const globalX = [0, 0, 0, 0];
  const allInc: number[] = [];
  for (const i of train) {
    const key = groupOf(i);
    const T = termOf(idx, i);
    const arm = armOf(idx.amount[i], idx.rate[i], T);
    const def = isDefault(idx, i) ? 1 : 0;
    const x = rowFeatures(idx, i, cfgPE.pti_cap);
    N++;
    D += def;
    historicalCounts[arm]++;
    let g = raw.get(key);
    if (!g) {
      g = { n: 0, d: 0, rate: 0, inc: [], x: [0, 0, 0, 0], rec: [], recT: new Map(), leaves: new Map() };
      raw.set(key, g);
    }
    g.n++;
    g.d += def;
    g.rate += idx.rate[i];
    g.inc.push(idx.income[i]);
    allInc.push(idx.income[i]);
    x.forEach((v, k) => {
      g.x[k] += v;
      globalX[k] += v;
    });
    let leaf = g.leaves.get(arm);
    if (!leaf) {
      leaf = { n: 0, d: 0, rec: [] };
      g.leaves.set(arm, leaf);
    }
    leaf.n++;
    leaf.d += def;
    if (def) {
      g.rec.push(idx.recov[i]);
      leaf.rec.push(idx.recov[i]);
      allRec.push(idx.recov[i]);
      const list = g.recT.get(T) ?? [];
      list.push(idx.recov[i]);
      g.recT.set(T, list);
    }
  }
  const globalXbar = globalX.map((v) => v / Math.max(N, 1));
  const globalMedianIncome = median(allInc);

  // policy effect: offset logit on the elapsed (else all) resolved training rows
  const features = cfgPE.features.filter((f) => ALL_FEATURES.includes(f));
  const featIdx = features.map((f) => ALL_FEATURES.indexOf(f));
  let policyEffect: PolicyEffectFit = {
    status: 'disabled',
    reason: null,
    features,
    beta: features.map(() => 0),
    se: features.map(() => 0),
    nFit: 0,
    sample: 'resolved',
    converged: false,
    nIter: 0,
    maxShift: cfgPE.max_abs_logit_shift,
  };
  if (config.hierarchy.policy_effect && features.length) {
    const elapsed = train.filter((i) => epochDiff(idx.epochs[idx.issue[i]], SNAPSHOT) >= termOf(idx, i));
    const rows = elapsed.length >= cfgPE.min_elapsed_rows ? elapsed : train;
    const sample = rows === elapsed ? 'elapsed' : 'resolved';
    if (rows.length < cfgPE.min_rows) {
      policyEffect = { ...policyEffect, status: 'fallback', sample, nFit: rows.length, reason: `insufficient data (${rows.length} usable rows < min_rows ${cfgPE.min_rows})` };
    } else {
      const k = features.length;
      const stat = new Map<string, { n: number; d: number; x: number[] }>();
      const feats = rows.map((i) => rowFeatures(idx, i, cfgPE.pti_cap));
      rows.forEach((i, r) => {
        const key = groupOf(i);
        const s = stat.get(key) ?? { n: 0, d: 0, x: new Array<number>(4).fill(0) };
        s.n++;
        s.d += isDefault(idx, i) ? 1 : 0;
        feats[r].forEach((v, j) => (s.x[j] += v));
        stat.set(key, s);
      });
      const xc = new Float64Array(rows.length * k);
      const off = new Float64Array(rows.length);
      const y = new Uint8Array(rows.length);
      rows.forEach((i, r) => {
        const s = stat.get(groupOf(i))!;
        off[r] = logit((s.d + 1) / (s.n + 2));
        y[r] = isDefault(idx, i) ? 1 : 0;
        featIdx.forEach((f, j) => (xc[r * k + j] = feats[r][f] - s.x[f] / s.n));
      });
      const fit = fitOffsetLogit(xc, off, y, k, cfgPE);
      policyEffect = { ...policyEffect, status: fit.converged ? 'fitted' : 'fallback', beta: fit.converged ? fit.beta : features.map(() => 0), se: fit.se, nFit: rows.length, sample, converged: fit.converged, nIter: fit.nIter, reason: fit.converged ? null : 'Newton–Raphson did not converge' };
    }
  }

  const gA = alpha0 + D;
  const gB = beta0 + N - D;
  const m0 = gA / (gA + gB);
  const global: Node = { key: 'global', alpha: gA, beta: gB, n: N, d: D, priorMean: null, kappa: null, hdi: betaHdi(gA, gB), tilt: 0 };
  const groupRates = [...raw.values()].map((g) => ({ mean: g.d / g.n, n: g.n }));
  const kappaGroup = config.hierarchy.kappa_group === 'auto' ? momKappa(groupRates, m0) : config.hierarchy.kappa_group;
  const globalRecovery = recoveryMom(allRec, 1, null, 'global');
  const minObs = config.recovery.min_obs;
  const gi = dims.indexOf('grade');
  const gradeOfKey = (key: string) => (gi >= 0 ? key.split('|')[gi] : null);

  const groups = new Map<string, GroupInfo>();
  const leafRatesAll: { mean: number; n: number }[] = [];
  for (const [key, g] of raw) {
    const a = kappaGroup * m0 + g.d;
    const b = kappaGroup * (1 - m0) + g.n - g.d;
    const recovery = recoveryMom(g.rec, minObs, globalRecovery, 'group');
    const termRecovery = new Map<number, RecoveryNode>();
    if (config.recovery.term_level) for (const [T, vals] of g.recT) if (vals.length >= minObs) termRecovery.set(T, recoveryMom(vals, minObs, null, 'group_term'));
    groups.set(key, {
      key,
      node: { key, alpha: a, beta: b, n: g.n, d: g.d, priorMean: m0, kappa: kappaGroup, hdi: betaHdi(a, b), tilt: 0 },
      grade: gradeOfKey(key),
      xbar: g.x.map((v) => v / g.n),
      medianIncome: median(g.inc),
      meanRate: g.rate / g.n,
      weight: g.n / N,
      recovery,
      termRecovery,
      leaves: new Map(),
      leafRecovery: new Map(),
    });
    for (const leaf of g.leaves.values()) leafRatesAll.push({ mean: leaf.d / leaf.n, n: leaf.n });
  }
  // groups never seen in training still exist (posterior = prior): cartesian product of observed values
  const values = dims.map((_, k) => [...new Set([...raw.keys()].map((key) => key.split('|')[k]))].sort());
  let combos: string[][] = [[]];
  for (const vals of values) combos = combos.flatMap((c) => vals.map((v) => [...c, v]));
  if (combos.length <= 2000) {
    for (const parts of combos) {
      const key = parts.join('|');
      if (groups.has(key)) continue;
      const a = kappaGroup * m0;
      const b = kappaGroup * (1 - m0);
      groups.set(key, {
        key,
        node: { key, alpha: a, beta: b, n: 0, d: 0, priorMean: m0, kappa: kappaGroup, hdi: betaHdi(a, b), tilt: 0 },
        grade: gradeOfKey(key),
        xbar: globalXbar,
        medianIncome: globalMedianIncome,
        meanRate: refRates.global,
        weight: 0,
        recovery: globalRecovery,
        termRecovery: new Map(),
        leaves: new Map(),
        leafRecovery: new Map(),
      });
    }
  }
  const kappaLeaf = config.hierarchy.kappa_leaf === 'auto' ? momKappa(leafRatesAll, m0) : config.hierarchy.kappa_leaf;
  const model: FitModel = {
    config, grid, dims, global, globalRecovery, groups, kappaGroup, kappaLeaf, train, test, groupOf, armOf, historicalCounts,
    policyEffect, refRates, realization, globalXbar, globalMedianIncome,
  };
  for (const [key, g] of raw) {
    const info = groups.get(key)!;
    for (const [arm, leaf] of g.leaves) {
      const node = leafPrior(model, info, arm);
      info.leaves.set(arm, { ...node, alpha: node.alpha + leaf.d, beta: node.beta + leaf.n - leaf.d, n: leaf.n, d: leaf.d });
      const term = grid.arms[arm].term;
      const parent = info.termRecovery.get(term) ?? info.recovery;
      info.leafRecovery.set(arm, recoveryMom(leaf.rec, minObs, parent, 'leaf'));
    }
  }
  return model;
}

/** t(g, a) = clip(β·(x_{g,a} − x̄_g), ±max_abs_logit_shift); 0 for the null arm or without a fitted effect. */
export function priorTilt(model: FitModel, g: GroupInfo | null, arm: number): number {
  const pe = model.policyEffect;
  const a = model.grid.arms[arm];
  if (!a || a.is_null || pe.status !== 'fitted') return 0;
  const income = g?.medianIncome ?? model.globalMedianIncome;
  const xbar = g?.xbar ?? model.globalXbar;
  const cap = model.config.hierarchy.policy_effect_cfg.pti_cap;
  const x = [a.rate, a.term === 60 ? 1 : 0, clamp((12 * a.installment) / Math.max(income, 1), 0, cap), Math.log(a.amount)];
  let t = 0;
  pe.features.forEach((f, j) => {
    const k = ALL_FEATURES.indexOf(f);
    t += pe.beta[j] * (x[k] - xbar[k]);
  });
  return clamp(t, -pe.maxShift, pe.maxShift);
}

function leafPrior(model: FitModel, g: GroupInfo, arm: number): Node {
  const anchor = g.node.alpha / (g.node.alpha + g.node.beta);
  const tilt = priorTilt(model, g, arm);
  const prior = clamp(sigmoid(logit(anchor) + tilt), 0.002, 0.95);
  return { key: `${g.key}#${arm}`, alpha: model.kappaLeaf * prior, beta: model.kappaLeaf * (1 - prior), n: 0, d: 0, priorMean: prior, kappa: model.kappaLeaf, hdi: null, tilt };
}

/** Posterior Beta(α, β) of leaf (group, arm); unobserved leaves return their (tilted) prior. */
export function leafPosterior(model: FitModel, group: string, arm: number): Node {
  const g = model.groups.get(group);
  if (!g) {
    const m = model.global.alpha / (model.global.alpha + model.global.beta);
    const tilt = priorTilt(model, null, arm);
    const prior = clamp(sigmoid(logit(m) + tilt), 0.002, 0.95);
    return { key: `${group}#${arm}`, alpha: model.kappaGroup * prior, beta: model.kappaGroup * (1 - prior), n: 0, d: 0, priorMean: prior, kappa: model.kappaGroup, hdi: null, tilt };
  }
  return g.leaves.get(arm) ?? leafPrior(model, g, arm);
}

/** Recovery chain leaf → (group, term) → group → global. */
export function leafRecovery(model: FitModel, group: string, arm: number): RecoveryNode {
  const g = model.groups.get(group);
  if (!g) return model.globalRecovery;
  const hit = g.leafRecovery.get(arm);
  if (hit) return hit;
  const term = model.grid.arms[arm]?.term ?? 0;
  return g.termRecovery.get(term) ?? g.recovery;
}

/** ρ(grade(g), term(a)) with fallback term → global → 1 (1 when the adjustment is off or for the null arm). */
export function realizationOf(model: FitModel, group: string, arm: number): number {
  const r = model.realization;
  const a = model.grid.arms[arm];
  if (!r.enabled || !a || a.is_null) return 1;
  const grade = model.groups.get(group)?.grade ?? null;
  const t = String(a.term);
  return (grade ? r.byGradeTerm[grade]?.[t] : undefined) ?? r.byTerm[t] ?? r.global;
}

export function nodeHdi(node: Node): [number, number] {
  return node.hdi ?? betaHdiApprox(node.alpha, node.beta);
}

const rewardCache = new WeakMap<FitModel, Map<string, RewardMoments>>();

/** Closed-form reward moments for (group, arm) with the realisation factor; cached per model. */
export function groupArmReward(model: FitModel, group: string, arm: number): RewardMoments {
  let cache = rewardCache.get(model);
  if (!cache) {
    cache = new Map();
    rewardCache.set(model, cache);
  }
  const key = `${group}#${arm}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const p = model.grid.arms[arm];
  const post = leafPosterior(model, group, arm);
  const rec = leafRecovery(model, group, arm);
  const n = post.alpha + post.beta;
  const pm = post.alpha / n;
  const pVar = (post.alpha * post.beta) / (n * n * (n + 1));
  const r = rewardMoments(p.amount, p.rate, p.term, pm, rec.mean, rec.variance, model.config.reward.funding_rate, realizationOf(model, group, arm), pVar);
  cache.set(key, r);
  return r;
}

export function riskAdjusted(model: FitModel, r: RewardMoments, lambda = model.config.reward.risk_aversion): number {
  return r.mean - lambda * r.std;
}

/** Reference rate of a borrower: sub-grade → grade → global training mean. */
export function referenceRate(model: FitModel, subGrade: string | null, grade: string | null): number | null {
  return (subGrade ? model.refRates.bySub[subGrade] : undefined) ?? (grade ? model.refRates.byGrade[grade] : undefined) ?? model.refRates.global;
}

export interface Eligibility {
  mask: boolean[];
  refRate: number | null;
  nEligibleLending: number;
  historicalAdmitted: boolean;
  fallback: boolean;
}

/** The four guardrails (docs/MATH.md §2.7) with the core's fallback arm and historical admission. */
export function eligibility(
  model: FitModel,
  b: { income: number | null; dti: number | null; subGrade: string | null; grade: string | null },
  historicalArm: number | null,
): Eligibility {
  const g = model.config.arm_space;
  const inc = b.income !== null && b.income > 0 ? b.income : null;
  const ref = referenceRate(model, b.subGrade, b.grade);
  const mask = model.grid.arms.map((a) => {
    if (a.is_null) return true;
    if (inc !== null) {
      if (g.max_payment_to_income !== null && a.installment > (g.max_payment_to_income * inc) / 12) return false;
      if (g.max_amount_to_income !== null && a.amount > g.max_amount_to_income * inc) return false;
      if (g.max_total_dti !== null && b.dti !== null && b.dti >= 0 && b.dti / 100 + (12 * a.installment) / inc > g.max_total_dti) return false;
    }
    if (g.rate_corridor !== null && ref !== null && Math.abs(a.rate - ref) > g.rate_corridor + 1e-9) return false;
    return true;
  });
  let nLend = mask.filter((m, k) => m && !model.grid.arms[k].is_null).length;
  let fallback = false;
  if (nLend === 0) {
    // the lending arm closest to the reference rate with the smallest installment
    const lend = model.grid.arms.filter((a) => !a.is_null);
    const gap = (a: { rate: number }) => (ref === null ? 0 : Math.abs(a.rate - ref));
    const best = Math.min(...lend.map(gap));
    const pick = lend.filter((a) => gap(a) <= best + 1e-9).sort((x, y) => x.installment - y.installment || x.arm - y.arm)[0];
    mask[pick.arm] = true;
    nLend = 1;
    fallback = true;
  }
  let historicalAdmitted = false;
  if (historicalArm !== null && historicalArm >= 0 && !mask[historicalArm]) {
    mask[historicalArm] = true;
    historicalAdmitted = true;
  }
  return { mask, refRate: ref, nEligibleLending: nLend, historicalAdmitted, fallback };
}

/** Group-level eligibility proxy (median income, grade reference rate) for the simulation oracle. */
export function groupEligible(model: FitModel, group: string): boolean[] {
  const g = model.groups.get(group);
  return eligibility(model, { income: g?.medianIncome ?? model.globalMedianIncome, dti: 18, subGrade: null, grade: g?.grade ?? null }, null).mask;
}

/** Best eligible arm per group (the hindsight oracle used by Simpson / regret views). */
export function groupBestArm(model: FitModel, group: string): { arm: number; value: number } {
  const ok = groupEligible(model, group);
  let best = model.grid.arms[0]?.is_null ? 0 : 1;
  let value = best === 0 ? 0 : -Infinity;
  for (const a of model.grid.arms) {
    if (a.is_null || !ok[a.arm]) continue;
    const v = riskAdjusted(model, groupArmReward(model, group, a.arm));
    if (v > value) {
      value = v;
      best = a.arm;
    }
  }
  return { arm: best, value };
}
