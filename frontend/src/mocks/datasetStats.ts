/** Dataset-level endpoints (summary, Table 3 variables) computed from the loan index. */
import type {
  CategoricalFeatureSummary,
  DatasetSummary,
  FeatureSummary,
  FeatureUnit,
  NumericFeatureSummary,
  Stage,
  VariableRow,
} from '@/api/types';
import { buildArmGrid, DEFAULT_ARM_SPACE } from '@/lib/domain';
import { type LoanIndex, isDefault, isResolved, loanRecord, termOf } from './loans';
import { seedOf, Rng } from './prng';
import { quantileSorted, round } from './stats';
import { GRADE_LETTERS, HOME_KEYS, PURPOSE_KEYS, STATUS_KEYS, TABLE3, VERIF_KEYS } from './vocab';

export interface DatasetFacts {
  n: number;
  nResolved: number;
  nDefault: number;
  statusCounts: Record<string, number>;
  issueMin: string;
  issueMax: string;
}

const factsCache = new WeakMap<LoanIndex, DatasetFacts>();

export function datasetFacts(idx: LoanIndex): DatasetFacts {
  const hit = factsCache.get(idx);
  if (hit) return hit;
  const facts = computeFacts(idx);
  factsCache.set(idx, facts);
  return facts;
}

function computeFacts(idx: LoanIndex): DatasetFacts {
  const counts = new Array<number>(STATUS_KEYS.length).fill(0);
  let res = 0;
  let def = 0;
  for (let i = 0; i < idx.n; i++) {
    counts[idx.status[i]]++;
    if (isResolved(idx, i)) {
      res++;
      if (isDefault(idx, i)) def++;
    }
  }
  const statusCounts: Record<string, number> = {};
  STATUS_KEYS.forEach((s, k) => {
    if (counts[k] > 0) statusCounts[s] = counts[k];
  });
  return {
    n: idx.n,
    nResolved: res,
    nDefault: def,
    statusCounts,
    issueMin: idx.epochs[0],
    issueMax: idx.epochs[idx.epochs.length - 1],
  };
}

/** Nearest arm on the default grid (term must match; normalised amount/rate distance). */
const DEFAULT_GRID = buildArmGrid(DEFAULT_ARM_SPACE);
export function nearestDefaultArm(amount: number, rate: number, term: number): number {
  return nearestArmIn(DEFAULT_GRID.arms, amount, rate, term);
}

export function nearestArmIn(
  arms: readonly { arm: number; amount: number; rate: number; term: number; is_null: boolean }[],
  amount: number,
  rate: number,
  term: number,
): number {
  const lend = arms.filter((a) => !a.is_null);
  const amounts = lend.map((a) => a.amount);
  const rates = lend.map((a) => a.rate);
  const aScale = Math.max(...amounts) - Math.min(...amounts) || 1;
  const rScale = Math.max(...rates) - Math.min(...rates) || 1;
  const hasTerm = lend.some((a) => a.term === term);
  let best = lend[0].arm;
  let bestD = Infinity;
  for (const a of lend) {
    if (hasTerm && a.term !== term) continue;
    const d = ((amount - a.amount) / aScale) ** 2 + ((rate - a.rate) / rScale) ** 2;
    if (d < bestD) {
      bestD = d;
      best = a.arm;
    }
  }
  return best;
}

function numericSummary(
  feature: string,
  stage: Stage,
  unit: FeatureUnit,
  values: number[],
  nullShare: number,
  edges: { lo: number; hi: number; width: number },
  scale: number,
): NumericFeatureSummary {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / Math.max(n, 1);
  const variance = sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(n - 1, 1);
  const nb = Math.round((edges.hi - edges.lo) / edges.width);
  const counts = new Array<number>(nb).fill(0);
  for (const v of sorted) {
    const k = Math.min(nb - 1, Math.max(0, Math.floor((v - edges.lo) / edges.width)));
    counts[k]++;
  }
  return {
    feature,
    kind: 'numeric',
    stage,
    unit,
    n: Math.round(n * scale),
    null_share: round(nullShare, 4),
    mean: round(mean, 4),
    std: round(Math.sqrt(variance), 4),
    p05: round(quantileSorted(sorted, 0.05), 4),
    p50: round(quantileSorted(sorted, 0.5), 4),
    p95: round(quantileSorted(sorted, 0.95), 4),
    bins: counts.map((c, k) => ({ lo: edges.lo + k * edges.width, hi: edges.lo + (k + 1) * edges.width, count: Math.round(c * scale) })),
  };
}

function categoricalSummary(feature: string, stage: Stage, levels: string[], counts: number[], scale = 1): CategoricalFeatureSummary {
  const rows = levels
    .map((level, k) => ({ level, count: Math.round(counts[k] * scale) }))
    .filter((r) => r.count > 0)
    .sort((a, b) => b.count - a.count);
  const head = rows.slice(0, 11);
  const tail = rows.slice(11).reduce((a, r) => a + r.count, 0);
  if (tail > 0) head.push({ level: 'other', count: tail });
  return { feature, kind: 'categorical', stage, n: rows.reduce((a, r) => a + r.count, 0), null_share: 0, levels: head };
}

const DEFAULT_FEATURES = [
  'fico_range_low',
  'dti',
  'annual_inc',
  'revol_util',
  'inq_last_6mths',
  'emp_length',
  'loan_amnt',
  'int_rate',
  'grade',
  'purpose',
  'home_ownership',
  'verification_status',
  'term',
];

export function datasetSummary(datasetId: number, idx: LoanIndex, wanted: string[] | null): DatasetSummary {
  const features = wanted?.length ? wanted : DEFAULT_FEATURES;
  // 1-in-k systematic sample for quantiles and row-level bureau fields
  const step = Math.max(1, Math.floor(idx.n / 20000));
  const sample: number[] = [];
  for (let i = 0; i < idx.n; i += step) sample.push(i);
  const scale = idx.n / sample.length;
  const rowStep = Math.max(1, Math.floor(idx.n / 3000));
  const rows: { revol: number | null; inq: number | null }[] = [];
  for (let i = 0; i < idx.n; i += rowStep) {
    const r = loanRecord(idx, i, nearestDefaultArm);
    rows.push({ revol: r.context.revol_util, inq: r.context.inq_last_6mths });
  }
  const rowScale = idx.n / rows.length;

  const count = (arr: ArrayLike<number>, levels: number) => {
    const c = new Array<number>(levels).fill(0);
    for (let i = 0; i < idx.n; i++) c[arr[i]]++;
    return c;
  };

  const out: FeatureSummary[] = [];
  for (const f of features) {
    switch (f) {
      case 'fico_range_low':
        out.push(numericSummary(f, 'pre_decision', 'score', sample.map((i) => idx.fico[i]), 0, { lo: 600, hi: 850, width: 10 }, scale));
        break;
      case 'dti':
        out.push(numericSummary(f, 'pre_decision', 'pct', sample.map((i) => idx.dti[i]), 0.0008, { lo: 0, hi: 50, width: 2.5 }, scale));
        break;
      case 'annual_inc':
        out.push(
          numericSummary(f, 'pre_decision', 'usd', sample.map((i) => idx.income[i]), 0, { lo: 0, hi: 250000, width: 12500 }, scale),
        );
        break;
      case 'emp_length': {
        const vals = sample.filter((i) => idx.emp[i] !== 255).map((i) => idx.emp[i]);
        out.push(numericSummary(f, 'pre_decision', 'years', vals, 1 - vals.length / sample.length, { lo: 0, hi: 11, width: 1 }, scale));
        break;
      }
      case 'loan_amnt':
        out.push(numericSummary(f, 'policy', 'usd', sample.map((i) => idx.amount[i]), 0, { lo: 0, hi: 40000, width: 2000 }, scale));
        break;
      case 'int_rate':
        out.push(numericSummary(f, 'policy', 'fraction', sample.map((i) => idx.rate[i]), 0, { lo: 0.05, hi: 0.31, width: 0.013 }, scale));
        break;
      case 'revol_util': {
        const vals = rows.map((r) => r.revol).filter((v): v is number => v !== null);
        out.push(numericSummary(f, 'pre_decision', 'fraction', vals, 0.0008, { lo: 0, hi: 1.2, width: 0.06 }, rowScale));
        break;
      }
      case 'inq_last_6mths': {
        const vals = rows.map((r) => r.inq).filter((v): v is number => v !== null);
        out.push(numericSummary(f, 'pre_decision', 'count', vals, 0, { lo: 0, hi: 8, width: 1 }, rowScale));
        break;
      }
      case 'grade':
        out.push(categoricalSummary(f, 'pre_decision', GRADE_LETTERS, count(idx.grade, 7)));
        break;
      case 'purpose':
        out.push(categoricalSummary(f, 'pre_decision', PURPOSE_KEYS, count(idx.purpose, PURPOSE_KEYS.length)));
        break;
      case 'home_ownership':
        out.push(categoricalSummary(f, 'pre_decision', HOME_KEYS, count(idx.home, 4)));
        break;
      case 'verification_status':
        out.push(categoricalSummary(f, 'pre_decision', VERIF_KEYS, count(idx.verif, 3)));
        break;
      case 'term':
        out.push(categoricalSummary(f, 'policy', ['36', '60'], count(idx.term60, 2)));
        break;
      default:
        break;
    }
  }

  const arrivals = idx.epochs.map((epoch) => ({ epoch, n: 0, n_resolved: 0, n_default: 0 }));
  const gradeCounts: Record<string, number> = {};
  const termCounts: Record<string, number> = { '36': 0, '60': 0 };
  for (let i = 0; i < idx.n; i++) {
    const a = arrivals[idx.issue[i]];
    a.n++;
    if (isResolved(idx, i)) {
      a.n_resolved++;
      if (isDefault(idx, i)) a.n_default++;
    }
    const g = GRADE_LETTERS[idx.grade[i]];
    gradeCounts[g] = (gradeCounts[g] ?? 0) + 1;
    termCounts[String(termOf(idx, i))]++;
  }
  return {
    dataset: datasetId,
    n_loans: idx.n,
    features: out,
    monthly_arrivals: arrivals,
    status_counts: datasetFacts(idx).statusCounts,
    grade_counts: gradeCounts,
    term_counts: termCounts,
  };
}

export function variableRows(idx: LoanIndex, source: string): VariableRow[] {
  const rng = new Rng(seedOf('vars', idx.seed, source));
  let empNull = 0;
  for (let i = 0; i < idx.n; i++) if (idx.emp[i] === 255) empNull++;
  return TABLE3.map((v) => {
    let nullShare = v.null_share;
    if (v.name === 'emp_length') nullShare = empNull / idx.n;
    else if (nullShare > 0) nullShare = Math.min(0.999, Math.max(0, nullShare * (1 + 0.02 * rng.normal())));
    return {
      name: v.name,
      stage: v.stage,
      primary_use: v.primary_use,
      kind: v.kind,
      present: true,
      dtype: v.dtype,
      null_share: Math.round(nullShare * 10000) / 10000,
    };
  });
}
