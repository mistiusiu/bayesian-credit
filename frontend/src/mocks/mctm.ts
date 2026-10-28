/**
 * Markov Chain Transition Matrix mock (SPEC §2.5). States C, D, R, L with R and L absorbing.
 * PROPOSAL forces P[C,R] = P[C,L] = 0; EXTENDED also allows C → R (prepayment / payoff).
 * Counts are drawn around a target matrix; P is the Dirichlet posterior mean; trajectories,
 * absorption probabilities and expected times are computed exactly from P.
 */
import type { CreditState, Mctm, TransitionStructure } from '@/api/types';
import { Rng, seedOf } from './prng';
import { normalQuantile, round } from './stats';

export const STATES: CreditState[] = ['current', 'delinquent', 'recovered', 'loss'];
const LABELS = ['Current', 'Delinquent', 'Recovered', 'Absorbed Loss'];

const TARGET: Record<TransitionStructure, number[][]> = {
  proposal: [
    [0.985, 0.015, 0, 0],
    [0.35, 0.4, 0.05, 0.2],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
  extended: [
    [0.9725, 0.015, 0.0125, 0],
    [0.35, 0.4, 0.05, 0.2],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
};

export function structuralZeros(structure: TransitionStructure): boolean[][] {
  return TARGET[structure].map((row, i) => row.map((v, j) => (i >= 2 ? i !== j : v === 0)));
}

export interface MctmCore {
  matrix: number[][];
  counts: number[][];
  low: number[][];
  high: number[][];
  ll: number;
  n: number;
  rowErr: number[];
}

/** Posterior matrix and holdout diagnostics for a fit (scale = loans / 200k). */
export function mctmCore(seedKey: string, structure: TransitionStructure, priorStrength: number, scale: number): MctmCore {
  const rng = new Rng(seedOf('mctm', seedKey, structure));
  const zeros = structuralZeros(structure);
  const target = TARGET[structure];
  const totals = [4_120_000 * scale, 92_000 * scale];
  const counts = target.map((row) => row.map(() => 0));
  const matrix = target.map((row) => row.slice());
  const low = target.map((row) => row.slice());
  const high = target.map((row) => row.slice());
  const z = normalQuantile(0.975);
  let ll = 0;
  let n = 0;
  const rowErr = [0, 0, 0, 0];
  for (let i = 0; i < 2; i++) {
    const allowed = target[i].map((_, j) => !zeros[i][j]);
    for (let j = 0; j < 4; j++) {
      if (!allowed[j]) continue;
      counts[i][j] = Math.max(1, Math.round(target[i][j] * totals[i] * (1 + 0.015 * rng.normal())));
    }
    const k = allowed.filter(Boolean).length;
    const rowTotal = counts[i].reduce((a, b) => a + b, 0);
    const a0 = rowTotal + k * priorStrength;
    for (let j = 0; j < 4; j++) {
      if (!allowed[j]) {
        matrix[i][j] = 0;
        low[i][j] = 0;
        high[i][j] = 0;
        continue;
      }
      const a = counts[i][j] + priorStrength;
      const m = a / a0;
      const sd = Math.sqrt((m * (1 - m)) / (a0 + 1));
      matrix[i][j] = m;
      low[i][j] = Math.max(0, m - z * sd);
      high[i][j] = Math.min(1, m + z * sd);
    }
    // holdout: ~35% of the training volume, empirical frequencies jittered around P
    let errSum = 0;
    for (let j = 0; j < 4; j++) {
      if (!allowed[j]) continue;
      const hc = Math.max(1, Math.round(matrix[i][j] * totals[i] * 0.35 * (1 + 0.04 * rng.normal())));
      ll += hc * Math.log(matrix[i][j]);
      n += hc;
      errSum += Math.abs(hc / (totals[i] * 0.35) - matrix[i][j]);
    }
    rowErr[i] = errSum / k;
  }
  return { matrix, counts, low, high, ll, n, rowErr };
}

const mulVec = (pi: number[], P: number[][]) => [0, 1, 2, 3].map((j) => pi.reduce((s, v, i) => s + v * P[i][j], 0));

export function absorption(P: number[][]): { B: number[][]; t: number[] } {
  // Q = transient block, N = (I − Q)⁻¹ (2×2 closed form), B = N·R, t = N·1
  const a = 1 - P[0][0];
  const b = -P[0][1];
  const c = -P[1][0];
  const d = 1 - P[1][1];
  const det = a * d - b * c;
  const N = [
    [d / det, -b / det],
    [-c / det, a / det],
  ];
  const B = [0, 1].map((i) => [2, 3].map((j) => N[i][0] * P[0][j] + N[i][1] * P[1][j]));
  const t = [0, 1].map((i) => N[i][0] + N[i][1]);
  return { B, t };
}

export function mctmPayload(
  fitId: number,
  core: MctmCore,
  structure: TransitionStructure,
  horizon: number,
  start: 'current' | 'delinquent',
  priorStrength: number,
): Mctm {
  const P = core.matrix;
  let pi = start === 'current' ? [1, 0, 0, 0] : [0, 1, 0, 0];
  const trajectory = [];
  const curve = [];
  for (let t = 0; t <= horizon; t++) {
    trajectory.push({ t, current: round(pi[0], 6), delinquent: round(pi[1], 6), recovered: round(pi[2], 6), loss: round(pi[3], 6) });
    curve.push({ t, p_default: round(pi[1] + pi[3], 6), p_loss: round(pi[3], 6), p_delinquent: round(pi[1], 6) });
    if (t < horizon) pi = mulVec(pi, P);
  }
  const pDR = P[1][2];
  const pDL = P[1][3];
  const split = pDR + pDL > 0 ? pDL / (pDR + pDL) : 0;
  const after = [0, 0, pi[2] + pi[0] + pi[1] * (1 - split), pi[3] + pi[1] * split];
  const { B, t } = absorption(P);
  const r6 = (m: number[][]) => m.map((row) => row.map((v) => round(v, 6)));
  return {
    fit: fitId,
    structure,
    states: STATES,
    state_labels: LABELS,
    matrix: r6(P),
    counts: core.counts,
    structural_zeros: structuralZeros(structure),
    matrix_low: r6(core.low),
    matrix_high: r6(core.high),
    absorption: r6(B),
    expected_steps: t.map((v) => round(v, 3)),
    start,
    horizon,
    trajectory,
    p_default_curve: curve,
    maturity: { t: horizon, before: pi.map((v) => round(v, 6)), after: after.map((v) => round(v, 6)), p_default: round(after[3], 6) },
    log_likelihood: round(core.ll, 1),
    n_transitions: core.n,
    row_prediction_error: core.rowErr.map((v) => round(v, 6)),
    prior_strength: priorStrength,
  };
}
