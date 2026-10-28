/**
 * Mock of the structured leaf prior (docs/MATH.md §2.6): an offset logistic regression
 * logit P(D) = logit(m_g) + β·(x − x̄_g) on within-group-centred features
 * x = [int_rate, 1[term = 60], payment-to-income, log amount], fitted by Newton–Raphson with a
 * small ridge on standardised features; standard errors from the observed information.
 */
import type { PolicyEffectConfig, PolicyEffectFeature } from '@/api/types';

export interface PolicyEffectFit {
  status: 'fitted' | 'fallback' | 'disabled';
  reason: string | null;
  features: PolicyEffectFeature[];
  beta: number[];
  se: number[];
  nFit: number;
  sample: 'elapsed' | 'resolved';
  converged: boolean;
  nIter: number;
  maxShift: number;
}

export const ALL_FEATURES: PolicyEffectFeature[] = ['int_rate', 'term60', 'pti', 'log_amount'];

/** Solve A x = b (small dense system, partial pivoting). */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / (M[r][r] || 1e-12);
  }
  return x;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/**
 * rows: centred features xc (row-major, k per row), offsets o, outcomes y.
 * Returns β in original units with standard errors.
 */
export function fitOffsetLogit(
  xc: Float64Array,
  o: Float64Array,
  y: Uint8Array,
  k: number,
  cfg: PolicyEffectConfig,
): { beta: number[]; se: number[]; converged: boolean; nIter: number } {
  const n = y.length;
  const sd = new Array<number>(k).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) sd[j] += xc[i * k + j] ** 2;
  for (let j = 0; j < k; j++) sd[j] = Math.sqrt(sd[j] / Math.max(n - 1, 1)) || 1;
  let bs = new Array<number>(k).fill(0);
  const loglik = (b: number[]) => {
    let ll = 0;
    for (let i = 0; i < n; i++) {
      let eta = o[i];
      for (let j = 0; j < k; j++) eta += (xc[i * k + j] / sd[j]) * b[j];
      ll += y[i] ? -Math.log1p(Math.exp(-eta)) : -Math.log1p(Math.exp(eta));
    }
    return ll / n - (cfg.ridge / 2) * b.reduce((a, v) => a + v * v, 0);
  };
  let current = loglik(bs);
  let converged = false;
  let it = 0;
  for (; it < cfg.max_iter; it++) {
    const g = new Array<number>(k).fill(0);
    const H = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    for (let i = 0; i < n; i++) {
      let eta = o[i];
      for (let j = 0; j < k; j++) eta += (xc[i * k + j] / sd[j]) * bs[j];
      const p = sigmoid(eta);
      const w = p * (1 - p);
      for (let a = 0; a < k; a++) {
        const za = xc[i * k + a] / sd[a];
        g[a] += za * (y[i] - p);
        for (let b = a; b < k; b++) H[a][b] += w * za * (xc[i * k + b] / sd[b]);
      }
    }
    for (let a = 0; a < k; a++) {
      g[a] = g[a] / n - cfg.ridge * bs[a];
      for (let b = a; b < k; b++) {
        H[a][b] = H[a][b] / n + (a === b ? cfg.ridge : 0);
        H[b][a] = H[a][b];
      }
    }
    let step = solve(H, g);
    let t = 1;
    let next = bs.map((v, j) => v + step[j]);
    let ll = loglik(next);
    while (ll < current && t > 1e-3) {
      t /= 2;
      step = step.map((v) => v / 2);
      next = bs.map((v, j) => v + step[j]);
      ll = loglik(next);
    }
    bs = next;
    current = ll;
    if (Math.max(...step.map(Math.abs)) < 1e-7) {
      converged = true;
      it++;
      break;
    }
  }
  // observed information in original units for the standard errors
  const I = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const beta = bs.map((v, j) => v / sd[j]);
  for (let i = 0; i < n; i++) {
    let eta = o[i];
    for (let j = 0; j < k; j++) eta += xc[i * k + j] * beta[j];
    const p = sigmoid(eta);
    const w = p * (1 - p);
    for (let a = 0; a < k; a++) for (let b = a; b < k; b++) I[a][b] += w * xc[i * k + a] * xc[i * k + b];
  }
  for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) I[a][b] = I[b][a];
  const se = beta.map((_, j) => {
    const e = new Array<number>(k).fill(0);
    e[j] = 1;
    return Math.sqrt(Math.max(solve(I, e)[j], 0));
  });
  return { beta, se, converged, nIter: it };
}
