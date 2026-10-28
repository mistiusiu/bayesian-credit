/**
 * PSI Sentinel mock (SPEC §2.9, Table 9). The aggregate PSI path drifts upward, trips the
 * breaker at the dataset's shift epoch, and then depends on the human action:
 * resegment / rollback rebase the reference (PSI falls, the breaker closes after
 * `recovery_readings` normal readings); resume keeps the reference (PSI stays in the
 * watch band and the breaker stays half-open). Bucket tables are tilted reference
 * distributions solved so Σ(a−e)ln(a/e) equals the reported PSI exactly.
 */
import type {
  BreakerState,
  BreakerTransition,
  PsiBucket,
  PsiBucketTable,
  ReviewAction,
  SentinelLevel,
  SimulationConfig,
} from '@/api/types';

import { Rng, seedOf } from './prng';
import { round } from './stats';

export interface ReviewDecision {
  /** Epoch at which the breaker tripped (review created). */
  tripEpoch: string;
  action: ReviewAction | null;
  /** Epoch whose processing applied the action (trip + auto_ack_epochs, or the resume epoch in pause mode). */
  ackEpoch: string | null;
}

export interface SentinelSeries {
  epochs: string[];
  features: string[];
  /** psi[f][e] */
  psi: number[][];
  aggregate: number[];
  level: SentinelLevel[];
  breaker: BreakerState[];
  transitions: BreakerTransition[];
  /** Index of the trip epoch, or −1. */
  tripIndex: number;
  rebasedAt: string[];
}

const SENSITIVITY: Record<string, number> = {
  fico_range_low: 1,
  dti: 0.86,
  annual_inc: 0.2,
  revol_util: 0.16,
  inq_last_6mths: 0.28,
  grade: 0.34,
  emp_length: 0.12,
  verification_status: 0.1,
};

export function levelOf(psi: number, cfg: SimulationConfig): SentinelLevel {
  if (psi > cfg.psi.thresholds.breaker) return 'breaker';
  if (psi >= cfg.psi.thresholds.watch) return 'watch';
  return 'normal';
}

/**
 * Aggregate PSI path. `shiftIdx` = epoch index of the injected shift (−1: no shift → drift only).
 * `decision` decides what happens after a trip.
 */
export function sentinelSeries(
  seedKey: string,
  epochs: string[],
  cfg: SimulationConfig,
  shiftIdx: number,
  decision: ReviewDecision | null,
  ackTs: (epoch: string) => string,
): SentinelSeries {
  const rng = new Rng(seedOf('psi', seedKey));
  const features = cfg.psi.features.length ? cfg.psi.features : ['fico_range_low'];
  const n = epochs.length;
  const agg: number[] = [];
  const breakerThr = cfg.psi.thresholds.breaker;
  const ackIdx = decision?.ackEpoch ? epochs.indexOf(decision.ackEpoch) : -1;
  const rebases = decision && decision.action && decision.action !== 'resume' && ackIdx >= 0;
  for (let e = 0; e < n; e++) {
    let v: number;
    if (shiftIdx < 0) {
      v = 0.025 + 0.085 * (e / Math.max(n - 1, 1)) ** 1.4;
    } else if (e < shiftIdx) {
      // upward drift that accelerates into the shift
      const lead = Math.max(0, 6 - (shiftIdx - e));
      v = 0.028 + 0.0028 * e + 0.019 * lead * lead * 0.5;
    } else if (rebases && e >= ackIdx) {
      const since = e - ackIdx;
      v = 0.045 + 0.0024 * since + 0.006 * Math.max(0, 3 - since);
    } else if (decision?.action === 'resume' && ackIdx >= 0 && e >= ackIdx) {
      v = 0.18 + 0.045 * Math.exp(-(e - ackIdx) / 4);
    } else {
      v = breakerThr + 0.04 + 0.03 * Math.min(1, (e - shiftIdx) / 2);
    }
    agg.push(Math.max(0.004, v * (1 + 0.05 * rng.normal())));
  }
  // force the trip epoch over the threshold and the epoch before under it
  if (shiftIdx >= 0 && shiftIdx < n) {
    agg[shiftIdx] = Math.max(agg[shiftIdx], breakerThr + 0.036);
    if (shiftIdx > 0) agg[shiftIdx - 1] = Math.min(agg[shiftIdx - 1], breakerThr - 0.04);
  }
  const psi = features.map((f, k) => {
    const s = SENSITIVITY[f] ?? 0.15;
    return agg.map((a) => {
      const lead = k === 0 ? 1 : s;
      const base = 0.008 + 0.006 * rng.next();
      return round(Math.max(0.002, base + (a - 0.008) * lead * (0.92 + 0.08 * rng.next())), 4);
    });
  });
  const aggregate = epochs.map((_, e) => {
    const vals = psi.map((row) => row[e]);
    return cfg.psi.aggregate === 'mean' ? vals.reduce((x, y) => x + y, 0) / vals.length : Math.max(...vals);
  });
  const level = aggregate.map((a) => levelOf(a, cfg));

  const breaker: BreakerState[] = [];
  const transitions: BreakerTransition[] = [];
  let state: BreakerState = 'closed';
  let normalRun = 0;
  let tripIndex = -1;
  for (let e = 0; e < n; e++) {
    const epoch = epochs[e];
    if (cfg.breaker_mode !== 'off') {
      if (state === 'closed' && level[e] === 'breaker') {
        state = 'open';
        tripIndex = tripIndex < 0 ? e : tripIndex;
        transitions.push({ epoch, ts: ackTs(epoch), from_state: 'closed', to_state: 'open', reason: `aggregate PSI ${aggregate[e].toFixed(3)} > ${breakerThr}`, psi_aggregate: round(aggregate[e], 4), review: null, action: null });
      } else if (state === 'open' && ackIdx >= 0 && e >= ackIdx && decision?.action) {
        state = 'half_open';
        normalRun = 0;
        transitions.push({ epoch, ts: ackTs(epoch), from_state: 'open', to_state: 'half_open', reason: `review acknowledged: ${decision.action}`, psi_aggregate: round(aggregate[e], 4), review: null, action: decision.action });
      } else if (state === 'half_open') {
        normalRun = level[e] === 'normal' ? normalRun + 1 : 0;
        if (normalRun >= cfg.psi.recovery_readings) {
          state = 'closed';
          transitions.push({ epoch, ts: ackTs(epoch), from_state: 'half_open', to_state: 'closed', reason: `${cfg.psi.recovery_readings} consecutive normal readings`, psi_aggregate: round(aggregate[e], 4), review: null, action: null });
        }
      }
    } else if (level[e] === 'breaker' && tripIndex < 0) {
      tripIndex = e;
    }
    breaker.push(state);
  }
  return {
    epochs,
    features,
    psi,
    aggregate: aggregate.map((a) => round(a, 4)),
    level,
    breaker,
    transitions,
    tripIndex,
    rebasedAt: rebases && decision?.ackEpoch ? [decision.ackEpoch] : [],
  };
}

interface BucketSpec {
  labels: string[];
  lo: (number | null)[];
  hi: (number | null)[];
  expected: number[];
  kind: 'numeric' | 'categorical';
  /** +1: drift moves mass to higher buckets, −1: lower. */
  direction: number;
}

function decileSpec(edges: number[], direction: number, fmtEdge: (v: number) => string): BucketSpec {
  const labels: string[] = [];
  for (let k = 0; k < edges.length - 1; k++) labels.push(`[${fmtEdge(edges[k])}, ${fmtEdge(edges[k + 1])})`);
  return { labels, lo: edges.slice(0, -1), hi: edges.slice(1), expected: labels.map(() => 1 / labels.length), kind: 'numeric', direction };
}

const k = (v: number) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v));

const SPECS: Record<string, BucketSpec> = {
  fico_range_low: decileSpec([660, 670, 680, 690, 700, 710, 725, 740, 760, 785, 850], -1, String),
  dti: decileSpec([0, 7.5, 10.6, 13, 15.3, 17.6, 20, 22.6, 25.6, 29.6, 50], 1, String),
  annual_inc: decileSpec([0, 36000, 45000, 52000, 60000, 67000, 76000, 87000, 100000, 125000, 2000000], -1, k),
  revol_util: decileSpec([0, 0.19, 0.29, 0.37, 0.44, 0.51, 0.58, 0.65, 0.73, 0.84, 1.2], 1, (v) => v.toFixed(2)),
  inq_last_6mths: { labels: ['0', '1', '2', '3', '4+'], lo: [0, 1, 2, 3, 4], hi: [1, 2, 3, 4, null], expected: [0.55, 0.27, 0.11, 0.045, 0.025], kind: 'categorical', direction: 1 },
  grade: { labels: ['A', 'B', 'C', 'D', 'E', 'F', 'G'], lo: [null, null, null, null, null, null, null], hi: [null, null, null, null, null, null, null], expected: [0.2, 0.3, 0.27, 0.14, 0.06, 0.022, 0.008], kind: 'categorical', direction: 1 },
};

const psiOf = (a: number[], e: number[]) => a.reduce((s, ai, i) => s + (ai - e[i]) * Math.log(ai / e[i]), 0);

export function bucketTable(simId: number, feature: string, epoch: string, psi: number, level: SentinelLevel, nRef: number, nWin: number): PsiBucketTable {
  const spec = SPECS[feature] ?? decileSpec([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, String);
  const m = spec.labels.length;
  const x = spec.labels.map((_, i) => spec.direction * ((i - (m - 1) / 2) / ((m - 1) / 2)));
  const tilt = (theta: number) => {
    const w = spec.expected.map((e, i) => e * Math.exp(theta * x[i]));
    const s = w.reduce((a, b) => a + b, 0);
    return w.map((v) => v / s);
  };
  let lo = 0;
  let hi = 8;
  for (let it = 0; it < 60; it++) {
    const mid = (lo + hi) / 2;
    if (psiOf(tilt(mid), spec.expected) < psi) lo = mid;
    else hi = mid;
  }
  const actual = tilt((lo + hi) / 2);
  const buckets: PsiBucket[] = spec.labels.map((label, i) => ({
    bucket: label,
    lo: spec.lo[i],
    hi: spec.hi[i],
    expected: round(spec.expected[i], 4),
    actual: round(actual[i], 4),
    contribution: round((actual[i] - spec.expected[i]) * Math.log(actual[i] / spec.expected[i]), 5),
  }));
  return { simulation: simId, feature, epoch, kind: spec.kind, psi: round(psi, 4), level, n_reference: nRef, n_window: nWin, buckets };
}

