/**
 * Replay-simulation mock (SPEC §2.11). Per-decision expected rewards are anchored to the
 * fit's own reward table: the hindsight oracle (best arm per group), the historical policy
 * (Lending Club's arms) and the uniform-arm mean. Adaptive policies close the gap to the
 * oracle at policy-specific rates; while the breaker is open their learning clock stops.
 */
import type { PolicyName, SimulationConfig } from '@/api/types';
import { policyBase } from '@/lib/domain';
import { epochRange } from '@/lib/epoch';
import type { LoanIndex } from './loans';
import { Rng, seedOf } from './prng';
import { type FitModel, groupArmReward, groupBestArm, groupEligible } from './riskModel';
import { type ReviewDecision, type SentinelSeries, sentinelSeries } from './sentinel';

export interface PolicySeries {
  policy: PolicyName;
  perDecision: number[];
  reward: number[];
  regret: number[];
  realized: number[];
  /** λ(e): how concentrated the policy is on the per-group oracle arms. */
  focus: number[];
  shares: Float64Array[];
}

export interface GroupMeta {
  key: string;
  w: number;
  best: number;
  hist: Float64Array;
  /** 1 where the arm passes the group's guardrails (null arm always 1). */
  ok: Float64Array;
  /** Uniform over the group's eligible lending arms (the random baseline). */
  uniform: Float64Array;
}

export interface SimSeries {
  epochs: string[];
  n: number[];
  oracle: number[];
  groups: GroupMeta[];
  broad: Float64Array;
  uniform: Float64Array;
  histShare: Float64Array;
  policies: Map<PolicyName, PolicySeries>;
  sentinel: SentinelSeries;
  shiftIdx: number;
}

/** Learning curves by policy kind (spec labels such as `ucb1(ucb_c=1.0)` use their base kind). */
const LEARNER: Partial<Record<string, { start: number; floor: number; tau: number }>> = {
  thompson: { start: 0.85, floor: 0.1, tau: 4.5 },
  thompson_se: { start: 0.82, floor: 0.13, tau: 3.4 },
  thompson_flat: { start: 0.9, floor: 0.34, tau: 6 },
  ucb1: { start: 0.9, floor: 0.2, tau: 8 },
  epsilon_greedy: { start: 0.9, floor: 0.3, tau: 6.5 },
  greedy: { start: 0.8, floor: 0.45, tau: 3 },
};

/** Share vector of an adaptive / baseline policy kind inside one group (eligible arms only). */
function shareIn(kind: string, lam: number, broad: Float64Array, g: GroupMeta): Float64Array {
  if (kind === 'historical') return g.hist;
  if (kind === 'random') return g.uniform;
  const K = broad.length;
  const s = new Float64Array(K);
  let z = 0;
  let nOk = 0;
  for (let a = 0; a < K; a++) {
    z += broad[a] * g.ok[a];
    nOk += g.ok[a];
  }
  for (let a = 0; a < K; a++) s[a] = ((1 - lam) * broad[a] * g.ok[a]) / (z || 1);
  s[g.best] += lam;
  if (kind === 'epsilon_greedy') for (let a = 0; a < K; a++) s[a] = 0.9 * s[a] + (0.1 * g.ok[a]) / (nOk || 1);
  return s;
}

export function simSeries(
  model: FitModel,
  idx: LoanIndex,
  cfg: SimulationConfig,
  seedKey: string,
  shiftEpoch: string | null,
  decision: ReviewDecision | null,
  ackTs: (epoch: string) => string,
): SimSeries {
  const epochs = epochRange(cfg.start, cfg.end);
  const raw = epochs.map((e) => {
    const k = idx.epochs.indexOf(e);
    return k >= 0 ? idx.monthCounts[k] : 0;
  });
  const total = raw.reduce((a, b) => a + b, 0);
  const scale = cfg.max_loans && cfg.max_loans < total ? cfg.max_loans / total : 1;
  const n = raw.map((c) => Math.round(c * scale));
  const shiftIdx = shiftEpoch ? epochs.indexOf(shiftEpoch) : -1;
  const sentinel = sentinelSeries(seedKey, epochs, cfg, shiftIdx, decision, ackTs);

  const arms = model.grid.arms;
  const K = arms.length;
  const nullArm = arms.findIndex((a) => a.is_null);
  const groups: GroupMeta[] = [...model.groups.values()].map((g) => {
    const hist = new Float64Array(K);
    let s = 0;
    for (const [arm, leaf] of g.leaves) {
      hist[arm] = leaf.n;
      s += leaf.n;
    }
    for (let a = 0; a < K; a++) hist[a] /= s || 1;
    const ok = Float64Array.from(groupEligible(model, g.key), (b) => (b ? 1 : 0));
    const uniform = Float64Array.from(ok, (v, a) => (arms[a].is_null ? 0 : v));
    const nu = uniform.reduce((x, y) => x + y, 0);
    for (let a = 0; a < K; a++) uniform[a] /= nu || 1;
    const best = groupBestArm(model, g.key);
    return { key: g.key, w: g.weight, best: best.value > 0 || nullArm < 0 ? best.arm : nullArm, hist, ok, uniform };
  });

  let O = 0;
  let H = 0;
  let U = 0;
  let W = 0;
  const pooled = new Float64Array(K);
  const pooledW = new Float64Array(K);
  const histShare = new Float64Array(K);
  const uniform = new Float64Array(K);
  for (const g of groups) {
    const bestVal = arms[g.best].is_null ? 0 : groupArmReward(model, g.key, g.best).mean;
    O += g.w * Math.max(0, bestVal);
    W += g.w;
    for (let a = 0; a < K; a++) {
      const r = arms[a].is_null ? 0 : groupArmReward(model, g.key, a).mean;
      pooled[a] += g.w * g.ok[a] * r;
      pooledW[a] += g.w * g.ok[a];
      H += g.w * g.hist[a] * r;
      U += g.w * g.uniform[a] * r;
      histShare[a] += g.w * g.hist[a];
      uniform[a] += g.w * g.uniform[a];
    }
  }
  for (let a = 0; a < K; a++) {
    pooled[a] /= pooledW[a] || 1;
    histShare[a] /= W || 1;
    uniform[a] /= W || 1;
  }
  // broad warm-start distribution: softmax of pooled expected reward over arms eligible somewhere
  const live = [...pooled].filter((_, a) => pooledW[a] > 0);
  const mean = live.reduce((a, b) => a + b, 0) / Math.max(live.length, 1);
  const spread = Math.sqrt(live.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(live.length, 1)) || 1;
  const top = Math.max(...live);
  const broad = new Float64Array(K);
  let bs = 0;
  for (let a = 0; a < K; a++) {
    broad[a] = pooledW[a] > 0 ? (pooledW[a] / (W || 1)) * Math.exp((pooled[a] - top) / (0.9 * spread)) : 0;
    bs += broad[a];
  }
  for (let a = 0; a < K; a++) broad[a] /= bs || 1;

  const post = (e: number) => (shiftIdx >= 0 && e >= shiftIdx ? 1 : 0);
  const oracle = epochs.map((_, e) => O * (1 - 0.06 * post(e)));
  const floorU = epochs.map((_, e) => U * (1 - 0.07 * post(e)));
  const histPer = epochs.map((_, e) => H * (1 - 0.09 * post(e)));
  const frozen = sentinel.breaker.map((b) => b === 'open');
  const ackIdx = decision?.ackEpoch ? epochs.indexOf(decision.ackEpoch) : -1;
  const rebased = !!decision?.action && decision.action !== 'resume';

  const gapFraction = (p: PolicyName, floorScale: number) => {
    const L = LEARNER[policyBase(p)];
    let clock = 0;
    return epochs.map((_, e) => {
      if (!L) return 0;
      if (!frozen[e]) clock++;
      let g = L.floor * floorScale + (L.start - L.floor * floorScale) * Math.exp(-clock / L.tau);
      if (shiftIdx >= 0 && e >= shiftIdx) {
        const tau = rebased && ackIdx >= 0 && e >= ackIdx ? 1.5 : 3.5;
        g += 0.12 * Math.exp(-(e - shiftIdx) / tau);
      }
      return Math.min(1, g);
    });
  };

  const build = (floorScale: number) => {
    const out = new Map<PolicyName, number[]>();
    for (const p of cfg.policies) {
      const kind = policyBase(p);
      if (kind === 'historical') out.set(p, histPer.slice());
      else if (kind === 'random') out.set(p, floorU.slice());
      else {
        const g = gapFraction(p, floorScale);
        out.set(p, epochs.map((_, e) => oracle[e] - g[e] * (oracle[e] - floorU[e])));
      }
    }
    return out;
  };
  // guarantee thompson > ucb1 > ε-greedy > historical > random in cumulative reward
  let floorScale = 1;
  let per = build(floorScale);
  const cum = (p: PolicyName) => (per.get(p) ?? []).reduce((a, r, e) => a + r * n[e], 0);
  for (let it = 0; it < 12; it++) {
    const ok =
      (!per.has('epsilon_greedy') || !per.has('historical') || cum('epsilon_greedy') > cum('historical') * 1.02) &&
      (!per.has('greedy') || !per.has('historical') || cum('greedy') > cum('historical') * 1.01);
    if (ok) break;
    floorScale *= 0.8;
    per = build(floorScale);
  }

  // Resolution delay kernel: share of an epoch's reward realised k epochs later (85% within 30 epochs).
  const kernel = Array.from({ length: 30 }, (_, i) => (i + 1) ** 1.2 * Math.exp(-(i + 1) / 6));
  const kernelSum = kernel.reduce((a, b) => a + b, 0);
  const delay = kernel.map((w) => (0.85 * w) / kernelSum);

  const policies = new Map<PolicyName, PolicySeries>();
  for (const p of cfg.policies) {
    const kind = policyBase(p);
    const r = per.get(p) ?? [];
    const prng = new Rng(seedOf('policy', seedKey, cfg.seed, p));
    const reward = r.map((v, e) => n[e] * v * (1 + 0.025 * prng.normal()));
    const regret = r.map((v, e) => Math.max(0, n[e] * (oracle[e] - v) * (1 + 0.02 * prng.normal())));
    const realized = reward.map((_, e) => {
      let s = 0;
      for (let k = 1; k <= Math.min(e, 30); k++) s += reward[e - k] * delay[k - 1];
      const penalty = kind === 'historical' && shiftIdx >= 0 && e >= shiftIdx ? 0.92 : 1;
      return s * penalty * (1 + 0.04 * prng.normal());
    });
    const L = LEARNER[kind];
    const focus = r.map((v, e) => {
      if (kind === 'historical' || kind === 'random') return 0;
      const gap = (oracle[e] - v) / Math.max(oracle[e] - floorU[e], 1e-9);
      return Math.min(0.97, Math.max(0, 1 - gap / (L?.start ?? 1)));
    });
    const shares = epochs.map((_, e) => {
      const s = new Float64Array(K);
      if (kind === 'historical') s.set(histShare);
      else if (kind === 'random') s.set(uniform);
      else
        for (const g of groups) {
          const gs = shareIn(kind, focus[e], broad, g);
          for (let a = 0; a < K; a++) s[a] += (g.w / (W || 1)) * gs[a];
        }
      return s;
    });
    policies.set(p, { policy: p, perDecision: r, reward, regret, realized, focus, shares });
  }
  return { epochs, n, oracle, groups, broad, uniform, histShare, policies, sentinel, shiftIdx };
}

/** Group-level share vector of policy p at epoch e. */
export function groupShare(series: SimSeries, p: PolicyName, e: number, g: GroupMeta): Float64Array {
  const ps = series.policies.get(p);
  if (!ps) return new Float64Array(series.broad.length);
  return shareIn(policyBase(p), ps.focus[e], series.broad, g);
}

/**
 * Bandit posterior of (group, arm) at epoch e: the warm start (risk-model reward) is pulled
 * towards the hindsight truth as effective observations accumulate.
 */
export function armPosterior(
  model: FitModel,
  series: SimSeries,
  p: PolicyName,
  g: GroupMeta,
  arm: number,
  e: number,
  cumSel: number,
): { mu: number; sigma: number; n: number } {
  if (model.grid.arms[arm].is_null) return { mu: 0, sigma: 0, n: cumSel };
  const r = groupArmReward(model, g.key, arm);
  const z = new Rng(seedOf('truth', g.key, arm)).normal();
  const shifted = series.shiftIdx >= 0 && e >= series.shiftIdx;
  const truth = r.mean * (1 + 0.12 * z) * (shifted ? 0.93 : 1);
  const nEff = 0.6 * cumSel;
  const kind = policyBase(p);
  const k0 = kind.startsWith('thompson') ? 25 : 8;
  // estimation error: a persistent per-arm draw plus a small per-epoch wobble, both shrinking with n
  const persistent = new Rng(seedOf('pnoise', g.key, arm, p)).normal();
  const wobble = new Rng(seedOf('pwobble', g.key, arm, e, p)).normal();
  const noise = (0.9 * persistent + 0.25 * wobble) * (r.std / Math.sqrt(nEff + 4));
  const mu = truth + (r.mean - truth) * (k0 / (k0 + nEff)) + noise;
  const sd = r.std * (0.95 + 0.1 * (k0 / (k0 + nEff)));
  // thompson_se samples from N(μ, σ²/n): the posterior scale is the standard error of the mean
  return { mu, sigma: kind === 'thompson_se' ? sd / Math.sqrt(Math.max(1, nEff)) : sd, n: nEff };
}
