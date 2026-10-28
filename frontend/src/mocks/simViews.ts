/** Endpoint payloads for a simulation, sliced to the epochs completed so far. */
import type {
  EventLevel,
  PolicyHeadline,
  PolicyName,
  PosteriorHistory,
  SentinelTimeline,
  SimArms,
  SimpsonPanel,
  SimulationConfig,
  SimulationEvent,
  SimulationEventType,
  Timeseries,
  TimeseriesRow,
} from '@/api/types';
import { groupDims, policyBase } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { type FitModel, groupArmReward } from './riskModel';
import type { ReviewDecision } from './sentinel';
import { type GroupMeta, type SimSeries, armPosterior, groupShare } from './simulation';
import { round } from './stats';

export interface LifecycleEvent {
  type: SimulationEventType;
  /** Epochs completed when it happened. */
  afterEpochs: number;
  ts: number;
  level: EventLevel;
  message: string;
  payload: Record<string, unknown>;
}

export interface SimView {
  id: number;
  config: SimulationConfig;
  series: SimSeries;
  model: FitModel;
  done: number;
  epochTs: number[];
  startedMs: number;
  lifecycle: LifecycleEvent[];
  reviewId: number | null;
  decision: ReviewDecision | null;
  completed: boolean;
  nReference: number;
}

const iso = (ms: number) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');

export function timeseries(v: SimView, only: PolicyName | null): Timeseries {
  const s = v.series;
  const rows: TimeseriesRow[] = [];
  const policies = v.config.policies.filter((p) => !only || p === only);
  for (const p of policies) {
    const ps = s.policies.get(p);
    if (!ps) continue;
    let cr = 0;
    let cz = 0;
    let cg = 0;
    for (let e = 0; e < v.done; e++) {
      cr += ps.reward[e];
      cz += ps.realized[e];
      cg += ps.regret[e];
      let rs = 0;
      let rn = 0;
      for (let k = Math.max(0, e - 5); k <= e; k++) {
        rs += ps.reward[k];
        rn += s.n[k];
      }
      const sh = ps.shares[e];
      const lend = 1 - sh[0];
      let amt = 0;
      let rate = 0;
      v.model.grid.arms.forEach((a, k) => {
        if (a.is_null) return;
        amt += sh[k] * a.amount;
        rate += sh[k] * a.rate;
      });
      rows.push({
        epoch: s.epochs[e],
        policy: p,
        n_decisions: s.n[e],
        reward: round(ps.reward[e], 2),
        cum_reward: round(cr, 2),
        realized: round(ps.realized[e], 2),
        cum_realized: round(cz, 2),
        regret: round(ps.regret[e], 2),
        cum_regret: round(cg, 2),
        rolling_reward: round(rs / Math.max(rn, 1), 2),
        decline_rate: round(sh[0], 4),
        mean_amount: lend > 1e-9 ? round(amt / lend, 0) : null,
        mean_rate: lend > 1e-9 ? round(rate / lend, 4) : null,
        breaker_state: s.sentinel.breaker[e],
        psi_aggregate: s.sentinel.aggregate[e],
      });
    }
  }
  const order = new Map(v.config.policies.map((p, i) => [p, i]));
  rows.sort((a, b) => (a.epoch === b.epoch ? (order.get(a.policy) ?? 0) - (order.get(b.policy) ?? 0) : a.epoch.localeCompare(b.epoch)));
  return { simulation: v.id, policies, epochs: s.epochs.slice(0, v.done), rows };
}

export function headline(v: SimView): PolicyHeadline[] {
  if (v.done === 0) return [];
  const ts = timeseries(v, null);
  return v.config.policies.map((p) => {
    const rows = ts.rows.filter((r) => r.policy === p);
    const last = rows[rows.length - 1];
    const n = rows.reduce((a, r) => a + r.n_decisions, 0);
    const w = (k: 'decline_rate' | 'mean_amount' | 'mean_rate') =>
      rows.reduce((a, r) => a + (r[k] ?? 0) * r.n_decisions, 0) / Math.max(n, 1);
    return {
      policy: p,
      n_decisions: n,
      cum_reward: last?.cum_reward ?? 0,
      cum_realized: last?.cum_realized ?? 0,
      cum_regret: last?.cum_regret ?? 0,
      mean_reward: round((last?.cum_reward ?? 0) / Math.max(n, 1), 2),
      decline_rate: round(w('decline_rate'), 4),
      mean_amount: round(w('mean_amount'), 0),
      mean_rate: round(w('mean_rate'), 4),
    };
  });
}

function cumGroupSelections(v: SimView, p: PolicyName, g: GroupMeta, e: number, arm: number): number {
  let c = 0;
  for (let k = 0; k <= e; k++) c += v.series.n[k] * g.w * groupShare(v.series, p, k, g)[arm];
  return c;
}

export function simArms(v: SimView, p: PolicyName, e: number, groupKey: string | null): SimArms {
  const s = v.series;
  const ps = s.policies.get(p);
  const g = groupKey ? s.groups.find((x) => x.key === groupKey) ?? null : null;
  const cumN = s.n.slice(0, e + 1).reduce((a, b) => a + b, 0);
  const nE = g ? s.n[e] * g.w : s.n[e];
  const arms = v.model.grid.arms.map((a) => {
    const share = g ? groupShare(s, p, e, g)[a.arm] : ps?.shares[e][a.arm] ?? 0;
    let cumSel = 0;
    if (g) cumSel = cumGroupSelections(v, p, g, e, a.arm);
    else for (let k = 0; k <= e; k++) cumSel += s.n[k] * (ps?.shares[k][a.arm] ?? 0);
    let mu = 0;
    let sigma = 0;
    let n = 0;
    if (g) ({ mu, sigma, n } = armPosterior(v.model, s, p, g, a.arm, e, cumSel));
    else {
      for (const gg of s.groups) {
        const post = armPosterior(v.model, s, p, gg, a.arm, e, cumSel * gg.w);
        mu += gg.w * post.mu;
        sigma += gg.w * post.sigma;
        n += post.n;
      }
    }
    const pure = ['random', 'historical'].includes(policyBase(p));
    return {
      ...a,
      mu: pure && cumSel < 1 ? null : round(mu, 2),
      sigma: pure && cumSel < 1 ? null : round(sigma, 2),
      n: round(n, 1),
      selections: Math.round(share * nE),
      cum_selections: Math.round(cumSel),
      share: round(share, 5),
      cum_share: round(cumSel / Math.max(g ? cumN * g.w : cumN, 1), 5),
    };
  });
  return {
    simulation: v.id,
    policy: p,
    epoch: s.epochs[e],
    group: g?.key ?? null,
    n_decisions: Math.round(nE),
    amounts: v.model.grid.amounts,
    rates: v.model.grid.rates,
    terms: v.model.grid.terms,
    arms,
    groups: [...s.groups].sort((a, b) => b.w - a.w).map((x) => ({ group: x.key, n: Math.round(cumN * x.w) })),
  };
}

export function posteriorHistory(v: SimView, p: PolicyName, groupKey: string | null, armsWanted: number[] | null): PosteriorHistory {
  const s = v.series;
  const g = (groupKey ? s.groups.find((x) => x.key === groupKey) : undefined) ?? [...s.groups].sort((a, b) => b.w - a.w)[0];
  const last = Math.max(0, v.done - 1);
  const every = 3;
  let arms = armsWanted?.filter((a) => a > 0 && a < v.model.grid.arms.length) ?? [];
  if (!arms.length) {
    const sel = v.model.grid.arms.filter((a) => !a.is_null).map((a) => ({ arm: a.arm, c: cumGroupSelections(v, p, g, last, a.arm) }));
    arms = sel.sort((a, b) => b.c - a.c).slice(0, 4).map((x) => x.arm);
  }
  const epochsIdx: number[] = [];
  for (let e = every - 1; e <= last; e += every) epochsIdx.push(e);
  if (v.done > 0 && epochsIdx[epochsIdx.length - 1] !== last) epochsIdx.push(last);
  const rows = epochsIdx.flatMap((e) =>
    arms.map((arm) => {
      const post = armPosterior(v.model, s, p, g, arm, e, cumGroupSelections(v, p, g, e, arm));
      return { epoch: s.epochs[e], arm, mu: round(post.mu, 2), sigma: round(post.sigma, 2), n: round(post.n, 1) };
    }),
  );
  return { simulation: v.id, policy: p, group: g.key, snapshot_every: every, arms: arms.map((a) => v.model.grid.arms[a]), rows };
}

export function simpson(v: SimView, p: PolicyName, e: number): SimpsonPanel {
  const s = v.series;
  const lending = v.model.grid.arms.filter((a) => !a.is_null);
  const cumN = s.n.slice(0, e + 1).reduce((a, b) => a + b, 0);
  // only arms with evidence (≥ 3 effective observations) can be called "best": the max over
  // unobserved, noisy posteriors would mostly measure the winner's curse
  const table = s.groups.map((g) => {
    const mus = new Map<number, number>();
    for (const a of lending) {
      const z = 0.6 * cumN * g.w * groupShare(s, p, e, g)[a.arm];
      const post = armPosterior(v.model, s, p, g, a.arm, e, z);
      if (post.n >= 3) mus.set(a.arm, post.mu);
    }
    return { g, mus };
  });
  let pooledBest = lending[0].arm;
  let pooledVal = -Infinity;
  for (const a of lending) {
    if (!table.some((t) => t.mus.has(a.arm))) continue;
    const val = table.reduce((acc, t) => acc + t.g.w * (t.mus.get(a.arm) ?? groupArmReward(v.model, t.g.key, a.arm).mean), 0);
    if (val > pooledVal) {
      pooledVal = val;
      pooledBest = a.arm;
    }
  }
  let forfeited = 0;
  const rows = table.map(({ g, mus }) => {
    const pooledMu = mus.get(pooledBest) ?? groupArmReward(v.model, g.key, pooledBest).mean;
    let best = pooledBest;
    let bestMu = pooledMu;
    for (const [arm, mu] of mus) {
      if (mu > bestMu) {
        bestMu = mu;
        best = arm;
      }
    }
    forfeited += g.w * (bestMu - pooledMu);
    return {
      group: g.key,
      dims: groupDims(g.key, v.model.dims),
      n: Math.round(cumN * g.w),
      group_best_arm: best,
      group_best_label: v.model.grid.arms[best].label,
      group_best_mu: round(bestMu, 2),
      pooled_arm_mu: round(pooledMu, 2),
      gap: round(bestMu - pooledMu, 2),
      agrees: best === pooledBest,
    };
  });
  rows.sort((a, b) => b.n - a.n);
  return {
    simulation: v.id,
    policy: p,
    epoch: s.epochs[e],
    pooled_best_arm: pooledBest,
    pooled_best_label: v.model.grid.arms[pooledBest].label,
    pooled_best_mu: round(pooledVal, 2),
    n_groups: rows.length,
    n_agree: rows.filter((r) => r.agrees).length,
    forfeited_per_decision: round(forfeited, 2),
    groups: rows,
  };
}

export function sentinelTimeline(v: SimView): SentinelTimeline {
  const st = v.series.sentinel;
  const d = v.done;
  const features_psi = st.features.flatMap((f, k) =>
    st.epochs.slice(0, d).map((epoch, e) => ({ epoch, feature: f, psi: st.psi[k][e], level: levelFromValue(st.psi[k][e], v.config) })),
  );
  const aggregate = st.epochs.slice(0, d).map((epoch, e) => {
    const vals = st.psi.map((row) => row[e]);
    const top = st.features[vals.indexOf(Math.max(...vals))];
    return {
      epoch,
      psi_max: round(Math.max(...vals), 4),
      psi_mean: round(vals.reduce((a, b) => a + b, 0) / vals.length, 4),
      aggregate: st.aggregate[e],
      level: st.level[e],
      breaker_state: st.breaker[e],
      top_feature: top,
    };
  });
  const lastIdx = d - 1;
  const transitions = st.transitions
    .filter((t) => st.epochs.indexOf(t.epoch) <= lastIdx)
    .map((t) => ({ ...t, ts: iso(v.epochTs[st.epochs.indexOf(t.epoch)] ?? v.startedMs), review: v.reviewId }));
  const cur = aggregate[aggregate.length - 1];
  return {
    simulation: v.id,
    features: st.features,
    thresholds: v.config.psi.thresholds,
    aggregate_mode: v.config.psi.aggregate,
    window_months: v.config.psi.window_months,
    reference: { from: '2012-01', to: '2014-12', n: v.nReference, rebased_at: st.rebasedAt.filter((e) => st.epochs.indexOf(e) <= lastIdx) },
    features_psi,
    aggregate,
    transitions,
    current: {
      epoch: cur?.epoch ?? null,
      breaker_state: cur?.breaker_state ?? 'closed',
      level: cur?.level ?? 'normal',
      psi_aggregate: cur?.aggregate ?? null,
      top_feature: cur?.top_feature ?? null,
    },
    shift_counts: {
      watch: aggregate.filter((a) => a.level === 'watch').length,
      breaker: aggregate.filter((a) => a.level === 'breaker').length,
    },
  };
}

function levelFromValue(psi: number, cfg: SimulationConfig) {
  if (psi > cfg.psi.thresholds.breaker) return 'breaker' as const;
  if (psi >= cfg.psi.thresholds.watch) return 'watch' as const;
  return 'normal' as const;
}

const PER_BORROWER: SimulationEventType[] = ['BorrowerArrived', 'PolicySelected'];

export function events(v: SimView): SimulationEvent[] {
  const s = v.series;
  const out: Omit<SimulationEvent, 'seq'>[] = [];
  const push = (type: SimulationEventType, epoch: string | null, ts: number, level: EventLevel, message: string, payload: Record<string, unknown> = {}) =>
    out.push({ type, epoch, ts: iso(ts), level, message, payload });
  const life = (after: number) =>
    v.lifecycle.filter((l) => l.afterEpochs === after).forEach((l) => push(l.type, s.epochs[Math.max(0, after - 1)] ?? null, l.ts, l.level, l.message, l.payload));
  push('RunStarted', s.epochs[0], v.startedMs, 'info', `run started: ${v.config.policies.length} policies, ${s.epochs.length} epochs, ${v.config.feedback_mode} feedback`, {
    policies: v.config.policies,
    seed: v.config.seed,
  });
  life(0);
  const ts = v.config.policies[0];
  for (let e = 0; e < v.done; e++) {
    const epoch = s.epochs[e];
    const t = v.epochTs[e] ?? v.startedMs;
    const st = s.sentinel;
    const top = st.features[st.psi.map((r) => r[e]).indexOf(Math.max(...st.psi.map((r) => r[e])))];
    const lvl: EventLevel = st.level[e] === 'breaker' ? 'critical' : st.level[e] === 'watch' ? 'warning' : 'info';
    push('SentinelEvaluated', epoch, t - 900, lvl, `PSI ${st.aggregate[e].toFixed(3)} (${st.level[e]}), top feature ${top}`, {
      aggregate: st.aggregate[e],
      level: st.level[e],
      top_feature: top,
    });
    for (const tr of st.transitions.filter((x) => x.epoch === epoch)) {
      if (tr.to_state === 'open')
        push('BreakerTripped', epoch, t - 850, 'critical', `breaker open: ${tr.reason}; policies frozen, fallback ${v.config.psi.fallback.replace('_', ' ')}`, {
          review: v.reviewId,
          psi: tr.psi_aggregate,
        });
      else if (tr.to_state === 'half_open')
        push('ReviewAcknowledged', epoch, t - 800, 'warning', `review #${v.reviewId ?? '?'} acknowledged: ${tr.action ?? ''}; breaker half-open`, { action: tr.action, review: v.reviewId });
      else push('SentinelEvaluated', epoch, t - 790, 'info', `breaker closed: ${tr.reason}`, { to_state: 'closed' });
    }
    const terminal = Math.round(s.n[Math.max(0, e - 9)] * 0.18);
    const interim = Math.round(s.n.slice(Math.max(0, e - v.config.interim_months), e).reduce((a, b) => a + b, 0) * 0.92);
    push('FeedbackReleased', epoch, t - 600, 'info', `released ${fmt.int(interim)} interim and ${fmt.int(terminal)} terminal observations`, { interim, terminal });
    if (e % 3 === 2) {
      const g = [...s.groups].sort((a, b) => b.w - a.w)[0];
      const arm = g.best;
      const post = armPosterior(v.model, s, ts, g, arm, e, s.n.slice(0, e + 1).reduce((a, b) => a + b, 0) * g.w * 0.4);
      push('PosteriorUpdated', epoch, t - 400, 'info', `${ts} · ${fmt.group(g.key)} · ${v.model.grid.arms[arm].label}: μ ${fmt.usd(post.mu)}, σ ${fmt.usd(post.sigma)}`, {
        policy: ts,
        group: g.key,
        arm,
        mu: round(post.mu, 2),
        sigma: round(post.sigma, 2),
      });
    }
    const rewards: Record<string, number> = {};
    for (const p of v.config.policies) rewards[p] = round(s.policies.get(p)?.reward[e] ?? 0, 2);
    push('EpochCompleted', epoch, t - 50, 'info', `epoch ${epoch}: ${fmt.int(s.n[e])} decisions, ${ts} reward ${fmt.usdCompact(rewards[ts])}`, {
      n_decisions: s.n[e],
      reward: rewards,
      breaker_state: st.breaker[e],
    });
    if ((e + 1) % v.config.checkpoint_every === 0)
      push('CheckpointWritten', epoch, t, 'info', `checkpoint written at ${epoch} (policies, risk model, feedback queue, RNG state)`, { epoch, bytes: 4_200_000 + e * 61_000 });
    life(e + 1);
  }
  if (v.completed) push('RunCompleted', s.epochs[v.done - 1] ?? null, (v.epochTs[v.done - 1] ?? v.startedMs) + 200, 'info', 'run completed; evaluation report written', {});
  return out.map((ev, i) => ({ ...ev, seq: i + 1 }));
}

/** A few sampled per-borrower events per epoch (only when requested by type). */
export function borrowerEvents(v: SimView, base: number): SimulationEvent[] {
  const s = v.series;
  const out: SimulationEvent[] = [];
  let seq = base;
  for (let e = Math.max(0, v.done - 4); e < v.done; e++) {
    const g = [...s.groups].sort((a, b) => b.w - a.w)[e % 5];
    for (let k = 0; k < 3; k++) {
      const loanId = 2_000_000 + e * 1000 + k;
      const t = iso((v.epochTs[e] ?? v.startedMs) - 700 + k * 10);
      out.push({ seq: ++seq, type: 'BorrowerArrived', epoch: s.epochs[e], ts: t, level: 'info', message: `borrower ${loanId} · ${fmt.group(g.key)}`, payload: { loan_id: loanId, group: g.key } });
      for (const p of v.config.policies) {
        const arm = policyBase(p) === 'random' ? 1 + ((e * 7 + k * 13) % (v.model.grid.arms.length - 1)) : g.best;
        out.push({ seq: ++seq, type: 'PolicySelected', epoch: s.epochs[e], ts: t, level: 'info', message: `${p} → ${v.model.grid.arms[arm].label}`, payload: { loan_id: loanId, policy: p, arm } });
      }
    }
  }
  return out;
}

export { PER_BORROWER };
