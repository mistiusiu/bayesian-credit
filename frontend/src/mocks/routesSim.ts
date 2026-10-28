/** Mock handlers: simulations, controls, sentinel reviews. */
import type { PolicyName, PsiSettings, SimulationConfig } from '@/api/types';
import { DEFAULT_HINDSIGHT, DEFAULT_PSI, PSI_FEATURES } from '@/lib/domain';
import { epochRange } from '@/lib/epoch';
import { paginate, renderReview, renderSim } from './render';
import { latestPortfolio, readyFit } from './routesData';
import { banditDetails, counterfactualDetails } from './reportDetails';
import type { Route } from './router';
import { bucketTable } from './sentinel';
import { borrowerEvents, events, headline, PER_BORROWER, posteriorHistory, sentinelTimeline, simArms, simpson, timeseries } from './simViews';
import { buildReport } from './report';
import { fitMetrics } from './fitViews';
import { type SimRec, datasetIndex, db, fitMctm, fitModel, iso, newJob, presetDecision, refine, simSeriesFor, tick } from './store';
import { Errors, badRequest, body, conflict, epochOk, isObj, notFound, num, str } from './validate';
import { simView } from './views';

const MS_PER_EPOCH = 380;

/** Core policy kinds (simulation/policies.py POLICY_KINDS). */
const POLICY_KINDS = ['thompson', 'thompson_se', 'thompson_flat', 'epsilon_greedy', 'ucb1', 'greedy', 'random', 'historical'];
const BANDIT_FIELDS = ['epsilon', 'ucb_c', 'posterior_scale', 'warm_start', 'interim_weight', 'prior_mu', 'prior_sigma', 'min_sigma'];

/** `kind` or `kind(field=value, …)` → the normalised label (whitespace removed, lower-cased), or null. */
function parsePolicySpec(spec: string): string | null {
  const label = spec.replace(/\s+/g, '').toLowerCase();
  const m = /^([a-z0-9_]+)(?:\((.*)\))?$/.exec(label);
  if (!m || !POLICY_KINDS.includes(m[1])) return null;
  const args = (m[2] ?? '').split(',').filter(Boolean);
  return args.every((a) => a.includes('=') && BANDIT_FIELDS.includes(a.split('=')[0])) ? label : null;
}

function getSim(id: number, now: number): SimRec {
  const sim = db.sims.get(id) ?? notFound('simulation');
  tick(sim, now);
  return sim;
}

function requireEpochs(sim: SimRec) {
  if (sim.done === 0) conflict('no completed epochs yet');
}

function policyParam(sim: SimRec, q: URLSearchParams): PolicyName {
  const p: PolicyName | null = q.get('policy');
  if (p && !sim.config.policies.includes(p)) badRequest('policy', `policy must be one of ${sim.config.policies.join(', ')}`);
  return p ?? sim.config.policies[0];
}

function epochParam(sim: SimRec, q: URLSearchParams): number {
  const epochs = epochRange(sim.config.start, sim.config.end);
  const e = q.get('epoch');
  if (!e) return sim.done - 1;
  const k = epochs.indexOf(e);
  if (k < 0 || k >= sim.done) badRequest('epoch', 'epoch must be a completed epoch (YYYY-MM)');
  return k;
}

function createSimulation(raw: unknown, now: number) {
  const b = body(raw);
  const e = new Errors();
  const fitId = num(b.fit);
  e.check(fitId !== null, 'fit', 'This field is required.');
  const name = str(b.name)?.trim() ?? '';
  e.check(name.length > 0, 'name', 'This field is required.');
  const rawPolicies = Array.isArray(b.policies) ? b.policies : [];
  const policies = rawPolicies.map((p) => (typeof p === 'string' ? parsePolicySpec(p) : null));
  e.check(policies.length > 0 && policies.every((p) => p !== null), 'policies', `Each policy must be one of ${POLICY_KINDS.join(', ')}, optionally with BanditConfig overrides, e.g. ucb1(ucb_c=1.0).`);
  e.check(new Set(policies).size === policies.length, 'policies', 'duplicate policies in the run');
  e.throwIfAny();
  const fit = readyFit(fitId!, now);
  const ds = db.datasets.get(fit.datasetId)!;
  const start = str(b.start) ?? '2015-01';
  const end = str(b.end) ?? '2018-12';
  e.check(epochOk(start) && epochOk(end) && start <= end, 'start', 'start and end are YYYY-MM with start ≤ end.');
  e.check(start >= (ds.params.issue_from ?? '2012-01') && end <= (ds.params.issue_to ?? '2018-12'), 'end', 'Epochs must lie inside the dataset issue range.');
  const fm = b.feedback_mode ?? 'simulated';
  e.check(fm === 'simulated' || fm === 'replay' || fm === 'hybrid', 'feedback_mode', 'simulated | replay | hybrid');
  const bm = b.breaker_mode ?? 'auto_ack';
  e.check(bm === 'pause' || bm === 'auto_ack' || bm === 'off', 'breaker_mode', 'pause | auto_ack | off');
  const psiIn = isObj(b.psi) ? b.psi : {};
  const psi: PsiSettings = { ...DEFAULT_PSI, ...(psiIn as Partial<PsiSettings>), thresholds: { ...DEFAULT_PSI.thresholds, ...(isObj(psiIn.thresholds) ? psiIn.thresholds : {}) } };
  e.check(psi.features.length > 0 && psi.features.every((f) => PSI_FEATURES.includes(f) || f === 'emp_length' || f === 'verification_status'), 'psi.features', 'Unknown PSI feature.');
  e.check(psi.thresholds.watch > 0 && psi.thresholds.watch < psi.thresholds.breaker, 'psi.thresholds', 'Need 0 < watch < breaker.');
  e.check(Number.isInteger(psi.window_months) && psi.window_months >= 1, 'psi.window_months', 'Must be an integer ≥ 1.');
  const every = num(b.checkpoint_every) ?? 6;
  e.check(Number.isInteger(every) && every >= 1, 'checkpoint_every', 'Must be an integer ≥ 1.');
  const iw = num(b.interim_weight) ?? 0.5;
  e.check(iw >= 0 && iw <= 1, 'interim_weight', 'Must be in [0, 1].');
  const maxLoans = b.max_loans === null ? null : num(b.max_loans) ?? 50000;
  if (maxLoans !== null) e.check(maxLoans >= 100, 'max_loans', 'Must be ≥ 100 or null.');
  e.throwIfAny();
  const hs = b.hindsight_sample ?? DEFAULT_HINDSIGHT;
  e.check(hs === 'elapsed' || hs === 'resolved', 'hindsight_sample', '"elapsed" | "resolved"');
  e.throwIfAny();
  const config: SimulationConfig = {
    policies: policies as PolicyName[],
    feedback_mode: fm as SimulationConfig['feedback_mode'],
    start,
    end,
    max_loans: maxLoans,
    seed: num(b.seed) ?? 42,
    interim_months: num(b.interim_months) ?? 12,
    interim_weight: iw,
    psi,
    breaker_mode: bm as SimulationConfig['breaker_mode'],
    checkpoint_every: every,
    hindsight_sample: hs as SimulationConfig['hindsight_sample'],
  };
  const id = db.next.sim++;
  const sim: SimRec = {
    id,
    name,
    fitId: fit.id,
    config,
    createdMs: now,
    jobId: newJob('simulation', { type: 'simulation', id }, now + 700, 0).id,
    status: 'queued',
    startMs: now + 700,
    msPerEpoch: MS_PER_EPOCH,
    markMs: now + 700,
    markDone: 0,
    done: 0,
    epochTs: [],
    startedMs: null,
    finishedMs: null,
    lifecycle: [],
    decision: null,
    reviewId: null,
    restarts: 0,
    lastRecovery: null,
    pid: db.next.pid++,
    workerAlive: true,
  };
  db.sims.set(id, sim);
  presetDecision(sim);
  return renderSim(sim, now);
}

function control(id: number, action: string, now: number) {
  const sim = getSim(id, now);
  const epochs = epochRange(sim.config.start, sim.config.end);
  const at = sim.done;
  const last = epochs[Math.max(0, at - 1)];
  switch (action) {
    case 'pause':
      if (sim.status !== 'running') conflict(`cannot pause a ${sim.status} simulation`);
      sim.status = 'paused';
      sim.markDone = at;
      sim.lifecycle.push({ type: 'RunPaused', afterEpochs: at, ts: now, level: 'info', message: `paused by operator at ${last}`, payload: {} });
      break;
    case 'resume':
      if (sim.status === 'paused') {
        sim.status = 'running';
        sim.msPerEpoch = Math.min(sim.msPerEpoch, MS_PER_EPOCH);
        sim.markMs = now;
        sim.markDone = at;
        sim.lifecycle.push({ type: 'RunResumed', afterEpochs: at, ts: now, level: 'info', message: `resumed by operator at ${last}`, payload: {} });
      } else if (sim.status === 'interrupted') {
        const every = sim.config.checkpoint_every;
        const ckpt = Math.floor(at / every) * every;
        sim.done = ckpt;
        sim.markDone = ckpt;
        sim.markMs = now + 1500;
        sim.msPerEpoch = Math.min(sim.msPerEpoch, MS_PER_EPOCH);
        sim.epochTs = sim.epochTs.slice(0, ckpt);
        sim.status = 'running';
        sim.restarts += 1;
        sim.workerAlive = true;
        sim.pid = db.next.pid++;
        sim.jobId = newJob('simulation', { type: 'simulation', id: sim.id }, now, 0).id;
        if (sim.lastRecovery) {
          sim.lastRecovery.resumed_at = iso(now);
          sim.lastRecovery.resumedMs = now;
          sim.lastRecovery.resumed_from_epoch = ckpt > 0 ? epochs[ckpt - 1] : null;
          sim.lastRecovery.replayed_epochs = at - ckpt;
        }
        sim.lifecycle.push({
          type: 'RunResumed',
          afterEpochs: ckpt,
          ts: now,
          level: 'warning',
          message: `restarted on pid ${sim.pid} from checkpoint ${ckpt > 0 ? epochs[ckpt - 1] : 'start'}; replaying ${at - ckpt} epoch(s)`,
          payload: { from_checkpoint: ckpt > 0 ? epochs[ckpt - 1] : null, replayed_epochs: at - ckpt },
        });
      } else if (sim.status === 'awaiting_review') conflict('acknowledge the pending sentinel review to resume');
      else conflict(`cannot resume a ${sim.status} simulation`);
      break;
    case 'cancel':
      if (['completed', 'cancelled', 'failed'].includes(sim.status)) conflict(`simulation already ${sim.status}`);
      sim.status = 'cancelled';
      sim.finishedMs = now;
      sim.workerAlive = false;
      sim.lifecycle.push({ type: 'RunCancelled', afterEpochs: at, ts: now, level: 'warning', message: `cancelled by operator at ${last}`, payload: {} });
      break;
    case 'chaos':
      if (sim.status !== 'running') conflict('chaos needs a running worker');
      sim.status = 'interrupted';
      sim.workerAlive = false;
      sim.markDone = at;
      sim.lastRecovery = { killed_at: iso(now), killed_at_epoch: at > 0 ? epochs[at - 1] : null, resumed_at: null, resumed_from_epoch: null, replayed_epochs: 0, deterministic_match: null, recovery_seconds: null, killedIdx: Math.max(0, at - 1), resumedMs: null };
      sim.lifecycle.push({ type: 'WorkerKilled', afterEpochs: at, ts: now, level: 'critical', message: `worker pid ${sim.pid} killed (SIGKILL, chaos)`, payload: { signal: 'SIGKILL', pid: sim.pid } });
      break;
    default:
      notFound('action');
  }
  return renderSim(sim, now);
}

function acknowledge(id: number, raw: unknown, now: number) {
  const r = db.reviews.get(id) ?? notFound('review');
  if (r.status !== 'pending') conflict('review already acknowledged');
  const b = body(raw);
  const action = b.action;
  if (action !== 'resume' && action !== 'resegment' && action !== 'rollback') badRequest('action', 'resume | resegment | rollback');
  const sim = getSim(r.simId, now);
  const epochs = epochRange(sim.config.start, sim.config.end);
  const tripIdx = epochs.indexOf(r.tripEpoch);
  r.status = 'acknowledged';
  r.action = action;
  r.note = str(b.note) ?? '';
  r.ackMs = now;
  r.ackBy = 'researcher';
  r.segAfter = action === 'resegment' ? refine(r.segBefore) : null;
  sim.decision = { tripEpoch: r.tripEpoch, action, ackEpoch: epochs[Math.min(tripIdx + 1, epochs.length - 1)] };
  if (sim.status === 'awaiting_review') {
    sim.status = 'running';
    sim.msPerEpoch = Math.min(sim.msPerEpoch, MS_PER_EPOCH);
    sim.markMs = now;
    sim.markDone = sim.done;
    sim.lifecycle.push({ type: 'RunResumed', afterEpochs: sim.done, ts: now, level: 'info', message: `review #${r.id} acknowledged (${action}); resuming`, payload: { review: r.id, action } });
  }
  return renderReview(r);
}

function report(sim: SimRec, now: number) {
  requireEpochs(sim);
  const fit = db.fits.get(sim.fitId)!;
  const ds = db.datasets.get(fit.datasetId)!;
  const v = simView(sim);
  const core = fitMctm(fit);
  const metrics = fitMetrics(fitModel(fit), datasetIndex(ds), { ll: core.ll, n: core.n, rowErr: core.rowErr }, fit.fitSeconds);
  const h = headline(v);
  const decisions = h[0]?.n_decisions ?? 0;
  const wall = ((sim.finishedMs ?? now) - (sim.startedMs ?? sim.createdMs)) / 1000;
  const series = simSeriesFor(sim);
  const topShare: Partial<Record<PolicyName, number>> = {};
  for (const p of sim.config.policies) {
    const sh = [...(series.policies.get(p)?.shares[sim.done - 1] ?? [])].sort((a, b) => b - a);
    topShare[p] = Math.round(sh.slice(0, 5).reduce((a, b) => a + b, 0) * 10000) / 10000;
  }
  return buildReport({
    simulation: sim.id,
    fit: fit.id,
    dataset: ds.id,
    final: sim.status === 'completed',
    metrics,
    headline: h,
    timeseries: timeseries(v, null),
    sentinel: sentinelTimeline(v),
    runtime: {
      cpu_seconds: Math.round(wall * 0.87 * 10) / 10,
      cpu_utilization: 0.87,
      peak_memory_mb: 1180 + sim.config.policies.length * 96,
      wall_seconds: Math.round(wall * 10) / 10,
      epochs: sim.done,
      decisions,
      decisions_per_second: Math.round((decisions * sim.config.policies.length) / Math.max(wall, 1)),
      restarts: sim.restarts,
      checkpoints_written: Math.floor(sim.done / sim.config.checkpoint_every) + sim.restarts,
      failed_workloads: sim.restarts,
      recovery_seconds: sim.lastRecovery?.recovery_seconds ?? null,
    },
    fitConfig: fit.config,
    simConfig: sim.config,
    generatedAt: iso(now),
    topShare,
    details: { bandit_performance: banditDetails(v), counterfactual_evaluation: counterfactualDetails(latestPortfolio(fit.id, now), metrics) },
  });
}

export const simRoutes: Route[] = [
  [
    'GET',
    /^\/simulations\/$/,
    (_m, q, _b, now) => {
      const fit = q.get('fit');
      const sims = [...db.sims.values()].filter((s) => !fit || s.fitId === Number(fit)).sort((a, b) => b.id - a.id);
      return paginate('/simulations/', sims.map((s) => renderSim(s, now)), q);
    },
  ],
  ['POST', /^\/simulations\/$/, (_m, _q, b, now) => createSimulation(b, now), 201],
  ['GET', /^\/simulations\/(\d+)\/$/, (m, _q, _b, now) => renderSim(getSim(Number(m[1]), now), now)],
  ['GET', /^\/simulations\/(\d+)\/timeseries\/$/, (m, q, _b, now) => {
    const sim = getSim(Number(m[1]), now);
    const p: PolicyName | null = q.get('policy');
    return timeseries(simView(sim), p);
  }],
  ['GET', /^\/simulations\/(\d+)\/arms\/$/, (m, q, _b, now) => {
    const sim = getSim(Number(m[1]), now);
    requireEpochs(sim);
    return simArms(simView(sim), policyParam(sim, q), epochParam(sim, q), q.get('group') || null);
  }],
  ['GET', /^\/simulations\/(\d+)\/posteriors\/$/, (m, q, _b, now) => {
    const sim = getSim(Number(m[1]), now);
    requireEpochs(sim);
    const arms = q.get('arms')?.split(',').map(Number).filter(Number.isFinite) ?? null;
    return posteriorHistory(simView(sim), policyParam(sim, q), q.get('group') || null, arms);
  }],
  ['GET', /^\/simulations\/(\d+)\/simpson\/$/, (m, q, _b, now) => {
    const sim = getSim(Number(m[1]), now);
    requireEpochs(sim);
    return simpson(simView(sim), policyParam(sim, q), epochParam(sim, q));
  }],
  ['GET', /^\/simulations\/(\d+)\/events\/$/, (m, q, _b, now) => {
    const sim = getSim(Number(m[1]), now);
    const v = simView(sim);
    const types = q.get('type')?.split(',').filter(Boolean) ?? null;
    let all = events(v);
    if (types?.some((t) => PER_BORROWER.includes(t as (typeof PER_BORROWER)[number]))) all = [...all, ...borrowerEvents(v, all.length)];
    const lastSeq = all.length;
    let rows = all.filter((ev) => (types ? types.includes(ev.type) : !PER_BORROWER.includes(ev.type)));
    const after = Number(q.get('after') ?? 0) || 0;
    rows = rows.filter((ev) => ev.seq > after);
    const limit = Math.min(1000, Math.max(1, Number(q.get('limit') ?? 100) || 100));
    const desc = q.get('order') === 'desc';
    const ordered = desc ? [...rows].reverse() : rows;
    return { simulation: sim.id, results: ordered.slice(0, limit), last_seq: lastSeq, has_more: ordered.length > limit };
  }],
  ['GET', /^\/simulations\/(\d+)\/sentinel\/$/, (m, _q, _b, now) => sentinelTimeline(simView(getSim(Number(m[1]), now)))],
  ['GET', /^\/simulations\/(\d+)\/sentinel\/buckets\/$/, (m, q, _b, now) => {
    const sim = getSim(Number(m[1]), now);
    requireEpochs(sim);
    const series = simSeriesFor(sim);
    const feature = q.get('feature') ?? series.sentinel.features[0];
    const f = series.sentinel.features.indexOf(feature);
    if (f < 0) badRequest('feature', 'not a monitored feature');
    const e = epochParam(sim, q);
    const psi = series.sentinel.psi[f][e];
    const level = psi > sim.config.psi.thresholds.breaker ? 'breaker' : psi >= sim.config.psi.thresholds.watch ? 'watch' : 'normal';
    const win = series.n.slice(Math.max(0, e - sim.config.psi.window_months + 1), e + 1).reduce((a, b) => a + b, 0);
    return bucketTable(sim.id, feature, series.epochs[e], psi, level, simView(sim).nReference, win);
  }],
  ['GET', /^\/simulations\/(\d+)\/report\/$/, (m, _q, _b, now) => report(getSim(Number(m[1]), now), now)],
  ['POST', /^\/simulations\/(\d+)\/(pause|resume|cancel|chaos)\/$/, (m, _q, _b, now) => control(Number(m[1]), m[2], now)],
  [
    'GET',
    /^\/sentinel\/reviews\/$/,
    (_m, q, _b, now) => {
      const sim = q.get('simulation');
      const status = q.get('status');
      for (const s of db.sims.values()) tick(s, now);
      const rows = [...db.reviews.values()]
        .filter((r) => (!sim || r.simId === Number(sim)) && (!status || r.status === status))
        .sort((a, b) => b.id - a.id)
        .map(renderReview);
      return paginate('/sentinel/reviews/', rows, q);
    },
  ],
  ['POST', /^\/sentinel\/reviews\/(\d+)\/acknowledge\/$/, (m, _q, b, now) => acknowledge(Number(m[1]), b, now)],
];

