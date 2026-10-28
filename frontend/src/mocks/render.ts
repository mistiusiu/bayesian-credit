/** Store records → API contract shapes. */
import type { Dataset, Fit, Job, JobStatus, Paginated, Review, Simulation } from '@/api/types';
import { policyBase } from '@/lib/domain';
import { epochRange } from '@/lib/epoch';
import { fmt } from '@/lib/format';
import { datasetFacts } from './datasetStats';
import { fitMetrics } from './fitViews';
import { type DatasetRec, type FitRec, type JobRec, type ReviewRec, type SimRec, datasetIndex, db, fitMctm, fitModel, iso, simSeriesFor, tick } from './store';
import { headline } from './simViews';
import { simView } from './views';

export function jobState(j: JobRec, now: number): { status: JobStatus; progress: number } {
  if (j.forced) return { status: j.forced.status, progress: Math.min(1, Math.max(0, (j.forced.atMs - j.startMs) / j.durationMs)) };
  if (now < j.startMs) return { status: 'queued', progress: 0 };
  const p = j.durationMs <= 0 ? 1 : Math.min(1, (now - j.startMs) / j.durationMs);
  return { status: p < 1 ? 'running' : 'succeeded', progress: p };
}

export function renderJob(j: JobRec, now: number): Job {
  if (j.kind === 'simulation' && j.target) {
    const sim = db.sims.get(j.target.id);
    if (sim) return renderSimJob(j, sim, now);
  }
  const { status, progress } = jobState(j, now);
  const shown = Math.max(1, Math.ceil(progress * j.lines.length));
  const lines = status === 'queued' ? ['queued: waiting for a worker slot'] : j.lines.slice(0, shown);
  const ts = (ms: number) => `[${iso(ms).slice(11, 19)}]`;
  const tail = lines.map((l, k) => `${ts(j.startMs + (k / Math.max(j.lines.length, 1)) * j.durationMs)} ${l}`);
  if (j.forced) tail.push(`${ts(j.forced.atMs)} ${j.forced.message}`);
  return {
    id: j.id,
    kind: j.kind,
    status,
    progress: Math.round(progress * 1000) / 1000,
    message: j.forced?.message ?? lines[lines.length - 1] ?? '',
    log_tail: tail,
    created_at: iso(j.createdMs),
    started_at: now >= j.startMs ? iso(j.startMs) : null,
    finished_at: status === 'succeeded' ? iso(j.startMs + j.durationMs) : j.forced ? iso(j.forced.atMs) : null,
    target: j.target,
    pid: status === 'running' ? j.pid : null,
  };
}

function renderSimJob(j: JobRec, sim: SimRec, now: number): Job {
  tick(sim, now);
  const epochs = epochRange(sim.config.start, sim.config.end);
  const status: JobStatus =
    sim.status === 'queued' ? 'queued' : sim.status === 'completed' ? 'succeeded' : sim.status === 'cancelled' ? 'cancelled' : sim.status === 'interrupted' || sim.status === 'failed' ? 'failed' : 'running';
  const tail = sim.epochTs.slice(-8).map((t, k, arr) => {
    const e = sim.epochTs.length - arr.length + k;
    return `[${iso(t).slice(11, 19)}] epoch ${epochs[e]} complete`;
  });
  const message =
    sim.status === 'awaiting_review'
      ? `paused at ${epochs[sim.done - 1]}: awaiting human review`
      : sim.status === 'interrupted'
        ? 'worker killed (SIGKILL); resumable from checkpoint'
        : sim.status === 'paused'
          ? `paused at ${epochs[sim.done - 1] ?? 'start'}`
          : sim.status === 'completed'
            ? 'simulation complete'
            : `epoch ${epochs[Math.min(sim.done, epochs.length - 1)]} running`;
  return {
    id: j.id,
    kind: 'simulation',
    status,
    progress: Math.round((sim.done / epochs.length) * 1000) / 1000,
    message,
    log_tail: tail,
    created_at: iso(j.createdMs),
    started_at: sim.startedMs ? iso(sim.startedMs) : null,
    finished_at: sim.finishedMs ? iso(sim.finishedMs) : null,
    target: j.target,
    pid: sim.workerAlive && status === 'running' ? sim.pid : null,
  };
}

const buildStatus = (s: JobStatus): Dataset['status'] => (s === 'succeeded' ? 'ready' : s);

export function renderDataset(d: DatasetRec, now: number): Dataset {
  const job = db.jobs.get(d.jobId)!;
  const status = buildStatus(jobState(job, now).status);
  const ready = status === 'ready';
  const f = ready ? datasetFacts(datasetIndex(d)) : null;
  return {
    id: d.id,
    name: d.name,
    source: d.params.source,
    status,
    params: d.params,
    n_loans: f?.n ?? null,
    n_resolved: f?.nResolved ?? null,
    issue_min: f?.issueMin ?? null,
    issue_max: f?.issueMax ?? null,
    resolved_share: f ? Math.round((f.nResolved / f.n) * 10000) / 10000 : null,
    default_rate: f ? Math.round((f.nDefault / f.nResolved) * 10000) / 10000 : null,
    status_counts: f?.statusCounts ?? {},
    created_at: iso(d.createdMs),
    job: renderJob(job, now),
  };
}

export function renderFit(f: FitRec, now: number): Fit {
  const job = db.jobs.get(f.jobId)!;
  const status = buildStatus(jobState(job, now).status);
  const ds = db.datasets.get(f.datasetId);
  let metrics = null;
  let nGroups = null;
  if (status === 'ready' && ds) {
    const model = fitModel(f);
    const core = fitMctm(f);
    metrics = fitMetrics(model, datasetIndex(ds), { ll: core.ll, n: core.n, rowErr: core.rowErr }, f.fitSeconds);
    nGroups = model.groups.size;
  }
  const nArms = f.config.arm_space.amounts.length * f.config.arm_space.rates.length * f.config.arm_space.terms.length + (f.config.arm_space.include_null ? 1 : 0);
  return {
    id: f.id,
    name: f.name,
    dataset: f.datasetId,
    dataset_name: ds?.name ?? '',
    status,
    config: f.config,
    metrics,
    n_groups: nGroups,
    n_arms: nArms,
    created_at: iso(f.createdMs),
    job: renderJob(job, now),
  };
}

export function renderSim(s: SimRec, now: number): Simulation {
  tick(s, now);
  const fit = db.fits.get(s.fitId)!;
  const total = epochRange(s.config.start, s.config.end).length;
  const series = simSeriesFor(s);
  const last = s.done - 1;
  const v = simView(s);
  const h = headline(v);
  const hist = h.find((p) => policyBase(p.policy) === 'historical');
  const adaptive = h.filter((p) => p.policy !== 'historical' && p.policy !== 'random');
  const best = adaptive.sort((a, b) => b.cum_reward - a.cum_reward)[0];
  const every = s.config.checkpoint_every;
  const ckpt = Math.floor(s.done / every) * every;
  const pending = [...db.reviews.values()].filter((r) => r.simId === s.id && r.status === 'pending').length;
  const rec = s.lastRecovery;
  return {
    id: s.id,
    name: s.name,
    fit: s.fitId,
    fit_name: fit.name,
    dataset: fit.datasetId,
    status: s.status,
    progress: Math.round((s.done / total) * 1000) / 1000,
    config: s.config,
    epochs_total: total,
    epochs_done: s.done,
    current_epoch: last >= 0 ? series.epochs[last] : null,
    breaker_state: last >= 0 ? series.sentinel.breaker[last] : 'closed',
    sentinel_level: last >= 0 ? series.sentinel.level[last] : 'normal',
    psi_aggregate: last >= 0 ? series.sentinel.aggregate[last] : null,
    pending_reviews: pending,
    last_checkpoint_epoch: ckpt > 0 ? series.epochs[ckpt - 1] : null,
    restarts: s.restarts,
    worker: {
      pid: s.workerAlive && ['running', 'paused', 'awaiting_review'].includes(s.status) ? s.pid : null,
      alive: s.workerAlive && ['running', 'paused', 'awaiting_review'].includes(s.status),
      last_heartbeat: s.status === 'running' ? iso(now - 900) : s.epochTs.length ? iso(s.epochTs[s.epochTs.length - 1]) : null,
    },
    last_recovery: rec
      ? {
          killed_at: rec.killed_at,
          killed_at_epoch: rec.killed_at_epoch,
          resumed_at: rec.resumed_at,
          resumed_from_epoch: rec.resumed_from_epoch,
          replayed_epochs: rec.replayed_epochs,
          deterministic_match: rec.deterministic_match,
          recovery_seconds: rec.recovery_seconds,
        }
      : null,
    headline: h,
    uplift_vs_historical: best && hist ? Math.round((best.cum_reward - hist.cum_reward) * 100) / 100 : null,
    created_at: iso(s.createdMs),
    started_at: s.startedMs ? iso(s.startedMs) : null,
    finished_at: s.finishedMs ? iso(s.finishedMs) : null,
    job: renderJob(db.jobs.get(s.jobId)!, now),
  };
}

export function renderReview(r: ReviewRec): Review {
  const sim = db.sims.get(r.simId)!;
  const series = simSeriesFor(sim);
  const k = series.epochs.indexOf(r.tripEpoch);
  const tops = series.sentinel.features
    .map((feature, i) => ({ feature, psi: series.sentinel.psi[i][k] }))
    .sort((a, b) => b.psi - a.psi)
    .slice(0, 3);
  const agg = series.sentinel.aggregate[k];
  return {
    id: r.id,
    simulation: r.simId,
    simulation_name: sim.name,
    epoch: r.tripEpoch,
    status: r.status,
    created_at: iso(r.createdMs),
    psi_aggregate: agg,
    level: series.sentinel.level[k],
    breaker_state: r.status === 'pending' ? 'open' : 'half_open',
    reason: `aggregate PSI ${fmt.num(agg, 3)} exceeded the breaker threshold ${sim.config.psi.thresholds.breaker} (${tops[0]?.feature ?? ''} ${fmt.num(tops[0]?.psi, 3)})`,
    top_features: tops,
    action: r.action,
    note: r.note,
    acknowledged_at: r.ackMs ? iso(r.ackMs) : null,
    acknowledged_by: r.ackBy,
    auto: r.auto,
    segmentation_before: r.segBefore,
    segmentation_after: r.segAfter,
  };
}

export function paginate<T>(path: string, items: T[], q: URLSearchParams): Paginated<T> {
  const limit = Math.min(500, Math.max(1, Number(q.get('limit') ?? 50) || 50));
  const offset = Math.max(0, Number(q.get('offset') ?? 0) || 0);
  const page = items.slice(offset, offset + limit);
  const link = (o: number) => {
    const p = new URLSearchParams(q);
    p.set('limit', String(limit));
    p.set('offset', String(o));
    return `/api/v1${path}?${p.toString()}`;
  };
  return {
    count: items.length,
    next: offset + limit < items.length ? link(offset + limit) : null,
    previous: offset > 0 ? link(Math.max(0, offset - limit)) : null,
    results: page,
  };
}
