/**
 * In-memory state of the mock API: datasets, fits, simulations, reviews, jobs. Seeded
 * with a realistic research history on module load; resources created from the UI
 * progress in wall-clock time so polling, progress bars and the breaker workflow are
 * visible. Everything resets on page reload.
 */
import type {
  ApiErrorBody,
  DatasetCreate,
  FitConfig,
  JobKind,
  JobTarget,
  PortfolioRequest,
  RecoveryInfo,
  ReviewAction,
  SegmentationDimension,
  SimulationConfig,
  SimulationStatus,
} from '@/api/types';
import { DEFAULT_FIT_CONFIG, DEFAULT_PSI } from '@/lib/domain';
import { epochRange } from '@/lib/epoch';
import { buildIndex, type LoanIndex } from './loans';
import { mctmCore, type MctmCore } from './mctm';
import { buildFitModel, type FitModel } from './riskModel';
import type { ReviewDecision } from './sentinel';
import { simSeries, type SimSeries } from './simulation';
import type { LifecycleEvent } from './simViews';

export class MockHttpError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody;
  constructor(status: number, body: ApiErrorBody) {
    super(typeof body.detail === 'string' ? body.detail : `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

export const iso = (ms: number) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');

export interface JobRec {
  id: number;
  kind: JobKind;
  target: JobTarget | null;
  createdMs: number;
  startMs: number;
  durationMs: number;
  forced: { status: 'failed' | 'cancelled'; atMs: number; message: string } | null;
  pid: number;
  lines: string[];
}

export interface DatasetRec {
  id: number;
  name: string;
  params: DatasetCreate;
  nominalN: number;
  createdMs: number;
  jobId: number;
}

export interface FitRec {
  id: number;
  name: string;
  datasetId: number;
  config: FitConfig;
  createdMs: number;
  jobId: number;
  fitSeconds: number;
}

export interface SimRec {
  id: number;
  name: string;
  fitId: number;
  config: SimulationConfig;
  createdMs: number;
  jobId: number;
  status: SimulationStatus;
  startMs: number;
  msPerEpoch: number;
  markMs: number;
  markDone: number;
  done: number;
  epochTs: number[];
  startedMs: number | null;
  finishedMs: number | null;
  lifecycle: LifecycleEvent[];
  decision: ReviewDecision | null;
  reviewId: number | null;
  restarts: number;
  lastRecovery: (RecoveryInfo & { killedIdx: number; resumedMs: number | null }) | null;
  pid: number;
  workerAlive: boolean;
}

export interface ReviewRec {
  id: number;
  simId: number;
  tripEpoch: string;
  createdMs: number;
  status: 'pending' | 'acknowledged';
  action: ReviewAction | null;
  note: string;
  ackMs: number | null;
  ackBy: string | null;
  auto: boolean;
  segBefore: SegmentationDimension[];
  segAfter: SegmentationDimension[] | null;
}

export interface PortfolioRec {
  jobId: number;
  fitId: number;
  request: PortfolioRequest;
}

const T0 = Date.now();
const HOUR = 3_600_000;

export const db = {
  jobs: new Map<number, JobRec>(),
  datasets: new Map<number, DatasetRec>(),
  fits: new Map<number, FitRec>(),
  sims: new Map<number, SimRec>(),
  reviews: new Map<number, ReviewRec>(),
  portfolios: new Map<number, PortfolioRec>(),
  next: { job: 101, dataset: 1, fit: 1, sim: 1, review: 1, pid: 41207 },
};

const JOB_LINES: Record<Exclude<JobKind, 'simulation'>, string[]> = {
  ingest: [
    'resolving source and schema',
    'projecting the 60 retained Table 3 columns (data minimisation, F.4)',
    'parsing percent strings and Mon-YYYY dates; policy_code == 1',
    'deriving y_* outcomes (snapshot 2019-03-31, charge-off lag 4 months)',
    'assigning y_group (grade|purpose_group|home_ownership) and y_historical_arm',
    'writing parquet cache',
    'dataset ready',
  ],
  fit: [
    'loading LoanTable and applying the temporal split',
    'segmenting borrowers into groups',
    'stage 1: hierarchical Beta-Bernoulli (global → group → leaf), κ shrinkage',
    'stage 2: Beta method-of-moments recovery with leaf → group → global fallback',
    'reconstructing the monthly loan-state panel',
    'MCTM: Dirichlet posterior rows, absorption analysis',
    'holdout evaluation: Brier, ROC-AUC, ECE, recovery MAE/RMSE, transition log-likelihood',
    'fit ready',
  ],
  shadow_portfolio: [
    'sampling borrowers from the holdout window',
    'evaluating every arm per borrower (posterior predictive)',
    'classifying rescued / upsized / repriced / declined',
    'aggregating expected profit and inclusion by grade',
    'portfolio summary ready',
  ],
};

export function newJob(kind: JobKind, target: JobTarget | null, startMs: number, durationMs: number): JobRec {
  const id = db.next.job++;
  const rec: JobRec = {
    id,
    kind,
    target,
    createdMs: startMs - 400,
    startMs,
    durationMs,
    forced: null,
    pid: db.next.pid++,
    lines: kind === 'simulation' ? [] : JOB_LINES[kind],
  };
  db.jobs.set(id, rec);
  return rec;
}

// ─────────────────────────────────────────────── lazily built heavy objects
const indexCache = new Map<number, LoanIndex>();
const modelCache = new Map<number, FitModel>();
const mctmCache = new Map<string, MctmCore>();
const seriesCache = new Map<string, SimSeries>();

export function datasetIndex(ds: DatasetRec): LoanIndex {
  let idx = indexCache.get(ds.id);
  if (!idx) {
    idx = buildIndex(ds.params, ds.nominalN);
    indexCache.set(ds.id, idx);
  }
  return idx;
}

export function fitModel(fit: FitRec): FitModel {
  let m = modelCache.get(fit.id);
  if (!m) {
    const ds = db.datasets.get(fit.datasetId);
    if (!ds) throw new MockHttpError(404, { detail: 'dataset not found' });
    m = buildFitModel(datasetIndex(ds), fit.config);
    modelCache.set(fit.id, m);
  }
  return m;
}

export function fitMctm(fit: FitRec, structure = fit.config.mctm.structure): MctmCore {
  const key = `${fit.id}|${structure}`;
  let c = mctmCache.get(key);
  if (!c) {
    const ds = db.datasets.get(fit.datasetId);
    c = mctmCore(`fit${fit.id}`, structure, fit.config.mctm.prior_strength, (ds?.nominalN ?? 200000) / 200000);
    mctmCache.set(key, c);
  }
  return c;
}

export function shiftEpochFor(fit: FitRec): string | null {
  const ds = db.datasets.get(fit.datasetId);
  return ds?.params.source === 'synthetic' ? ds.params.shift?.at ?? null : null;
}

export function simSeriesFor(sim: SimRec): SimSeries {
  const key = `${sim.id}|${JSON.stringify(sim.decision)}`;
  let s = seriesCache.get(key);
  if (!s) {
    const fit = db.fits.get(sim.fitId)!;
    const ds = db.datasets.get(fit.datasetId)!;
    const epochTs = (e: string) => {
      const k = epochRange(sim.config.start, sim.config.end).indexOf(e);
      return iso(sim.epochTs[k] ?? sim.createdMs);
    };
    s = simSeries(fitModel(fit), datasetIndex(ds), sim.config, `sim${sim.id}`, shiftEpochFor(fit), sim.decision, epochTs);
    seriesCache.set(key, s);
  }
  return s;
}

// ─────────────────────────────────────────────── simulation state machine
export function ensureReview(sim: SimRec, tripIdx: number, now: number): ReviewRec {
  if (sim.reviewId) return db.reviews.get(sim.reviewId)!;
  const fit = db.fits.get(sim.fitId)!;
  const epochs = epochRange(sim.config.start, sim.config.end);
  const rec: ReviewRec = {
    id: db.next.review++,
    simId: sim.id,
    tripEpoch: epochs[tripIdx],
    createdMs: sim.epochTs[tripIdx] ?? now,
    status: 'pending',
    action: null,
    note: '',
    ackMs: null,
    ackBy: null,
    auto: sim.config.breaker_mode === 'auto_ack',
    segBefore: fit.config.segmentation.dimensions,
    segAfter: null,
  };
  db.reviews.set(rec.id, rec);
  sim.reviewId = rec.id;
  return rec;
}

export function refine(dims: SegmentationDimension[]): SegmentationDimension[] {
  const order: SegmentationDimension[] = ['fico_band', 'dti_band', 'income_band', 'emp_band', 'verification_status', 'application_type', 'sub_grade'];
  const next = order.find((d) => !dims.includes(d));
  return next ? [...dims, next] : dims;
}

/** Advance a simulation to `now`. */
export function tick(sim: SimRec, now: number): void {
  if (sim.status === 'queued') {
    if (now < sim.startMs) return;
    sim.status = 'running';
    sim.startedMs = sim.startMs;
    sim.markMs = sim.startMs;
    sim.markDone = 0;
  }
  if (sim.status !== 'running') return;
  const epochs = epochRange(sim.config.start, sim.config.end);
  const total = epochs.length;
  let done = Math.min(total, sim.markDone + Math.max(0, Math.floor((now - sim.markMs) / sim.msPerEpoch)));
  const trip = simSeriesFor(sim).sentinel.tripIndex;
  if (trip >= 0 && sim.config.breaker_mode === 'pause' && !sim.decision?.action && done > trip) {
    done = trip + 1;
    for (let k = sim.epochTs.length; k < done; k++) sim.epochTs.push(sim.markMs + (k - sim.markDone + 1) * sim.msPerEpoch);
    sim.done = done;
    sim.status = 'awaiting_review';
    sim.decision = { tripEpoch: epochs[trip], action: null, ackEpoch: null };
    const review = ensureReview(sim, trip, now);
    sim.lifecycle.push({ type: 'RunPaused', afterEpochs: done, ts: sim.epochTs[done - 1] + 300, level: 'warning', message: `paused: awaiting human review #${review.id} (breaker_mode = pause)`, payload: { review: review.id } });
    return;
  }
  for (let k = sim.epochTs.length; k < done; k++) sim.epochTs.push(sim.markMs + (k - sim.markDone + 1) * sim.msPerEpoch);
  sim.done = done;
  if (trip >= 0 && sim.config.breaker_mode === 'auto_ack' && done > trip) {
    const review = ensureReview(sim, trip, now);
    const ackIdx = sim.decision?.ackEpoch ? epochs.indexOf(sim.decision.ackEpoch) : -1;
    if (review.status === 'pending' && ackIdx >= 0 && done > ackIdx) {
      review.status = 'acknowledged';
      review.action = sim.decision?.action ?? 'resume';
      review.ackMs = sim.epochTs[ackIdx] ?? now;
      review.ackBy = 'auto_ack';
      review.note = `auto-acknowledged after ${sim.config.psi.auto_ack_epochs} epoch(s)`;
      review.segAfter = review.action === 'resegment' ? refine(review.segBefore) : null;
    }
  }
  const rec = sim.lastRecovery;
  if (rec && rec.deterministic_match === null && done > rec.killedIdx) {
    rec.deterministic_match = true;
    rec.recovery_seconds = Math.round(((sim.epochTs[rec.killedIdx] ?? now) - Date.parse(rec.killed_at)) / 100) / 10;
  }
  if (done >= total) {
    sim.status = 'completed';
    sim.finishedMs = sim.epochTs[total - 1] ?? now;
  }
}

/** Decide the auto-ack review action up front (independent of wall-clock time). */
export function presetDecision(sim: SimRec): void {
  if (sim.config.breaker_mode !== 'auto_ack') return;
  const trip = simSeriesFor(sim).sentinel.tripIndex;
  if (trip < 0) return;
  const epochs = epochRange(sim.config.start, sim.config.end);
  sim.decision = {
    tripEpoch: epochs[trip],
    action: sim.config.psi.auto_action,
    ackEpoch: epochs[Math.min(trip + sim.config.psi.auto_ack_epochs, epochs.length - 1)],
  };
}

// ─────────────────────────────────────────────── seed history
function seed(): void {
  // Oldest first so ids increase with time; the default selection (newest ready dataset →
  // newest ready fit → completed simulation) lands on the synthetic twin's research history.
  const lc: DatasetRec = {
    id: db.next.dataset++,
    name: 'Lending Club 2012–2018 · 200k sample',
    params: {
      name: 'Lending Club 2012–2018 · 200k sample',
      source: 'lending_club',
      path: 'data/raw/accepted_2007_to_2018Q4.csv.gz',
      n: null,
      seed: 7,
      max_rows: 200000,
      sample_fraction: null,
      issue_from: '2012-01',
      issue_to: '2018-12',
      shift: null,
    },
    nominalN: 200000,
    createdMs: T0 - 74 * HOUR,
    jobId: 0,
  };
  const syn: DatasetRec = {
    id: db.next.dataset++,
    name: 'Synthetic twin · 200k · shift 2016-06',
    params: {
      name: 'Synthetic twin · 200k · shift 2016-06',
      source: 'synthetic',
      path: null,
      n: 200000,
      seed: 42,
      max_rows: null,
      sample_fraction: null,
      issue_from: '2012-01',
      issue_to: '2018-12',
      shift: { at: '2016-06', deltas: { fico_range_low: -25, dti: 5 } },
    },
    nominalN: 200000,
    createdMs: T0 - 70 * HOUR,
    jobId: 0,
  };
  for (const ds of [lc, syn]) {
    ds.jobId = newJob('ingest', { type: 'dataset', id: ds.id }, ds.createdMs + 500, 41_000).id;
    db.datasets.set(ds.id, ds);
  }
  const fits: [string, number, FitConfig, number, number][] = [
    ['LC baseline', lc.id, DEFAULT_FIT_CONFIG, 60, 36.7],
    [
      'extended MCTM · κ auto',
      syn.id,
      { ...DEFAULT_FIT_CONFIG, hierarchy: { ...DEFAULT_FIT_CONFIG.hierarchy, kappa_group: 'auto', kappa_leaf: 'auto' }, mctm: { ...DEFAULT_FIT_CONFIG.mctm, structure: 'extended' } },
      55,
      41.9,
    ],
    ['baseline · grade × purpose × home', syn.id, DEFAULT_FIT_CONFIG, 50, 38.4],
  ];
  let baselineFit = 0;
  for (const [name, datasetId, config, hoursAgo, secs] of fits) {
    const id = db.next.fit++;
    const createdMs = T0 - hoursAgo * HOUR;
    db.fits.set(id, { id, name, datasetId, config, createdMs, jobId: newJob('fit', { type: 'fit', id }, createdMs + 500, secs * 1000).id, fitSeconds: secs });
    baselineFit = id;
  }

  const simBase = (id: number, name: string, config: SimulationConfig, createdMs: number, msPerEpoch: number): SimRec => ({
    id,
    name,
    fitId: baselineFit,
    config,
    createdMs,
    jobId: newJob('simulation', { type: 'simulation', id }, createdMs + 4000, 0).id,
    status: 'running',
    startMs: createdMs + 4000,
    msPerEpoch,
    markMs: createdMs + 4000,
    markDone: 0,
    done: 0,
    epochTs: [],
    startedMs: createdMs + 4000,
    finishedMs: null,
    lifecycle: [],
    decision: null,
    reviewId: null,
    restarts: 0,
    lastRecovery: null,
    pid: db.next.pid++,
    workerAlive: true,
  });

  // Sim 1 — pause mode: tripped at 2016-06 and still waiting for a human decision
  const s1 = simBase(
    db.next.sim++,
    'Governance drill · breaker pause · hybrid feedback',
    {
      policies: ['thompson', 'ucb1', 'historical'],
      feedback_mode: 'hybrid',
      start: '2015-01',
      end: '2018-12',
      max_loans: 30000,
      seed: 7,
      interim_months: 12,
      interim_weight: 0.5,
      psi: DEFAULT_PSI,
      breaker_mode: 'pause',
      checkpoint_every: 6,
      hindsight_sample: 'elapsed',
    },
    T0 - 47 * HOUR,
    9800,
  );
  db.sims.set(s1.id, s1);
  tick(s1, T0);

  // Sim 2 — completed; auto-acknowledged resegment at the trip, one chaos kill + checkpoint recovery
  const s2 = simBase(
    db.next.sim++,
    'Thompson vs baselines · simulated feedback',
    {
      policies: ['thompson', 'ucb1', 'epsilon_greedy', 'historical', 'random'],
      feedback_mode: 'simulated',
      start: '2015-01',
      end: '2018-12',
      max_loans: 50000,
      seed: 42,
      interim_months: 12,
      interim_weight: 0.5,
      psi: { ...DEFAULT_PSI, auto_action: 'resegment' },
      breaker_mode: 'auto_ack',
      checkpoint_every: 6,
      hindsight_sample: 'elapsed',
    },
    T0 - 46 * HOUR,
    9600,
  );
  db.sims.set(s2.id, s2);
  presetDecision(s2);
  const killIdx = 27;
  const ckpt = 24;
  const downtime = 41_000;
  for (let k = 0; k < 48; k++) {
    const replay = k >= killIdx ? downtime + (killIdx - ckpt) * s2.msPerEpoch : 0;
    s2.epochTs.push(s2.startMs + (k + 1) * s2.msPerEpoch + replay);
  }
  const killedMs = s2.epochTs[killIdx - 1] + 3200;
  s2.lifecycle.push(
    { type: 'WorkerKilled', afterEpochs: killIdx, ts: killedMs, level: 'critical', message: `worker pid ${s2.pid} killed (SIGKILL, chaos experiment)`, payload: { signal: 'SIGKILL', pid: s2.pid } },
    {
      type: 'RunResumed',
      afterEpochs: killIdx,
      ts: killedMs + downtime,
      level: 'warning',
      message: `resumed from checkpoint 2016-12 on pid ${s2.pid + 1}; replaying ${killIdx - ckpt} epochs`,
      payload: { from_checkpoint: '2016-12', replayed_epochs: killIdx - ckpt },
    },
  );
  s2.restarts = 1;
  s2.pid += 1;
  s2.lastRecovery = {
    killed_at: iso(killedMs),
    killed_at_epoch: '2017-04',
    resumed_at: iso(killedMs + downtime),
    resumed_from_epoch: '2016-12',
    replayed_epochs: killIdx - ckpt,
    deterministic_match: true,
    recovery_seconds: Math.round((downtime + (killIdx - ckpt) * s2.msPerEpoch) / 100) / 10,
    killedIdx: killIdx,
    resumedMs: killedMs + downtime,
  };
  s2.markDone = 48;
  s2.done = 48;
  s2.status = 'completed';
  s2.finishedMs = s2.epochTs[47];
  const trip2 = simSeriesFor(s2).sentinel.tripIndex;
  if (trip2 >= 0) {
    const r = ensureReview(s2, trip2, T0);
    r.status = 'acknowledged';
    r.action = 'resegment';
    r.ackMs = s2.epochTs[trip2 + 1];
    r.ackBy = 'auto_ack';
    r.note = 'auto-acknowledged after 1 epoch(s)';
    r.segAfter = refine(r.segBefore);
  }

  // One finished portfolio evaluation on the baseline fit
  const pj = newJob('shadow_portfolio', { type: 'fit', id: baselineFit }, T0 - 20 * HOUR, 52_000);
  db.portfolios.set(pj.id, { jobId: pj.id, fitId: baselineFit, request: { max_loans: 5000, seed: 7, filters: { resolved_only: false } } });
}

seed();
