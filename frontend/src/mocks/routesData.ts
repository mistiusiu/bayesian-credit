/** Mock handlers: health, jobs, datasets, fits, shadow engine. */
import type {
  DatasetCreate,
  FitConfig,
  Health,
  HierarchyConfig,
  PolicyEffectConfig,
  PortfolioRequest,
  PortfolioResult,
  PortfolioRun,
  SegmentationDimension,
} from '@/api/types';
import { DEFAULT_FIT_CONFIG, GUARDRAIL_INFO, POLICY_EFFECT_FEATURES, SEGMENTATION_DIMENSIONS } from '@/lib/domain';
import { datasetSummary, nearestDefaultArm, variableRows } from './datasetStats';
import { calibration, fitArms, hierarchyRows, riskEstimate } from './fitViews';
import { filterLoans, indexOfLoanId, loanRecord } from './loans';
import { mctmPayload } from './mctm';
import { paginate, renderDataset, renderFit, renderJob, jobState } from './render';
import { portfolioResult } from './portfolio';
import { completeContext, evaluateShadow, isPreDecisionKey } from './shadow';
import { type DatasetRec, type FitRec, datasetIndex, db, fitMctm, fitModel, newJob } from './store';
import { Errors, badRequest, body, conflict, epochOk, isObj, notFound, num, str } from './validate';
import type { Route } from './router';

const list = (q: URLSearchParams, key: string) =>
  q
    .get(key)
    ?.split(',')
    .map((s) => s.trim())
    .filter(Boolean);

function readyDataset(id: number, now: number): DatasetRec {
  const ds = db.datasets.get(id) ?? notFound('dataset');
  const st = renderDataset(ds, now).status;
  if (st !== 'ready') conflict(`dataset ${id} is not ready (status ${st})`);
  return ds;
}

export function readyFit(id: number, now: number): FitRec {
  const fit = db.fits.get(id) ?? notFound('fit');
  const st = jobState(db.jobs.get(fit.jobId)!, now).status;
  if (st !== 'succeeded') conflict(`fit ${id} is not ready (status ${st === 'queued' || st === 'running' ? st : 'failed'})`);
  return fit;
}

function createDataset(raw: unknown, now: number) {
  const b = body(raw);
  const e = new Errors();
  const name = str(b.name)?.trim() ?? '';
  e.check(name.length > 0, 'name', 'This field is required.');
  const source = b.source;
  e.check(source === 'synthetic' || source === 'lending_club', 'source', 'Must be "synthetic" or "lending_club".');
  const n = num(b.n) ?? 200000;
  if (source === 'synthetic') e.check(n >= 1000 && n <= 2_500_000, 'n', 'Must be between 1,000 and 2,500,000.');
  const maxRows = b.max_rows === null || b.max_rows === undefined ? null : num(b.max_rows);
  if (source === 'lending_club' && maxRows !== null) e.check(maxRows >= 1000, 'max_rows', 'Must be at least 1,000.');
  const frac = b.sample_fraction === null || b.sample_fraction === undefined ? null : num(b.sample_fraction);
  if (frac !== null) e.check(frac > 0 && frac <= 1, 'sample_fraction', 'Must be in (0, 1].');
  e.check(epochOk(b.issue_from), 'issue_from', 'Use YYYY-MM.');
  e.check(epochOk(b.issue_to), 'issue_to', 'Use YYYY-MM.');
  const from = (str(b.issue_from) ?? '2012-01');
  const to = (str(b.issue_to) ?? '2018-12');
  e.check(from <= to, 'issue_to', 'Must not precede issue_from.');
  e.check(from >= '2007-06' && to <= '2018-12', 'issue_from', 'Lending Club issue dates span 2007-06 to 2018-12.');
  let shift: DatasetCreate['shift'] = null;
  if (isObj(b.shift) && source === 'synthetic') {
    const at = str(b.shift.at);
    e.check(epochOk(at) && !!at && at > from && at <= to, 'shift', 'shift.at must be a YYYY-MM inside the issue range.');
    const deltas: Record<string, number> = {};
    if (isObj(b.shift.deltas)) for (const [k, v] of Object.entries(b.shift.deltas)) if (num(v) !== null) deltas[k] = v as number;
    shift = { at: at ?? '2016-06', deltas };
  }
  e.throwIfAny();
  const nominal = source === 'synthetic' ? n : Math.min(2_500_000, Math.round((maxRows ?? 2_260_701) * (frac ?? 1)));
  const id = db.next.dataset++;
  const params: DatasetCreate = {
    name,
    source: source as DatasetCreate['source'],
    path: source === 'lending_club' ? str(b.path) ?? 'data/raw/accepted_2007_to_2018Q4.csv.gz' : null,
    n: source === 'synthetic' ? n : null,
    seed: num(b.seed) ?? 42,
    max_rows: source === 'lending_club' ? maxRows : null,
    sample_fraction: frac,
    issue_from: from,
    issue_to: to,
    shift,
  };
  const job = newJob('ingest', { type: 'dataset', id }, now + 500, 4200 + Math.min(4000, nominal / 100));
  const rec: DatasetRec = { id, name, params, nominalN: nominal, createdMs: now, jobId: job.id };
  db.datasets.set(id, rec);
  return renderDataset(rec, now);
}

const GUARDRAIL_KEYS = GUARDRAIL_INFO.map((g) => g.key);

function createFit(raw: unknown, now: number) {
  const b = body(raw);
  const e = new Errors();
  const dsId = num(b.dataset);
  e.check(dsId !== null, 'dataset', 'This field is required.');
  const name = str(b.name)?.trim() ?? '';
  e.check(name.length > 0, 'name', 'This field is required.');
  e.throwIfAny();
  const ds = readyDataset(dsId!, now);
  const d = DEFAULT_FIT_CONFIG;
  const seg = isObj(b.segmentation) && Array.isArray(b.segmentation.dimensions) ? (b.segmentation.dimensions as SegmentationDimension[]) : d.segmentation.dimensions;
  const valid = new Set(SEGMENTATION_DIMENSIONS.map((x) => x.key));
  e.check(seg.length > 0 && seg.every((x) => valid.has(x)) && new Set(seg).size === seg.length, 'segmentation', 'Choose one or more distinct supported dimensions.');
  const arm = { ...d.arm_space, ...(isObj(b.arm_space) ? b.arm_space : {}) };
  e.check(Array.isArray(arm.amounts) && arm.amounts.length > 0 && arm.amounts.every((x) => num(x) !== null && x > 0), 'arm_space', 'amounts must be positive numbers.');
  e.check(Array.isArray(arm.rates) && arm.rates.length > 0 && arm.rates.every((x) => num(x) !== null && x > 0 && x < 1), 'arm_space', 'rates are fractions in (0, 1), e.g. 0.13.');
  e.check(Array.isArray(arm.terms) && arm.terms.length > 0 && arm.terms.every((x) => Number.isInteger(x) && x > 0), 'arm_space', 'terms must be positive integers (months).');
  for (const k of GUARDRAIL_KEYS) {
    const v = arm[k];
    e.check(v === null || ((num(v) ?? 0) > 0 && v <= 10), `arm_space.${k}`, 'Must be a positive fraction, or null to switch the guardrail off.');
  }
  const hierIn = isObj(b.hierarchy) ? b.hierarchy : {};
  e.check(!('policy_effect' in hierIn) || typeof hierIn.policy_effect === 'boolean', 'hierarchy.policy_effect', 'Must be true or false.');
  e.check(!('policy_effect_cfg' in hierIn) || isObj(hierIn.policy_effect_cfg), 'hierarchy.policy_effect_cfg', 'Must be an object (partial PolicyEffectConfig).');
  const pe = { ...d.hierarchy.policy_effect_cfg, ...(isObj(hierIn.policy_effect_cfg) ? (hierIn.policy_effect_cfg as Partial<PolicyEffectConfig>) : {}) };
  const hier: HierarchyConfig = { ...d.hierarchy, ...(hierIn as Partial<HierarchyConfig>), policy_effect_cfg: pe };
  for (const k of ['kappa_group', 'kappa_leaf'] as const) e.check(hier[k] === 'auto' || (num(hier[k]) ?? 0) > 0, `hierarchy.${k}`, 'Must be > 0 or "auto".');
  e.check((num(hier.alpha0) ?? 0) > 0 && (num(hier.beta0) ?? 0) > 0, 'hierarchy', 'alpha0 and beta0 must be > 0.');
  const featureKeys = POLICY_EFFECT_FEATURES.map((f) => f.key as string);
  e.check(Array.isArray(pe.features) && pe.features.length > 0 && pe.features.every((f) => featureKeys.includes(f)) && new Set(pe.features).size === pe.features.length, 'hierarchy.policy_effect_cfg', `features: one or more of ${featureKeys.join(', ')}.`);
  e.check((num(pe.ridge) ?? -1) >= 0 && (num(pe.max_abs_logit_shift) ?? 0) > 0 && (num(pe.pti_cap) ?? 0) > 0, 'hierarchy.policy_effect_cfg', 'ridge ≥ 0, max_abs_logit_shift > 0 and pti_cap > 0.');
  e.check([pe.min_rows, pe.max_iter].every((x) => Number.isInteger(x) && x >= 1) && Number.isInteger(pe.min_elapsed_rows) && pe.min_elapsed_rows >= 0, 'hierarchy.policy_effect_cfg', 'min_rows and max_iter are integers ≥ 1; min_elapsed_rows an integer ≥ 0.');
  const recovery = { ...d.recovery, ...(isObj(b.recovery) ? b.recovery : {}) };
  e.check(Number.isInteger(recovery.min_obs) && recovery.min_obs >= 1, 'recovery.min_obs', 'Must be an integer ≥ 1.');
  e.check(typeof recovery.term_level === 'boolean', 'recovery.term_level', 'Must be true or false.');
  const mctm = { ...d.mctm, ...(isObj(b.mctm) ? b.mctm : {}) };
  e.check(mctm.structure === 'proposal' || mctm.structure === 'extended', 'mctm.structure', 'Must be "proposal" or "extended".');
  e.check((num(mctm.prior_strength) ?? -1) >= 0, 'mctm.prior_strength', 'Must be ≥ 0.');
  const reward = { ...d.reward, ...(isObj(b.reward) ? b.reward : {}) };
  e.check((num(reward.funding_rate) ?? -1) >= 0 && reward.funding_rate <= 0.5, 'reward.funding_rate', 'Annual fraction in [0, 0.5].');
  e.check((num(reward.risk_aversion) ?? -1) >= 0, 'reward.risk_aversion', 'Must be ≥ 0.');
  e.check((num(reward.lendable_threshold) ?? -1) >= 0 && reward.lendable_threshold <= 1, 'reward.lendable_threshold', 'Must be in [0, 1].');
  e.check(reward.exposure_basis === 'full' || reward.exposure_basis === 'outstanding', 'reward.exposure_basis', 'Must be "full" or "outstanding".');
  e.check(reward.prepayment_adjustment === 'empirical' || reward.prepayment_adjustment === null, 'reward.prepayment_adjustment', 'Must be "empirical" or null.');
  const trainEnd = str(b.train_end) ?? d.train_end;
  const range = ds.params;
  e.check(epochOk(trainEnd) && trainEnd > (range.issue_from ?? '2012-01') && trainEnd <= (range.issue_to ?? '2018-12'), 'train_end', 'Must be a YYYY-MM inside the dataset issue range.');
  e.throwIfAny();
  const id = db.next.fit++;
  const config: FitConfig = { segmentation: { dimensions: seg }, arm_space: arm, hierarchy: hier, recovery, mctm, reward, train_end: trainEnd };
  const job = newJob('fit', { type: 'fit', id }, now + 500, 6500);
  const rec: FitRec = { id, name, datasetId: ds.id, config, createdMs: now, jobId: job.id, fitSeconds: 33 + (id % 7) * 1.7 };
  db.fits.set(id, rec);
  return renderFit(rec, now);
}

function shadowEvaluate(fitId: number, raw: unknown, now: number) {
  const fit = readyFit(fitId, now);
  const b = body(raw);
  const model = fitModel(fit);
  const ds = db.datasets.get(fit.datasetId)!;
  if (b.loan_id !== undefined) {
    const idx = datasetIndex(ds);
    const i = indexOfLoanId(idx, Number(b.loan_id));
    if (i === null) notFound(`loan ${JSON.stringify(b.loan_id)}`);
    const rec = loanRecord(idx, i, nearestDefaultArm);
    return evaluateShadow(model, fit.id, {
      loanId: rec.loan_id,
      issue: rec.issue_d,
      context: rec.context,
      policy: { amount: rec.policy.loan_amnt, rate: rec.policy.int_rate, term: rec.policy.term },
      outcome: rec.outcome,
    });
  }
  if (!isObj(b.context)) badRequest('non_field_errors', 'Provide either loan_id or context.');
  const bad = Object.keys(b.context).filter((k) => !isPreDecisionKey(k));
  if (bad.length) badRequest('context', `LeakageError: columns not available at stage 'pre_decision': ${bad.join(', ')}`);
  const hp = isObj(b.historical_policy) ? b.historical_policy : null;
  return evaluateShadow(model, fit.id, {
    loanId: null,
    issue: null,
    context: completeContext(b.context),
    policy: hp ? { amount: Number(hp.amount), rate: Number(hp.rate), term: Number(hp.term) } : null,
    outcome: null,
  });
}

const portfolioCache = new Map<number, Omit<PortfolioResult, 'job'>>();

function portfolioFor(jobId: number, now: number): PortfolioResult {
  const rec = db.portfolios.get(jobId) ?? notFound('portfolio job');
  const job = db.jobs.get(jobId)!;
  const rendered = renderJob(job, now);
  if (rendered.status !== 'succeeded') return { job: rendered, fit: rec.fitId, request: rec.request, summary: null, by_grade: [], inclusion: [], top_loans: [] };
  let res = portfolioCache.get(jobId);
  if (!res) {
    const fit = db.fits.get(rec.fitId)!;
    res = portfolioResult(fitModel(fit), datasetIndex(db.datasets.get(fit.datasetId)!), fit.id, rec.request);
    portfolioCache.set(jobId, res);
  }
  return { job: rendered, ...res };
}

/** The fit's most recent succeeded portfolio evaluation (fed to the report's counterfactual block). */
export function latestPortfolio(fitId: number, now: number): Omit<PortfolioResult, 'job'> | null {
  const runs = [...db.portfolios.values()].filter((p) => p.fitId === fitId).sort((a, b) => b.jobId - a.jobId);
  for (const run of runs) {
    const res = portfolioFor(run.jobId, now);
    if (res.summary) return res;
  }
  return null;
}

function createPortfolio(fitId: number, raw: unknown, now: number) {
  readyFit(fitId, now);
  const b = body(raw);
  const e = new Errors();
  const max = num(b.max_loans) ?? 5000;
  e.check(max >= 100 && max <= 50000, 'max_loans', 'Must be between 100 and 50,000.');
  e.throwIfAny();
  const request: PortfolioRequest = { max_loans: max, seed: num(b.seed) ?? 42, filters: isObj(b.filters) ? (b.filters) : {} };
  const job = newJob('shadow_portfolio', { type: 'fit', id: fitId }, now + 400, 4500);
  db.portfolios.set(job.id, { jobId: job.id, fitId, request });
  return renderJob(job, now);
}

export const dataRoutes: Route[] = [
  ['GET', /^\/health\/$/, (): Health => ({ status: 'ok', version: '0.1.0-mock', python: '3.14.0', core_version: '0.1.0' })],
  [
    'GET',
    /^\/jobs\/$/,
    (_m, q, _b, now) => {
      const statuses = list(q, 'status');
      const kinds = list(q, 'kind');
      const jobs = [...db.jobs.values()]
        .map((j) => renderJob(j, now))
        .filter((j) => (!statuses || statuses.includes(j.status)) && (!kinds || kinds.includes(j.kind)))
        .sort((a, b) => b.id - a.id);
      return paginate('/jobs/', jobs, q);
    },
  ],
  ['GET', /^\/jobs\/(\d+)\/$/, (m, _q, _b, now) => renderJob(db.jobs.get(Number(m[1])) ?? notFound('job'), now)],
  [
    'GET',
    /^\/datasets\/$/,
    (_m, q, _b, now) => paginate('/datasets/', [...db.datasets.values()].sort((a, b) => b.id - a.id).map((d) => renderDataset(d, now)), q),
  ],
  ['POST', /^\/datasets\/$/, (_m, _q, b, now) => createDataset(b, now), 201],
  ['GET', /^\/datasets\/(\d+)\/$/, (m, _q, _b, now) => renderDataset(db.datasets.get(Number(m[1])) ?? notFound('dataset'), now)],
  ['GET', /^\/datasets\/(\d+)\/variables\/$/, (m, _q, _b, now) => {
    const ds = readyDataset(Number(m[1]), now);
    return variableRows(datasetIndex(ds), ds.params.source);
  }],
  ['GET', /^\/datasets\/(\d+)\/summary\/$/, (m, q, _b, now) => {
    const ds = readyDataset(Number(m[1]), now);
    return datasetSummary(ds.id, datasetIndex(ds), list(q, 'features') ?? null);
  }],
  ['GET', /^\/datasets\/(\d+)\/loans\/$/, (m, q, _b, now) => {
    const ds = readyDataset(Number(m[1]), now);
    const idx = datasetIndex(ds);
    const resolved = q.get('resolved');
    const hits = filterLoans(idx, {
      grade: list(q, 'grade'),
      purpose: list(q, 'purpose'),
      home_ownership: list(q, 'home_ownership'),
      resolved: resolved === 'true' ? true : resolved === 'false' ? false : undefined,
      search: q.get('search') ?? undefined,
    });
    const page = paginate(`/datasets/${ds.id}/loans/`, hits, q);
    return { ...page, results: page.results.map((i) => loanRecord(idx, i, nearestDefaultArm)) };
  }],
  [
    'GET',
    /^\/fits\/$/,
    (_m, q, _b, now) => {
      const dsId = q.get('dataset');
      const fits = [...db.fits.values()].filter((f) => !dsId || f.datasetId === Number(dsId)).sort((a, b) => b.id - a.id);
      return paginate('/fits/', fits.map((f) => renderFit(f, now)), q);
    },
  ],
  ['POST', /^\/fits\/$/, (_m, _q, b, now) => createFit(b, now), 201],
  ['GET', /^\/fits\/(\d+)\/$/, (m, _q, _b, now) => renderFit(db.fits.get(Number(m[1])) ?? notFound('fit'), now)],
  ['GET', /^\/fits\/(\d+)\/groups\/$/, (m, q, _b, now) => hierarchyRows(fitModel(readyFit(Number(m[1]), now)), q.get('level'), q.get('group'))],
  ['GET', /^\/fits\/(\d+)\/risk\/$/, (m, q, _b, now) => {
    const fit = readyFit(Number(m[1]), now);
    const group = q.get('group') || badRequest('group', 'This query parameter is required.');
    const arm = q.get('arm');
    return riskEstimate(fitModel(fit), fit.id, group, arm === null || arm === '' ? null : Number(arm));
  }],
  ['GET', /^\/fits\/(\d+)\/calibration\/$/, (m, _q, _b, now) => {
    const fit = readyFit(Number(m[1]), now);
    return calibration(fitModel(fit), datasetIndex(db.datasets.get(fit.datasetId)!), fit.id);
  }],
  ['GET', /^\/fits\/(\d+)\/mctm\/$/, (m, q, _b, now) => {
    const fit = readyFit(Number(m[1]), now);
    const structure = q.get('structure') === 'extended' ? 'extended' : q.get('structure') === 'proposal' ? 'proposal' : fit.config.mctm.structure;
    const horizon = Math.min(120, Math.max(1, Number(q.get('horizon') ?? 60) || 60));
    const start = q.get('start') === 'delinquent' ? 'delinquent' : 'current';
    return mctmPayload(fit.id, fitMctm(fit, structure), structure, horizon, start, fit.config.mctm.prior_strength);
  }],
  ['GET', /^\/fits\/(\d+)\/arms\/$/, (m, _q, _b, now) => {
    const fit = readyFit(Number(m[1]), now);
    return fitArms(fitModel(fit), fit.id);
  }],
  ['POST', /^\/fits\/(\d+)\/shadow\/evaluate\/$/, (m, _q, b, now) => shadowEvaluate(Number(m[1]), b, now)],
  ['POST', /^\/fits\/(\d+)\/shadow\/portfolio\/$/, (m, _q, b, now) => createPortfolio(Number(m[1]), b, now), 202],
  ['GET', /^\/fits\/(\d+)\/shadow\/portfolio\/$/, (m, _q, _b, now): PortfolioRun[] => {
    const fitId = Number(m[1]);
    return [...db.portfolios.values()]
      .filter((p) => p.fitId === fitId)
      .sort((a, b) => b.jobId - a.jobId)
      .map((p) => {
        const r = portfolioFor(p.jobId, now);
        return { job: r.job, request: r.request, summary: r.summary };
      });
  }],
  ['GET', /^\/fits\/(\d+)\/shadow\/portfolio\/(\d+)\/$/, (m, _q, _b, now) => {
    const res = portfolioFor(Number(m[2]), now);
    if (res.fit !== Number(m[1])) notFound('portfolio job');
    return res;
  }],
];
