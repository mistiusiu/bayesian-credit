/**
 * Typed client for the Bayesian Credit API: one fetch wrapper, one function per endpoint
 * (`api`), stable query keys (`qk`) and a TanStack Query hook per endpoint. Running
 * resources are polled with `refetchInterval`; refetches keep the previous frame
 * (`placeholderData: keepPreviousData`) so charts never flash.
 *
 * With VITE_API_MOCK=1 requests are answered by the in-browser mock (src/mocks), which
 * is code-split and only loaded in that mode.
 */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type {
  AcknowledgeRequest,
  ApiErrorBody,
  BuildStatus,
  Calibration,
  Dataset,
  DatasetCreate,
  DatasetSummary,
  EvaluationReport,
  EventPage,
  Fit,
  FitArms,
  FitCreate,
  HierarchyRow,
  Health,
  Id,
  Job,
  LoanRecord,
  Mctm,
  Paginated,
  PolicyName,
  PortfolioRequest,
  PortfolioResult,
  PortfolioRun,
  PosteriorHistory,
  PsiBucketTable,
  Review,
  RiskEstimate,
  SentinelTimeline,
  ShadowEvaluateRequest,
  ShadowReport,
  SimArms,
  SimpsonPanel,
  Simulation,
  SimulationCreate,
  Timeseries,
  VariableRow,
} from './types';

export const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1';
export const MOCK_MODE = import.meta.env.VITE_API_MOCK === '1';
/** Polling cadence for running jobs / simulations. */
export const POLL_MS = 1500;

// ───────────────────────────────────────────────────────────── fetch wrapper
export class ApiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody | null;

  constructor(status: number, body: ApiErrorBody | null) {
    super(ApiError.describe(status, body));
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }

  /** DRF field errors ({ field: [messages] }), excluding `detail` / `code`. */
  get fieldErrors(): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    if (!this.body) return out;
    for (const [k, v] of Object.entries(this.body)) {
      if (k === 'detail' || k === 'code') continue;
      if (Array.isArray(v)) out[k] = v.map(String);
      else if (typeof v === 'string') out[k] = [v];
    }
    return out;
  }

  private static describe(status: number, body: ApiErrorBody | null): string {
    if (body && typeof body.detail === 'string') return body.detail;
    if (body) {
      const first = Object.entries(body).find(([k]) => k !== 'code');
      if (first) return `${first[0]}: ${Array.isArray(first[1]) ? first[1].join(' ') : String(first[1])}`;
    }
    return status === 0 ? 'Network error: the API is unreachable' : `HTTP ${status}`;
  }
}

type QueryValue = string | number | boolean | null | undefined | readonly (string | number)[];
export type QueryParams = Record<string, QueryValue>;

export function toSearch(params?: QueryParams): URLSearchParams {
  const s = new URLSearchParams();
  if (!params) return s;
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === '') continue;
    if (Array.isArray(v)) {
      if (v.length) s.set(k, v.join(','));
    } else s.set(k, String(v));
  }
  return s;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

export async function request<T>(
  method: 'GET' | 'POST',
  path: string,
  opts: { query?: QueryParams; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const search = toSearch(opts.query);
  if (MOCK_MODE) {
    const { mockFetch } = await import('@/mocks');
    const res = await mockFetch(method, path, search, opts.body);
    if (res.status >= 400) throw new ApiError(res.status, isObject(res.body) ? (res.body) : null);
    return res.body as T;
  }
  const qs = search.toString();
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}${qs ? `?${qs}` : ''}`, {
      method,
      headers: { Accept: 'application/json', ...(opts.body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: opts.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError(0, null);
  }
  const text = await res.text();
  const isJson = (res.headers.get('content-type') ?? '').includes('json');
  let data: unknown = null;
  if (text && isJson) {
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      data = null;
    }
  }
  if (!res.ok) {
    // HTML error pages (proxy errors, another app on the port) are summarised, never shown raw
    const hint = isJson ? '' : ' (non-JSON response: is this the Bayesian Credit API?)';
    throw new ApiError(res.status, isObject(data) ? data : { detail: `HTTP ${res.status} ${res.statusText} from ${path}${hint}` });
  }
  if (text && !isJson) throw new ApiError(res.status, { detail: `Expected JSON from ${path}, got ${res.headers.get('content-type') ?? 'no content type'}` });
  return data as T;
}

/** Accept both the DRF envelope and a bare array (defensive against unpaginated list views). */
export function asPage<T>(data: Paginated<T> | T[]): Paginated<T> {
  return Array.isArray(data) ? { count: data.length, next: null, previous: null, results: data } : data;
}

const get = <T>(path: string, query?: QueryParams, signal?: AbortSignal) => request<T>('GET', path, { query, signal });
const post = <T>(path: string, body?: unknown) => request<T>('POST', path, { body: body ?? {} });
const page = async <T>(path: string, query?: QueryParams, signal?: AbortSignal) => asPage(await get<Paginated<T> | T[]>(path, query, signal));

// ───────────────────────────────────────────────────────────── endpoints
export type LoanQuery = {
  limit?: number;
  offset?: number;
  grade?: string[];
  purpose?: string[];
  home_ownership?: string[];
  resolved?: boolean | null;
  search?: string;
};
export type ControlAction = 'pause' | 'resume' | 'cancel' | 'chaos';
export type EventQuery = { type?: string[]; after?: number; limit?: number; order?: 'asc' | 'desc' };

export const api = {
  health: () => get<Health>('/health/'),
  jobs: {
    list: (q: { status?: string[]; kind?: string[] } = {}) => page<Job>('/jobs/', q),
    get: (id: Id) => get<Job>(`/jobs/${id}/`),
  },
  datasets: {
    list: () => page<Dataset>('/datasets/', { limit: 200 }),
    get: (id: Id) => get<Dataset>(`/datasets/${id}/`),
    create: (body: DatasetCreate) => post<Dataset>('/datasets/', body),
    variables: (id: Id) => get<VariableRow[]>(`/datasets/${id}/variables/`),
    summary: (id: Id, features?: string[]) => get<DatasetSummary>(`/datasets/${id}/summary/`, { features }),
    loans: (id: Id, q: LoanQuery, signal?: AbortSignal) => page<LoanRecord>(`/datasets/${id}/loans/`, { ...q }, signal),
  },
  fits: {
    list: (dataset?: Id | null) => page<Fit>('/fits/', { dataset, limit: 200 }),
    get: (id: Id) => get<Fit>(`/fits/${id}/`),
    create: (body: FitCreate) => post<Fit>('/fits/', body),
    groups: (id: Id, q: { level?: string; group?: string } = {}) => get<HierarchyRow[]>(`/fits/${id}/groups/`, q),
    risk: (id: Id, group: string, arm: number | null) => get<RiskEstimate>(`/fits/${id}/risk/`, { group, arm }),
    calibration: (id: Id) => get<Calibration>(`/fits/${id}/calibration/`),
    mctm: (id: Id, q: { structure?: string; horizon?: number; start?: string }) => get<Mctm>(`/fits/${id}/mctm/`, q),
    arms: (id: Id) => get<FitArms>(`/fits/${id}/arms/`),
    shadowEvaluate: (id: Id, body: ShadowEvaluateRequest) => post<ShadowReport>(`/fits/${id}/shadow/evaluate/`, body),
    portfolioCreate: (id: Id, body: PortfolioRequest) => post<Job>(`/fits/${id}/shadow/portfolio/`, body),
    portfolioList: (id: Id) => get<PortfolioRun[]>(`/fits/${id}/shadow/portfolio/`),
    portfolioGet: (id: Id, jobId: Id) => get<PortfolioResult>(`/fits/${id}/shadow/portfolio/${jobId}/`),
  },
  simulations: {
    list: (fit?: Id | null) => page<Simulation>('/simulations/', { fit, limit: 200 }),
    get: (id: Id) => get<Simulation>(`/simulations/${id}/`),
    create: (body: SimulationCreate) => post<Simulation>('/simulations/', body),
    timeseries: (id: Id, policy?: PolicyName) => get<Timeseries>(`/simulations/${id}/timeseries/`, { policy }),
    arms: (id: Id, q: { policy?: PolicyName; epoch?: string | null; group?: string | null }) => get<SimArms>(`/simulations/${id}/arms/`, q),
    posteriors: (id: Id, q: { policy?: PolicyName; group?: string | null; arms?: number[] }) =>
      get<PosteriorHistory>(`/simulations/${id}/posteriors/`, q),
    simpson: (id: Id, q: { policy?: PolicyName; epoch?: string | null }) => get<SimpsonPanel>(`/simulations/${id}/simpson/`, q),
    events: (id: Id, q: EventQuery) => get<EventPage>(`/simulations/${id}/events/`, q),
    sentinel: (id: Id) => get<SentinelTimeline>(`/simulations/${id}/sentinel/`),
    buckets: (id: Id, feature: string, epoch: string) => get<PsiBucketTable>(`/simulations/${id}/sentinel/buckets/`, { feature, epoch }),
    report: (id: Id) => get<EvaluationReport>(`/simulations/${id}/report/`),
    control: (id: Id, action: ControlAction) => post<Simulation>(`/simulations/${id}/${action}/`),
  },
  reviews: {
    list: (q: { simulation?: Id | null; status?: 'pending' | 'acknowledged' } = {}) => page<Review>('/sentinel/reviews/', { ...q, limit: 200 }),
    acknowledge: (id: Id, body: AcknowledgeRequest) => post<Review>(`/sentinel/reviews/${id}/acknowledge/`, body),
  },
};

// ───────────────────────────────────────────────────────────── query keys
export const qk = {
  health: ['health'] as const,
  jobs: (q: object) => ['jobs', q] as const,
  datasets: ['datasets'] as const,
  dataset: (id: Id) => ['datasets', id] as const,
  variables: (id: Id) => ['datasets', id, 'variables'] as const,
  summary: (id: Id, features?: string[]) => ['datasets', id, 'summary', features ?? null] as const,
  loans: (id: Id, q: LoanQuery) => ['datasets', id, 'loans', q] as const,
  fits: (dataset?: Id | null) => ['fits', 'list', dataset ?? null] as const,
  fit: (id: Id) => ['fits', id] as const,
  fitPart: (id: Id, part: string, q?: object) => ['fits', id, part, q ?? null] as const,
  shadow: (fit: Id, req: ShadowEvaluateRequest) => ['fits', fit, 'shadow', req] as const,
  portfolios: (fit: Id) => ['fits', fit, 'portfolio'] as const,
  portfolio: (fit: Id, job: Id) => ['fits', fit, 'portfolio', job] as const,
  simulations: (fit?: Id | null) => ['simulations', 'list', fit ?? null] as const,
  simulation: (id: Id) => ['simulations', id] as const,
  simPart: (id: Id, part: string, q?: object) => ['simulations', id, part, q ?? null] as const,
  reviews: (q: object) => ['reviews', q] as const,
};

const building = (s: BuildStatus) => s === 'queued' || s === 'running';
export const simLive = (s: Simulation | undefined | null) => !!s && (s.status === 'queued' || s.status === 'running');
const on = (id: Id | null | undefined): id is Id => id !== null && id !== undefined;

// ───────────────────────────────────────────────────────────── hooks
export const useHealth = () => useQuery({ queryKey: qk.health, queryFn: api.health, refetchInterval: 30_000, retry: 0 });

export const useActiveJobs = () =>
  useQuery({
    queryKey: qk.jobs({ status: ['queued', 'running'] }),
    queryFn: () => api.jobs.list({ status: ['queued', 'running'] }),
    refetchInterval: 3000,
  });

export const useJob = (id: Id | null | undefined) =>
  useQuery({
    queryKey: ['jobs', id],
    queryFn: () => api.jobs.get(id as Id),
    enabled: on(id),
    refetchInterval: (q) => (q.state.data && ['queued', 'running'].includes(q.state.data.status) ? POLL_MS : false),
  });

export const useDatasets = () =>
  useQuery({
    queryKey: qk.datasets,
    queryFn: api.datasets.list,
    refetchInterval: (q) => (q.state.data?.results.some((d) => building(d.status)) ? POLL_MS : false),
  });

export const useDataset = (id: Id | null | undefined) =>
  useQuery({
    queryKey: qk.dataset(id as Id),
    queryFn: () => api.datasets.get(id as Id),
    enabled: on(id),
    refetchInterval: (q) => (q.state.data && building(q.state.data.status) ? POLL_MS : false),
  });

export const useVariables = (id: Id | null | undefined, ready: boolean) =>
  useQuery({ queryKey: qk.variables(id as Id), queryFn: () => api.datasets.variables(id as Id), enabled: on(id) && ready });

export const useDatasetSummary = (id: Id | null | undefined, ready: boolean) =>
  useQuery({ queryKey: qk.summary(id as Id), queryFn: () => api.datasets.summary(id as Id), enabled: on(id) && ready });

export const useLoans = (id: Id | null | undefined, q: LoanQuery, ready: boolean) =>
  useQuery({
    queryKey: qk.loans(id as Id, q),
    queryFn: ({ signal }) => api.datasets.loans(id as Id, q, signal),
    enabled: on(id) && ready,
    placeholderData: keepPreviousData,
  });

export const useFits = (dataset?: Id | null, enabled = true) =>
  useQuery({
    queryKey: qk.fits(dataset),
    queryFn: () => api.fits.list(dataset),
    enabled,
    refetchInterval: (q) => (q.state.data?.results.some((f) => building(f.status)) ? POLL_MS : false),
  });

export const useFit = (id: Id | null | undefined) =>
  useQuery({
    queryKey: qk.fit(id as Id),
    queryFn: () => api.fits.get(id as Id),
    enabled: on(id),
    refetchInterval: (q) => (q.state.data && building(q.state.data.status) ? POLL_MS : false),
  });

export const useGroups = (id: Id | null | undefined, ready: boolean, q: { level?: string; group?: string } = {}) =>
  useQuery({
    queryKey: qk.fitPart(id as Id, 'groups', q),
    queryFn: () => api.fits.groups(id as Id, q),
    enabled: on(id) && ready && (q.level !== 'leaf' || !!q.group),
  });

export const useRisk = (id: Id | null | undefined, ready: boolean, group: string | null, arm: number | null) =>
  useQuery({
    queryKey: qk.fitPart(id as Id, 'risk', { group, arm }),
    queryFn: () => api.fits.risk(id as Id, group as string, arm),
    enabled: on(id) && ready && !!group,
    placeholderData: keepPreviousData,
  });

export const useCalibration = (id: Id | null | undefined, ready: boolean) =>
  useQuery({ queryKey: qk.fitPart(id as Id, 'calibration'), queryFn: () => api.fits.calibration(id as Id), enabled: on(id) && ready });

export const useMctm = (id: Id | null | undefined, ready: boolean, q: { structure?: string; horizon?: number; start?: string }) =>
  useQuery({
    queryKey: qk.fitPart(id as Id, 'mctm', q),
    queryFn: () => api.fits.mctm(id as Id, q),
    enabled: on(id) && ready,
    placeholderData: keepPreviousData,
  });

export const useFitArms = (id: Id | null | undefined, ready: boolean) =>
  useQuery({ queryKey: qk.fitPart(id as Id, 'arms'), queryFn: () => api.fits.arms(id as Id), enabled: on(id) && ready });

export const useShadowReport = (fit: Id | null | undefined, ready: boolean, req: ShadowEvaluateRequest | null) =>
  useQuery({
    queryKey: qk.shadow(fit as Id, req as ShadowEvaluateRequest),
    queryFn: () => api.fits.shadowEvaluate(fit as Id, req as ShadowEvaluateRequest),
    enabled: on(fit) && ready && req !== null,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });

export const usePortfolioRuns = (fit: Id | null | undefined, ready: boolean) =>
  useQuery({
    queryKey: qk.portfolios(fit as Id),
    queryFn: () => api.fits.portfolioList(fit as Id),
    enabled: on(fit) && ready,
    refetchInterval: (q) => (q.state.data?.some((r) => ['queued', 'running'].includes(r.job.status)) ? POLL_MS : false),
  });

export const usePortfolio = (fit: Id | null | undefined, job: Id | null | undefined) =>
  useQuery({
    queryKey: qk.portfolio(fit as Id, job as Id),
    queryFn: () => api.fits.portfolioGet(fit as Id, job as Id),
    enabled: on(fit) && on(job),
    refetchInterval: (q) => (q.state.data && ['queued', 'running'].includes(q.state.data.job.status) ? POLL_MS : false),
  });

export const useSimulations = (fit?: Id | null, enabled = true) =>
  useQuery({
    queryKey: qk.simulations(fit),
    queryFn: () => api.simulations.list(fit),
    enabled,
    refetchInterval: (q) => (q.state.data?.results.some(simLive) ? POLL_MS : false),
  });

export const useSimulation = (id: Id | null | undefined) =>
  useQuery({
    queryKey: qk.simulation(id as Id),
    queryFn: () => api.simulations.get(id as Id),
    enabled: on(id),
    refetchInterval: (q) => (simLive(q.state.data) ? POLL_MS : false),
  });

/** Options shared by simulation sub-resources: enabled once epochs exist, polled while live. */
const simPart = (id: Id | null | undefined, sim: Simulation | undefined) => ({
  enabled: on(id) && !!sim && sim.epochs_done > 0,
  refetchInterval: simLive(sim) ? POLL_MS : (false as const),
  placeholderData: keepPreviousData,
});

export const useTimeseries = (id: Id | null | undefined, sim: Simulation | undefined) =>
  useQuery({ queryKey: qk.simPart(id as Id, 'timeseries'), queryFn: () => api.simulations.timeseries(id as Id), ...simPart(id, sim) });

export const useSimArms = (id: Id | null | undefined, sim: Simulation | undefined, q: { policy?: PolicyName; epoch?: string | null; group?: string | null }) =>
  useQuery({ queryKey: qk.simPart(id as Id, 'arms', q), queryFn: () => api.simulations.arms(id as Id, q), ...simPart(id, sim) });

export const usePosteriors = (id: Id | null | undefined, sim: Simulation | undefined, q: { policy?: PolicyName; group?: string | null }) =>
  useQuery({ queryKey: qk.simPart(id as Id, 'posteriors', q), queryFn: () => api.simulations.posteriors(id as Id, q), ...simPart(id, sim) });

export const useSimpson = (id: Id | null | undefined, sim: Simulation | undefined, q: { policy?: PolicyName; epoch?: string | null }) =>
  useQuery({ queryKey: qk.simPart(id as Id, 'simpson', q), queryFn: () => api.simulations.simpson(id as Id, q), ...simPart(id, sim) });

export const useEvents = (id: Id | null | undefined, sim: Simulation | undefined, q: EventQuery) =>
  useQuery({
    queryKey: qk.simPart(id as Id, 'events', q),
    queryFn: () => api.simulations.events(id as Id, q),
    enabled: on(id) && !!sim,
    refetchInterval: simLive(sim) ? POLL_MS : false,
    placeholderData: keepPreviousData,
  });

export const useSentinel = (id: Id | null | undefined, sim: Simulation | undefined) =>
  useQuery({ queryKey: qk.simPart(id as Id, 'sentinel'), queryFn: () => api.simulations.sentinel(id as Id), ...simPart(id, sim) });

export const useBuckets = (id: Id | null | undefined, sim: Simulation | undefined, feature: string | null, epoch: string | null) =>
  useQuery({
    queryKey: qk.simPart(id as Id, 'buckets', { feature, epoch }),
    queryFn: () => api.simulations.buckets(id as Id, feature as string, epoch as string),
    ...simPart(id, sim),
    enabled: on(id) && !!sim && sim.epochs_done > 0 && !!feature && !!epoch,
  });

export const useReport = (id: Id | null | undefined, sim: Simulation | undefined) =>
  useQuery({ queryKey: qk.simPart(id as Id, 'report'), queryFn: () => api.simulations.report(id as Id), ...simPart(id, sim) });

export const useReviews = (q: { simulation?: Id | null; status?: 'pending' | 'acknowledged' } = {}) =>
  useQuery({ queryKey: qk.reviews(q), queryFn: () => api.reviews.list(q), refetchInterval: 4000 });

// ───────────────────────────────────────────────────────────── mutations
export function useCreateDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.datasets.create,
    onSuccess: () => Promise.all([qc.invalidateQueries({ queryKey: qk.datasets }), qc.invalidateQueries({ queryKey: ['jobs'] })]),
  });
}

export function useCreateFit() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.fits.create,
    onSuccess: () => Promise.all([qc.invalidateQueries({ queryKey: ['fits'] }), qc.invalidateQueries({ queryKey: ['jobs'] })]),
  });
}

export function useCreateSimulation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.simulations.create,
    onSuccess: () => Promise.all([qc.invalidateQueries({ queryKey: ['simulations'] }), qc.invalidateQueries({ queryKey: ['jobs'] })]),
  });
}

export function useSimulationControl(id: Id | null | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (action: ControlAction) => api.simulations.control(id as Id, action),
    onSuccess: (sim) => {
      qc.setQueryData(qk.simulation(sim.id), sim);
      return Promise.all([
        qc.invalidateQueries({ queryKey: ['simulations'] }),
        qc.invalidateQueries({ queryKey: ['jobs'] }),
        qc.invalidateQueries({ queryKey: ['reviews'] }),
      ]);
    },
  });
}

export function useAcknowledgeReview() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: Id; body: AcknowledgeRequest }) => api.reviews.acknowledge(id, body),
    onSuccess: () => Promise.all([qc.invalidateQueries({ queryKey: ['reviews'] }), qc.invalidateQueries({ queryKey: ['simulations'] })]),
  });
}

export function useCreatePortfolio(fit: Id | null | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: PortfolioRequest) => api.fits.portfolioCreate(fit as Id, body),
    onSuccess: () =>
      Promise.all([qc.invalidateQueries({ queryKey: qk.portfolios(fit as Id) }), qc.invalidateQueries({ queryKey: ['jobs'] })]),
  });
}
