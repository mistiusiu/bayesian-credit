/**
 * Bayesian Credit REST API — authoritative JSON contract (base path `/api/v1/`).
 *
 * The Django/DRF backend implements exactly these shapes; `docs/API.md` is the
 * human-readable twin of this file and the two must change together.
 *
 * Conventions (apply everywhere unless a field says otherwise)
 * - keys are snake_case; ids are integers; booleans are real JSON booleans
 * - timestamps: RFC 3339 / ISO-8601 UTC strings, e.g. "2026-10-16T09:41:07Z"
 * - epochs (issue months, simulation steps): "YYYY-MM" strings, e.g. "2016-06"
 * - money: plain JSON numbers in US dollars (not cents, not strings)
 * - rates, shares, probabilities: fractions in [0, 1] (0.1299, never "12.99%")
 * - `dti` keeps Lending Club's percent-point scale (18.5 means 18.5 %)
 * - "not available" is `null`, never NaN / "" / omitted (except optional request fields)
 * - collection endpoints return the DRF LimitOffset envelope {@link Paginated}
 * - errors: 4xx/5xx with `{ "detail": "..." }` or DRF field errors `{ "<field>": ["..."] }`
 */

// ─────────────────────────────────────────────────────────────── primitives
export type Id = number;
/** RFC 3339 UTC timestamp, e.g. "2026-10-16T09:41:07Z". */
export type IsoDateTime = string;
/** Issue month / simulation epoch, "YYYY-MM". */
export type Epoch = string;
/** Value in [0, 1]. */
export type Fraction = number;
/** US dollars. */
export type Usd = number;
/** Segmenter key: dimension values joined by "|" in `segmentation.dimensions` order, e.g. "B|debt_consolidation|RENT". */
export type GroupKey = string;
/** Arm index into the fit's arm space. 0 is the null arm (decline) when `include_null` is true. */
export type ArmId = number;

// ─────────────────────────────────────────────────────────────── vocabulary
/** Table 3 / Appendix F lifecycle stage of a retained variable. */
export type Stage = 'pre_decision' | 'policy' | 'disbursement' | 'interim' | 'terminal';
export type VariableKind = 'numeric' | 'categorical' | 'date';
export type DataSource = 'synthetic' | 'lending_club';
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type JobKind = 'ingest' | 'fit' | 'simulation' | 'shadow_portfolio';
/** Lifecycle of datasets and fits (derived from their job). */
export type BuildStatus = 'queued' | 'running' | 'ready' | 'failed' | 'cancelled';
/**
 * Simulation lifecycle.
 * - `awaiting_review`: breaker tripped with `breaker_mode="pause"`; waits for POST /sentinel/reviews/{id}/acknowledge/
 * - `interrupted`: the worker died (chaos SIGKILL, OOM, node loss); POST /resume/ restarts from the last checkpoint
 */
export type SimulationStatus =
  | 'queued'
  | 'running'
  | 'paused'
  | 'awaiting_review'
  | 'interrupted'
  | 'completed'
  | 'failed'
  | 'cancelled';
export type TransitionStructure = 'proposal' | 'extended';
/** MCTM states in matrix order: index 0..3 = current, delinquent, recovered, loss. */
export type CreditState = 'current' | 'delinquent' | 'recovered' | 'loss';
export type FeedbackMode = 'simulated' | 'replay' | 'hybrid';
export type BreakerMode = 'pause' | 'auto_ack' | 'off';
export type BreakerState = 'closed' | 'open' | 'half_open';
/** Table 9 bands: normal < 0.10 ≤ watch ≤ 0.25 < breaker (thresholds configurable). */
export type SentinelLevel = 'normal' | 'watch' | 'breaker';
export type ReviewAction = 'resume' | 'resegment' | 'rollback';
export type ReviewStatus = 'pending' | 'acknowledged';
export type BreakerFallback = 'champion_snapshot' | 'decline' | 'historical';
/** Policies the console can configure. `thompson_se` = Thompson sampling with N(μ, σ²/n). */
export type KnownPolicy = 'thompson' | 'thompson_se' | 'ucb1' | 'epsilon_greedy' | 'greedy' | 'historical' | 'random';
/**
 * A policy spec as served. Besides the known names the API accepts any core spec, e.g.
 * `"epsilon_greedy(epsilon=0.05)"`, and echoes it verbatim in configs, rows and reports.
 */
export type PolicyName = KnownPolicy | (string & Record<never, never>);
export type SegmentationDimension =
  | 'grade'
  | 'sub_grade'
  | 'purpose_group'
  | 'home_ownership'
  | 'fico_band'
  | 'dti_band'
  | 'income_band'
  | 'emp_band'
  | 'verification_status'
  | 'application_type';
export type HierarchyLevel = 'global' | 'group' | 'leaf';
export type RecoveryBasis = 'principal' | 'post_chargeoff';
export type EventLevel = 'info' | 'warning' | 'critical';

// ─────────────────────────────────────────────────────────────── envelopes
/** DRF LimitOffsetPagination envelope. Query params: `limit` (default 50, max 500) and `offset`. */
export interface Paginated<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

/** Error body. DRF field errors arrive as top-level `{ field: [messages] }` and are also accepted. */
export interface ApiErrorBody {
  detail?: string;
  code?: string;
  [field: string]: unknown;
}

// ─────────────────────────────────────────────────────────────── health & jobs
/** GET /health/ */
export interface Health {
  status: 'ok' | 'degraded';
  version: string;
  python: string;
  core_version: string;
}

export interface JobTarget {
  type: 'dataset' | 'fit' | 'simulation';
  id: Id;
}

/** Background work (`python manage.py run_job <id>` subprocess or k8s Job). GET /jobs/{id}/ */
export interface Job {
  id: Id;
  kind: JobKind;
  status: JobStatus;
  /** 0..1 */
  progress: Fraction;
  /** Latest human-readable status line. */
  message: string;
  /** Last ≤ 50 log lines, oldest first. */
  log_tail: string[];
  created_at: IsoDateTime;
  started_at: IsoDateTime | null;
  finished_at: IsoDateTime | null;
  /** Resource the job builds or drives. Shadow-portfolio jobs point at their fit. */
  target: JobTarget | null;
  pid: number | null;
}

// ─────────────────────────────────────────────────────────────── datasets
/** Synthetic distribution shift (only for source="synthetic"). */
export interface DistributionShift {
  /** First affected issue month. */
  at: Epoch;
  /** Additive shift per feature, in the feature's own units, e.g. `{ "fico_range_low": -25, "dti": 5 }`. */
  deltas: Record<string, number>;
}

/** POST /datasets/ → 201 {@link Dataset} (status "queued", `job` populated). */
export interface DatasetCreate {
  name: string;
  source: DataSource;
  /** lending_club only; defaults to the server's configured Kaggle file. */
  path?: string | null;
  /** synthetic only; number of loans (default 200000). */
  n?: number | null;
  /** default 42 */
  seed?: number | null;
  /** lending_club only; cap on rows read. */
  max_rows?: number | null;
  /** lending_club only; Bernoulli row sample in (0, 1]. */
  sample_fraction?: number | null;
  issue_from?: Epoch | null;
  issue_to?: Epoch | null;
  shift?: DistributionShift | null;
}

/** GET /datasets/ (Paginated) and GET /datasets/{id}/ */
export interface Dataset {
  id: Id;
  name: string;
  source: DataSource;
  status: BuildStatus;
  /** Echo of the creation request with server defaults filled in. */
  params: DatasetCreate;
  /** Statistics are null until status = "ready". */
  n_loans: number | null;
  n_resolved: number | null;
  issue_min: Epoch | null;
  issue_max: Epoch | null;
  /** Share of loans with a terminal outcome (Fully Paid / Charged Off / Default variants). */
  resolved_share: Fraction | null;
  /** Default share among resolved loans. */
  default_rate: Fraction | null;
  /** Raw `loan_status` → count. */
  status_counts: Record<string, number>;
  created_at: IsoDateTime;
  job: Job | null;
}

/** GET /datasets/{id}/variables/ → VariableRow[] (Table 3, not paginated, Table 3 order). */
export interface VariableRow {
  name: string;
  stage: Stage;
  primary_use: string;
  kind: VariableKind;
  present: boolean;
  /** pandas dtype string ("float64", "string", "datetime64[ns]", "int64") or null when absent. */
  dtype: string | null;
  null_share: Fraction | null;
}

export type FeatureUnit = 'usd' | 'fraction' | 'pct' | 'count' | 'score' | 'years' | 'months' | 'none';

export interface HistogramBin {
  lo: number;
  hi: number;
  count: number;
}

export interface NumericFeatureSummary {
  feature: string;
  kind: 'numeric';
  stage: Stage;
  unit: FeatureUnit;
  n: number;
  null_share: Fraction;
  mean: number;
  std: number;
  p05: number;
  p50: number;
  p95: number;
  /** ~20 equal-width bins between p01 and p99 (outliers clipped into the edge bins). */
  bins: HistogramBin[];
}

export interface CategoricalFeatureSummary {
  feature: string;
  kind: 'categorical';
  stage: Stage;
  n: number;
  null_share: Fraction;
  /** Sorted by count desc; at most 12 levels, the tail folded into "other". */
  levels: { level: string; count: number }[];
}

export type FeatureSummary = NumericFeatureSummary | CategoricalFeatureSummary;

export interface MonthlyArrival {
  epoch: Epoch;
  n: number;
  n_resolved: number;
  n_default: number;
}

/**
 * GET /datasets/{id}/summary/?features=a,b,c
 * Without `features` the server returns its default set: fico_range_low, dti, annual_inc,
 * revol_util, inq_last_6mths, emp_length, loan_amnt, int_rate, grade, purpose, home_ownership,
 * verification_status, term.
 */
export interface DatasetSummary {
  dataset: Id;
  n_loans: number;
  features: FeatureSummary[];
  /** Every month between issue_min and issue_max, zero-filled. */
  monthly_arrivals: MonthlyArrival[];
  status_counts: Record<string, number>;
  grade_counts: Record<string, number>;
  term_counts: Record<string, number>;
}

/** X_i — the 37 PRE_DECISION columns of Table 3. Nothing else may appear here (LeakageGuard). */
export interface BorrowerContext {
  annual_inc: number | null;
  annual_inc_joint: number | null;
  application_type: string | null;
  dti: number | null;
  dti_joint: number | null;
  emp_length: number | null;
  fico_range_low: number | null;
  fico_range_high: number | null;
  home_ownership: string | null;
  purpose: string | null;
  verification_status: string | null;
  verified_status_joint: string | null;
  delinq_2yrs: number | null;
  delinq_amnt: number | null;
  inq_last_6mths: number | null;
  inq_last_12m: number | null;
  open_acc: number | null;
  total_acc: number | null;
  revol_bal: number | null;
  revol_util: number | null;
  installment: number | null;
  mort_acc: number | null;
  pub_rec: number | null;
  pub_rec_bankruptcies: number | null;
  collections_12_mths_ex_med: number | null;
  num_accts_ever_120_pd: number | null;
  pct_tl_nvr_dlq: number | null;
  num_sats: number | null;
  tot_coll_amt: number | null;
  tot_cur_bal: number | null;
  tot_hi_cred_lim: number | null;
  total_bal_ex_mort: number | null;
  total_bc_limit: number | null;
  bc_util: number | null;
  all_util: number | null;
  grade: string | null;
  sub_grade: string | null;
}

/** POLICY-stage columns: the historical arm actually offered. */
export interface LoanPolicy {
  loan_amnt: Usd;
  int_rate: Fraction;
  term: number;
}

/** Derived TERMINAL outcome (`y_*` columns). Post-hoc only; never offered to a policy. */
export interface LoanOutcome {
  loan_status: string;
  resolved: boolean;
  /** y_default; null while unresolved. */
  defaulted: boolean | null;
  /** y_recovery_rate (principal basis by default); null unless defaulted. */
  recovery_rate: Fraction | null;
  realized_revenue: Usd | null;
  realized_loss: Usd | null;
  /** Realized reward of the historical policy: revenue − loss − funding on the realized schedule. */
  realized_reward: Usd | null;
  months_on_book: number | null;
  resolution_epoch: Epoch | null;
}

/**
 * GET /datasets/{id}/loans/ → Paginated<LoanRecord>
 * Query: limit, offset, grade=A,B · purpose=debt_consolidation,credit_card · home_ownership=RENT,OWN
 * · resolved=true|false · search=<loan_id substring>. Ordered by loan_id ascending.
 */
export interface LoanRecord {
  loan_id: number;
  issue_d: Epoch;
  funded_amnt: Usd;
  /** y_group under the dataset's default segmentation (grade, purpose_group, home_ownership). */
  group: GroupKey;
  /** y_historical_arm on the default arm grid. */
  historical_arm: ArmId;
  context: BorrowerContext;
  policy: LoanPolicy;
  outcome: LoanOutcome;
}

// ─────────────────────────────────────────────────────────────── fits
export interface ArmSpaceConfig {
  amounts: Usd[];
  rates: Fraction[];
  terms: number[];
  include_null: boolean;
  /** Guardrail: installment ≤ x · annual_inc / 12 (default 0.20; null = off). */
  max_payment_to_income: Fraction | null;
  /** Guardrail: amount ≤ x · annual_inc (default 0.50; null = off). */
  max_amount_to_income: Fraction | null;
  /** Guardrail: dti/100 + 12·installment/annual_inc ≤ x (QM 43 % ceiling; default 0.43; null = off). */
  max_total_dti: Fraction | null;
  /** Guardrail: |rate − sub-grade reference rate| ≤ x (default 0.05; null = off). */
  rate_corridor: Fraction | null;
}

/** The guardrails actually applied (echo of the arm-space settings). */
export interface Guardrails {
  max_payment_to_income: Fraction | null;
  rate_corridor: Fraction | null;
  max_amount_to_income: Fraction | null;
  max_total_dti: Fraction | null;
}

export type PolicyEffectFeature = 'int_rate' | 'term60' | 'pti' | 'log_amount';

/** Offset logistic regression behind the structured leaf prior (docs/MATH.md §2.6). */
export interface PolicyEffectConfig {
  features: PolicyEffectFeature[];
  ridge: number;
  max_abs_logit_shift: number;
  min_rows: number;
  min_elapsed_rows: number;
  max_iter: number;
  pti_cap: number;
}

export interface HierarchyConfig {
  kappa_group: number | 'auto';
  kappa_leaf: number | 'auto';
  alpha0: number;
  beta0: number;
  /** Structured leaf prior: tilt each leaf's prior mean by the within-group policy effect (default true). */
  policy_effect: boolean;
  policy_effect_cfg: PolicyEffectConfig;
}

export interface RecoveryConfig {
  min_obs: number;
  basis: RecoveryBasis;
  /** Stage-2 fallback leaf → (group, term) → group → global (default true, docs/MATH.md §2.9). */
  term_level: boolean;
}

export interface MctmConfig {
  structure: TransitionStructure;
  prior_strength: number;
  chargeoff_lag_months: number;
}

export interface RewardConfig {
  /** Annual cost of funds applied to the weighted-average life. */
  funding_rate: Fraction;
  /** λ in score = mean − λ·std. */
  risk_aversion: number;
  /** P(reward > 0) needed to call a borrower lendable. */
  lendable_threshold: Fraction;
  /** `"empirical"`: interest and funding × realised share ρ(grade, term) (docs/MATH.md §2.8); null = contract schedule. */
  prepayment_adjustment: 'empirical' | null;
  /** Exposure used for the expected loss ("full" = L_t, SPEC §2.7). */
  exposure_basis: 'full' | 'outstanding';
}

/** Fully resolved fit configuration (server defaults applied). */
export interface FitConfig {
  segmentation: { dimensions: SegmentationDimension[] };
  arm_space: ArmSpaceConfig;
  hierarchy: HierarchyConfig;
  recovery: RecoveryConfig;
  mctm: MctmConfig;
  reward: RewardConfig;
  /** Fit on issue_d < first day of this month; resolved test loans are issue_d ≥ train_end. */
  train_end: Epoch;
}

/** POST /fits/ → 201 {@link Fit}. Every config block is optional; omitted keys take the defaults in docs/API.md. */
export interface FitCreate {
  dataset: Id;
  name: string;
  segmentation?: { dimensions: SegmentationDimension[] };
  arm_space?: Partial<ArmSpaceConfig>;
  hierarchy?: Partial<Omit<HierarchyConfig, 'policy_effect_cfg'>> & { policy_effect_cfg?: Partial<PolicyEffectConfig> };
  recovery?: Partial<RecoveryConfig>;
  mctm?: Partial<MctmConfig>;
  reward?: Partial<RewardConfig>;
  train_end?: Epoch;
}

/**
 * Holdout metrics (Table 11 predictive dimensions). Computed on resolved loans with
 * issue_d ≥ train_end — recommended: matured cohorts only (issue + term ≤ snapshot), so the
 * resolved-only sample is not dominated by early defaults. `n_test` reports the size.
 */
export interface FitMetrics {
  brier: number;
  /** Brier of the constant training base rate (skill reference). */
  brier_baseline: number;
  auc: number;
  /** Expected calibration error over the 10 reliability bins (prediction deciles, see {@link ReliabilityBin}). */
  ece: number;
  log_loss: number;
  recovery_mae: number;
  recovery_rmse: number;
  /** Holdout transition log-likelihood (nats, summed over loan-months). */
  transition_log_lik: number;
  transition_log_lik_per_obs: number;
  /** Mean absolute row error of P vs holdout empirical transition frequencies. */
  transition_prediction_error: number;
  n_train: number;
  n_test: number;
  n_defaults_train: number;
  base_rate_train: Fraction;
  base_rate_test: Fraction;
  fit_seconds: number;
  /** Within-group policy effect behind the leaf-prior tilt (absent on fits created before API §13.1). */
  policy_effect?: PolicyEffectDiagnostics | null;
  /** Realised share ρ of scheduled interest (absent on older fits). */
  interest_realization?: InterestRealization | null;
  guardrails?: Guardrails | null;
  reference_rates?: ReferenceRates | null;
  /** Served extras: the same metrics on all resolved holdout loans, sizes and timings. */
  brier_secondary?: number | null;
  auc_secondary?: number | null;
  n_holdout?: number;
  n_train_all?: number;
  evaluate_seconds?: number;
  job_seconds?: number;
}

/**
 * Offset logistic regression logit P(D) = logit(m_g) + β·(x − x̄_g), x = [int_rate, 1[term=60],
 * payment-to-income, log amount] centred within group (docs/MATH.md §2.6).
 */
export interface PolicyEffectDiagnostics {
  /** `fitted` · `fallback` (β = 0) · `disabled` · `no_arm_space` · `missing_columns: …` · `not_fitted`. */
  status: string;
  /** Status reported by the regression itself. */
  model_status?: string | null;
  installed?: boolean;
  type?: string | null;
  /** Why the fit fell back to β = 0. */
  reason?: string | null;
  /** Logit change per unit of each feature (rate as a fraction, term60 0/1, PTI as a fraction, log USD). */
  beta?: Partial<Record<PolicyEffectFeature, number>>;
  std_errors?: Partial<Record<PolicyEffectFeature, number>>;
  n_fit?: number;
  /** `elapsed` (loans whose term elapsed) or `resolved`. */
  sample?: string | null;
  converged?: boolean;
  n_iter?: number;
  /** Clip on the tilt |t(g, a)|. */
  max_abs_logit_shift?: number;
}

/** ρ = realised / scheduled interest of fully-paid training loans (docs/MATH.md §2.8). */
export interface InterestRealization {
  global: number;
  /** Keyed by term ("36", "60"). */
  by_term: Record<string, number>;
  /** grade → term → ρ; cells with fewer than `min_count` loans are absent (fallback to term, then global). */
  by_grade_term: Record<string, Record<string, number>>;
  n_fit: number;
  sample: string | null;
  min_count: number;
  clip: [number, number];
  prepayment_adjustment: 'empirical' | null;
}

/** Mean training int_rate (the rate-corridor anchor; the corridor uses the sub-grade → grade → global chain). */
export interface ReferenceRates {
  n_fit: number;
  by_grade: Record<string, number>;
  global: number;
}

/** GET /fits/ (Paginated) and GET /fits/{id}/ */
export interface Fit {
  id: Id;
  name: string;
  dataset: Id;
  dataset_name: string;
  status: BuildStatus;
  config: FitConfig;
  metrics: FitMetrics | null;
  n_groups: number | null;
  /** Including the null arm. */
  n_arms: number | null;
  created_at: IsoDateTime;
  job: Job | null;
}

/** Stage-2 recovery posterior attached to a hierarchy node. */
/** Stage-2 node that produced a recovery estimate (`group_term` = the (group, term) level, MATH §2.9). */
export type RecoveryCoreSource = 'leaf' | 'group_term' | 'group' | 'global' | 'prior';

export interface RecoverySummary {
  mean: Fraction;
  alpha: number;
  beta: number;
  n_obs: number;
  /** Level whose data produced the estimate, in HierarchyLevel vocabulary (`group_term` reports as `group`). */
  source: HierarchyLevel;
  core_source?: RecoveryCoreSource;
}

/**
 * GET /fits/{id}/groups/?level=global|group|leaf&group=<GroupKey> → HierarchyRow[] (not paginated).
 * - no params: the global row followed by every group row
 * - level=group: every group, including groups never seen in training (n_obs 0, posterior = prior):
 *   the cartesian product of observed dimension values (84 for the default segmentation)
 * - level=leaf requires `group` (400 otherwise); returns that group's observed leaves (n_obs > 0),
 *   with `dims.arm` = the arm label
 */
export interface HierarchyRow {
  level: HierarchyLevel;
  /** "global" · group key · leaf key "<group>#<arm>". */
  key: string;
  parent: string | null;
  group: GroupKey | null;
  arm: ArmId | null;
  /** Dimension → value for this node, e.g. { grade: "B", purpose_group: "credit_card", home_ownership: "RENT" }. */
  dims: Record<string, string>;
  alpha: number;
  beta: number;
  n_obs: number;
  defaults: number;
  /** Posterior mean (α + d) / (α + β + n). */
  mean: Fraction;
  hdi_low: Fraction;
  hdi_high: Fraction;
  /** Mean of this node's prior (the parent's posterior mean — the shrinkage target). Null for global. */
  prior_mean: Fraction | null;
  /** Prior concentration κ used for this node. Null for global. */
  kappa: number | null;
  /** n / (n + κ): share of the posterior mean that comes from the node's own data. */
  shrinkage_weight: Fraction;
  /** defaults / n_obs, null when n_obs = 0. */
  empirical_rate: Fraction | null;
  recovery: RecoverySummary | null;
  /** Number of groups (global) / observed leaves (group) / 0 (leaf). */
  n_children: number;
  /**
   * Policy-effect tilt t(g, a) of the leaf prior mean on the logit scale (0 for global / group rows).
   * Computed by the core's `summary()`; served on leaf rows, `null` on global/group rows and on fits saved before the tilt existed.
   */
  prior_tilt_logit?: number | null;
}

export interface BetaSummary {
  alpha: number;
  beta: number;
  mean: Fraction;
  std: number;
  hdi_low: Fraction;
  hdi_high: Fraction;
}

export interface DensityPoint {
  x: number;
  density: number;
}

export interface LendingPolicy {
  amount: Usd;
  rate: Fraction;
  term: number;
}

/** R = Revenue − Expected Loss − Funding (§3.3); closed-form expectation. */
export interface RewardBreakdown {
  mean: Usd;
  std: Usd;
  revenue: Usd;
  expected_loss: Usd;
  funding_cost: Usd;
  /** mean − λ·std */
  risk_adjusted: Usd;
  /** Realised share ρ applied to interest and funding (1 when the adjustment is off). */
  realization?: number;
  /** Contractual interest before default and prepayment adjustments. */
  gross_interest?: Usd;
}

/**
 * GET /fits/{id}/risk/?group=<GroupKey>&arm=<ArmId>
 * `group` is required. Without `arm` the group-level posterior is returned (arm/policy/reward null).
 * Density grids have 101 points on [0, 1].
 */
export interface RiskEstimate {
  fit: Id;
  group: GroupKey;
  arm: ArmId | null;
  policy: LendingPolicy | null;
  label: string;
  p_default: BetaSummary & { level: HierarchyLevel; n_obs: number; defaults: number; shrinkage_weight: Fraction };
  recovery: BetaSummary & { level: HierarchyLevel; n_obs: number; core_source?: RecoveryCoreSource };
  /** E[p]·(1 − E[R]) */
  expected_loss_fraction: Fraction;
  reward: RewardBreakdown | null;
  pdf: {
    p_default: DensityPoint[];
    /** The node's prior (shrinkage target) for comparison. */
    p_default_prior: DensityPoint[];
    recovery: DensityPoint[];
  };
}

/** One of 10 equal-frequency bins (prediction deciles); lo / hi = min / max prediction in the bin. */
export interface ReliabilityBin {
  bin: number;
  lo: Fraction;
  hi: Fraction;
  n: number;
  mean_predicted: Fraction;
  observed_rate: Fraction;
}

export interface RocPoint {
  fpr: Fraction;
  tpr: Fraction;
  threshold: Fraction;
}

export interface RecoveryFit {
  /** Global Beta(α^R, β^R) by method of moments on defaulted training loans. */
  alpha: number;
  beta: number;
  mean: Fraction;
  n_defaults: number;
  /** Holdout realized recovery rates, 20 bins on [0, 1]. */
  histogram: HistogramBin[];
  pdf: DensityPoint[];
  mae: number;
  rmse: number;
  by_grade: { grade: string; mean: Fraction; n: number }[];
  n_holdout_defaults?: number;
}

/** GET /fits/{id}/calibration/ — holdout reliability (10 bins), ROC (≤ 200 points, fpr ascending) and recovery fit. */
export interface Calibration {
  fit: Id;
  n_test: number;
  base_rate: Fraction;
  brier: number;
  auc: number;
  ece: number;
  log_loss: number;
  bins: ReliabilityBin[];
  roc: RocPoint[];
  recovery: RecoveryFit;
  /** Served extras: ECE on equal-width bins, and the metrics on all resolved holdout loans. */
  ece_equal_width?: number;
  secondary?: PredictiveSubset | null;
}

export interface PredictiveSubset {
  subset: string;
  n: number;
  n_defaults: number;
  base_rate: Fraction;
  mean_pred: Fraction;
  brier: number;
  brier_climatology: number;
  brier_skill: number;
  auc: number;
  ece: number;
  log_loss: number;
}

export interface MctmTrajectoryRow {
  t: number;
  current: Fraction;
  delinquent: Fraction;
  recovered: Fraction;
  loss: Fraction;
}

export interface PDefaultPoint {
  t: number;
  /** π_D(t) + π_L(t) */
  p_default: Fraction;
  p_loss: Fraction;
  p_delinquent: Fraction;
}

/**
 * GET /fits/{id}/mctm/?structure=proposal|extended&horizon=60&start=current|delinquent
 * Defaults: structure = the fit's configured structure, horizon 60, start current.
 * Matrices are indexed by `states` (current, delinquent, recovered, loss).
 */
export interface Mctm {
  fit: Id;
  structure: TransitionStructure;
  states: CreditState[];
  state_labels: string[];
  /** 4×4 posterior-mean transition matrix; rows sum to 1; absorbing rows are unit vectors. */
  matrix: number[][];
  /** 4×4 observed loan-month transition counts (training panel). */
  counts: number[][];
  /** 4×4, true where P is forced to 0 by the structure (C→R, C→L under "proposal"; absorbing rows). */
  structural_zeros: boolean[][];
  /** 4×4 elementwise 95% marginal credible bounds from the Dirichlet rows. */
  matrix_low: number[][];
  matrix_high: number[][];
  /** B = N·R, N = (I − Q)⁻¹: rows [current, delinquent] × cols [recovered, loss]. */
  absorption: number[][];
  /** Expected months to absorption from [current, delinquent]. */
  expected_steps: number[];
  start: 'current' | 'delinquent';
  horizon: number;
  /** π_t = π₀Pᵗ for t = 0..horizon (horizon + 1 rows). */
  trajectory: MctmTrajectoryRow[];
  /** P(Default, t) for t = 0..horizon. */
  p_default_curve: PDefaultPoint[];
  /** Maturity collapse at t = horizon: Current → Recovered, Delinquent splits p_DR : p_DL. Vectors in `states` order. */
  maturity: { t: number; before: number[]; after: number[]; p_default: Fraction };
  /** Holdout transition log-likelihood (nats) and its loan-month count. */
  log_likelihood: number;
  n_transitions: number;
  /** Holdout mean absolute error per row (absorbing rows 0). */
  row_prediction_error: number[];
  prior_strength: number;
  /** Structure the fit was estimated with (when `structure` was overridden by the query). */
  fitted_structure?: TransitionStructure;
}

export interface ArmRow {
  arm: ArmId;
  amount: Usd;
  rate: Fraction;
  term: number;
  is_null: boolean;
  installment: Usd;
  /** "$15k @ 13% / 36m" or "decline". */
  label: string;
}

export interface FitArmRow extends ArmRow {
  /** Group-share-weighted expected reward of offering this arm to every borrower (USD per loan). */
  expected_reward: Usd;
  reward_std: Usd;
  p_default: Fraction;
  expected_recovery: Fraction;
  /** Training loans whose historical policy snaps to this arm. */
  historical_count: number;
  historical_share: Fraction;
}

/** GET /fits/{id}/arms/ — arms in grid order (null arm first, then term-major, amount, rate). */
export interface FitArms {
  fit: Id;
  amounts: Usd[];
  rates: Fraction[];
  terms: number[];
  arms: FitArmRow[];
}

// ─────────────────────────────────────────────────────────────── shadow engine
export interface ShadowEvaluateByLoan {
  loan_id: number;
}

export interface ShadowEvaluateByContext {
  /** PRE_DECISION columns only; any other key is rejected with 400 (LeakageError). */
  context: Partial<BorrowerContext>;
  historical_policy?: LendingPolicy | null;
}

/** POST /fits/{id}/shadow/evaluate/ body. */
export type ShadowEvaluateRequest = ShadowEvaluateByLoan | ShadowEvaluateByContext;

export interface ShadowArmRow extends ArmRow {
  /**
   * Passes every guardrail whose input is known (payment-to-income, amount-to-income, total DTI,
   * rate corridor). The null arm, an admitted historical arm and the fallback arm are always eligible.
   */
  eligible: boolean;
  /** Realised share ρ(grade, term) applied to this arm's interest and funding. */
  realization: number;
  p_default_mean: Fraction;
  p_default_hdi_low: Fraction;
  p_default_hdi_high: Fraction;
  expected_recovery: Fraction;
  reward_mean: Usd;
  reward_std: Usd;
  reward_q05: Usd;
  reward_q50: Usd;
  reward_q95: Usd;
  /** P(reward > 0) under the posterior predictive. */
  prob_positive: Fraction;
  /** reward_mean − λ·reward_std (the argmax criterion). */
  risk_adjusted: Usd;
}

/** POST /fits/{id}/shadow/evaluate/ → 200 ShadowReport (synchronous). */
export interface ShadowReport {
  fit: Id;
  loan_id: number | null;
  issue_d: Epoch | null;
  group: GroupKey;
  /** The exact context the engine saw (PRE_DECISION only). */
  context: BorrowerContext;
  /** Columns used; all are PRE_DECISION. */
  context_columns: string[];
  /** True when the context passed LeakageGuard.assert_allowed(..., PRE_DECISION). */
  leakage_safe: boolean;
  n_samples: number;
  risk_aversion: number;
  lendable_threshold: Fraction;
  max_payment_to_income: Fraction;
  /** One row per arm, grid order (null arm first). */
  arms: ShadowArmRow[];
  /** argmax risk_adjusted among eligible arms. */
  recommended_arm: ArmId;
  historical_policy: LendingPolicy | null;
  historical_arm: ArmId | null;
  historical_expected_reward: Usd | null;
  historical_realized_reward: Usd | null;
  /** recommended reward_mean − historical expected reward. */
  uplift_expected: Usd | null;
  /** recommended reward_mean − historical realized reward. */
  uplift_vs_realized: Usd | null;
  /** ∃ eligible lending arm with prob_positive ≥ lendable_threshold. */
  lendable: boolean;
  /** Post-hoc outcome of the historical loan (evaluation only; null for context-only requests). */
  outcome: LoanOutcome | null;
  eligibility: ShadowEligibility;
  /** Served conveniences. */
  recommended_policy?: LendingPolicy | null;
  recommended_label?: string;
  declined?: boolean;
  /** Expected reward of the snapped historical arm (vs `historical_expected_reward` of the exact policy). */
  historical_arm_expected_reward?: Usd | null;
}

/** How the guardrails shaped this borrower's arm set (docs/MATH.md §2.7). */
export interface ShadowEligibility {
  /** Training mean int_rate of the borrower's sub-grade → grade → global; null without reference rates. */
  reference_rate: Fraction | null;
  /** Arms in the final mask (decline and an admitted historical arm included). */
  n_eligible: number;
  /** Lending arms passing the guardrails. */
  n_eligible_lending: number;
  /** The historical arm failed the guardrails but is kept for the comparison. */
  historical_admitted: boolean;
  guardrails: Guardrails;
  rate_corridor_applied: boolean;
}

export interface PortfolioFilters {
  grade?: string[];
  purpose?: string[];
  home_ownership?: string[];
  issue_from?: Epoch;
  issue_to?: Epoch;
  resolved_only?: boolean;
}

/** POST /fits/{id}/shadow/portfolio/ → 202 {@link Job} (kind "shadow_portfolio"). */
export interface PortfolioRequest {
  max_loans: number;
  seed: number;
  filters?: PortfolioFilters;
}

/** First matching class wins: declined > rescued > upsized > downsized > repriced_down > repriced_up > unchanged. */
export type PortfolioOutcomeClass = 'rescued' | 'upsized' | 'downsized' | 'repriced_down' | 'repriced_up' | 'declined' | 'unchanged';

export interface PortfolioSummary {
  n_loans: number;
  n_lendable: number;
  share_lendable: Fraction;
  expected_profit_historical: Usd;
  expected_profit_shadow: Usd;
  uplift: Usd;
  /** uplift / |expected_profit_historical| */
  uplift_pct: number;
  realized_profit_historical: Usd | null;
  counts: Record<PortfolioOutcomeClass, number>;
  /** Served extras. */
  uplift_risk_adjusted?: Usd;
  n_realized?: number;
  seconds?: number;
  /** What the shadow policy offers vs what was issued (absent on runs before API §13.1). */
  recommendation_mix?: RecommendationMix | null;
  /** Uplift attributable to repricing, then resizing, then term choice. */
  uplift_decomposition?: UpliftDecomposition | null;
  /** The shadow policy re-optimised under mean − λ·std; keys "0.0", "0.5", "1.0". */
  by_risk_aversion?: Record<string, RiskAversionPoint> | null;
  eligibility?: PortfolioEligibility | null;
}

/** Distribution of a set of decisions over the arm grid (lending decisions unless stated). */
export interface SelectionMix {
  n: number;
  n_lending: number;
  decline_share: Fraction;
  /** Keyed by term ("36", "60"); shares of lending decisions. */
  by_term: Record<string, Fraction>;
  /** Keys "<10%", "10-15%", "15-20%", "20-25%", ">=25%". */
  by_rate: Record<string, Fraction>;
  /** Keys "<$10k", "$10k-20k", "$20k-30k", ">=$30k". */
  by_amount: Record<string, Fraction>;
  mean_rate: Fraction | null;
  mean_amount: Usd | null;
  share_36m: Fraction | null;
  share_at_max_rate: Fraction | null;
  share_at_max_amount: Fraction | null;
}

export interface RecommendationMix {
  recommended: SelectionMix;
  historical: SelectionMix;
  mean_rate_recommended?: Fraction | null;
  mean_rate_historical?: Fraction | null;
  mean_amount_recommended?: Usd | null;
  mean_amount_historical?: Usd | null;
  share_36m_recommended?: Fraction | null;
  share_36m_historical?: Fraction | null;
  share_at_max_rate_recommended?: Fraction | null;
  share_at_max_amount_recommended?: Fraction | null;
  /** Recommended rates relative to the sub-grade reference rate. */
  reference?: {
    n: number;
    mean_reference_rate: Fraction | null;
    mean_rate_minus_reference: number | null;
    share_above_reference: Fraction | null;
    share_within_corridor: Fraction | null;
    historical_mean_rate_minus_reference: number | null;
  } | null;
}

export interface UpliftStep {
  expected_profit: Usd;
  uplift: Usd;
  /**
   * Fraction of |expected_profit_historical| everywhere (the API converts the core's percent),
   * including inside `EvaluationReport.details`.
   */
  uplift_pct: number;
}

export interface UpliftDecomposition {
  expected_profit_historical: Usd;
  /** Best admissible arm keeping the historical term and amount (rate only). */
  repricing_only: UpliftStep;
  /** Best admissible arm keeping the historical term (rate and amount). */
  repricing_and_resizing: UpliftStep;
  /** Best admissible arm (rate, amount and term). */
  unrestricted: UpliftStep;
  note?: string;
}

export interface RiskAversionPoint {
  risk_aversion: number;
  expected_profit_shadow: Usd;
  expected_profit_historical: Usd;
  uplift: Usd;
  /** Fraction of |expected_profit_historical| (also a fraction inside report `details`). */
  uplift_pct: number;
  mean_amount_recommended: Usd | null;
  mean_rate_recommended: Fraction | null;
  share_36m_recommended: Fraction | null;
  share_at_max_amount_recommended: Fraction | null;
  share_at_max_rate_recommended: Fraction | null;
  decline_share: Fraction;
}

export interface PortfolioEligibility {
  guardrails: Guardrails;
  /** Whether the fit had reference rates (rate corridor active). */
  reference_rates: boolean;
  share_with_reference_rate: Fraction | null;
  mean_n_eligible_lending: number | null;
  /** Borrowers left with at most one eligible lending arm (the fallback arm). */
  share_single_fallback_arm: Fraction | null;
}

export interface PortfolioGradeRow {
  grade: string;
  n: number;
  expected_profit_historical: Usd;
  expected_profit_shadow: Usd;
  realized_profit_historical: Usd | null;
  counts: Record<PortfolioOutcomeClass, number>;
}

export interface InclusionRow {
  grade: string;
  n: number;
  share_lendable: Fraction;
  decline_rate_shadow: Fraction;
  mean_rate_historical: Fraction;
  mean_rate_shadow: Fraction | null;
  mean_amount_historical: Usd;
  mean_amount_shadow: Usd | null;
}

export interface PortfolioLoanRow {
  loan_id: number;
  grade: string;
  group: GroupKey;
  historical_arm: ArmId;
  recommended_arm: ArmId;
  historical_label: string;
  recommended_label: string;
  historical_expected_reward: Usd;
  shadow_expected_reward: Usd;
  realized_reward: Usd | null;
  class: PortfolioOutcomeClass;
}

/** GET /fits/{id}/shadow/portfolio/{job_id}/ — summary is null until the job succeeds. */
export interface PortfolioResult {
  job: Job;
  fit: Id;
  request: PortfolioRequest;
  summary: PortfolioSummary | null;
  by_grade: PortfolioGradeRow[];
  inclusion: InclusionRow[];
  /** The 50 loans with the largest |shadow − historical| expected reward. */
  top_loans: PortfolioLoanRow[];
}

/** GET /fits/{id}/shadow/portfolio/ → PortfolioRun[] (newest first, not paginated). */
export interface PortfolioRun {
  job: Job;
  request: PortfolioRequest;
  summary: PortfolioSummary | null;
}

// ─────────────────────────────────────────────────────────────── simulations
export interface PsiSettings {
  features: string[];
  window_months: number;
  thresholds: { watch: number; breaker: number };
  aggregate: 'max' | 'mean';
  /** Consecutive NORMAL readings that close a HALF_OPEN breaker. */
  recovery_readings: number;
  fallback: BreakerFallback;
  /** Action applied automatically when breaker_mode = "auto_ack". */
  auto_action: ReviewAction;
  auto_ack_epochs: number;
}

export interface SimulationConfig {
  policies: PolicyName[];
  feedback_mode: FeedbackMode;
  /** Inclusive first and last epochs. */
  start: Epoch;
  end: Epoch;
  max_loans: number | null;
  seed: number;
  interim_months: number;
  interim_weight: Fraction;
  psi: PsiSettings;
  breaker_mode: BreakerMode;
  /** Checkpoint cadence in epochs. */
  checkpoint_every: number;
  /** Sample of the hindsight oracle's model: loans whose term elapsed (default) or all resolved loans. */
  hindsight_sample: HindsightSample;
}

export type HindsightSample = 'elapsed' | 'resolved';

/** POST /simulations/ → 201 {@link Simulation}. Omitted fields take the defaults in docs/API.md. */
export interface SimulationCreate {
  fit: Id;
  name: string;
  policies: PolicyName[];
  feedback_mode?: FeedbackMode;
  start?: Epoch;
  end?: Epoch;
  max_loans?: number | null;
  seed?: number;
  interim_months?: number;
  interim_weight?: Fraction;
  psi?: Partial<PsiSettings>;
  breaker_mode?: BreakerMode;
  checkpoint_every?: number;
  hindsight_sample?: HindsightSample;
}

export interface PolicyHeadline {
  policy: PolicyName;
  n_decisions: number;
  cum_reward: Usd;
  cum_realized: Usd;
  cum_regret: Usd;
  /** cum_reward / n_decisions */
  mean_reward: Usd;
  decline_rate: Fraction;
  mean_amount: Usd | null;
  mean_rate: Fraction | null;
}

/** Outcome of the most recent kill → resume-from-checkpoint cycle (G.6 state recovery). */
export interface RecoveryInfo {
  killed_at: IsoDateTime;
  killed_at_epoch: Epoch | null;
  resumed_at: IsoDateTime | null;
  resumed_from_epoch: Epoch | null;
  /** Epochs recomputed after the restart (killed_at_epoch − checkpoint). */
  replayed_epochs: number;
  /** Replayed EpochCompleted metrics identical to the pre-kill ones; null until replay passes the kill point. */
  deterministic_match: boolean | null;
  recovery_seconds: number | null;
}

/** GET /simulations/ (Paginated) and GET /simulations/{id}/; also returned by the control actions. */
export interface Simulation {
  id: Id;
  name: string;
  fit: Id;
  fit_name: string;
  dataset: Id;
  status: SimulationStatus;
  progress: Fraction;
  config: SimulationConfig;
  epochs_total: number;
  epochs_done: number;
  /** Last completed epoch. */
  current_epoch: Epoch | null;
  breaker_state: BreakerState;
  sentinel_level: SentinelLevel;
  psi_aggregate: number | null;
  pending_reviews: number;
  last_checkpoint_epoch: Epoch | null;
  /** Number of resume-from-checkpoint restarts. */
  restarts: number;
  worker: { pid: number | null; alive: boolean; last_heartbeat: IsoDateTime | null };
  last_recovery: RecoveryInfo | null;
  /** One entry per configured policy; [] before the first epoch completes. */
  headline: PolicyHeadline[];
  /** Cumulative reward of the best adaptive policy minus the historical policy (null if historical not run). */
  uplift_vs_historical: Usd | null;
  created_at: IsoDateTime;
  started_at: IsoDateTime | null;
  finished_at: IsoDateTime | null;
  job: Job | null;
  /** Served extras: progress in percent and the job's status line. */
  pct?: number;
  message?: string;
}

/**
 * One policy × epoch. Sums are over the epoch's decisions unless named `cum_*` / `rolling_*`.
 * breaker_state / psi_aggregate are global per epoch (repeated on every policy row).
 */
export interface TimeseriesRow {
  epoch: Epoch;
  policy: PolicyName;
  n_decisions: number;
  /** Σ expected risk-adjusted reward of the arms chosen this epoch (USD). */
  reward: Usd;
  cum_reward: Usd;
  /** Σ realized reward released this epoch (interim × weight + terminal). */
  realized: Usd;
  cum_realized: Usd;
  /** Σ pseudo-regret vs the hindsight oracle this epoch. */
  regret: Usd;
  cum_regret: Usd;
  /** Mean reward per decision over the trailing 6 epochs. */
  rolling_reward: Usd;
  decline_rate: Fraction;
  /** Mean offered amount / rate over non-declined decisions (null if all declined). */
  mean_amount: Usd | null;
  mean_rate: Fraction | null;
  breaker_state: BreakerState;
  psi_aggregate: number | null;
  /** Served extras from the runner. */
  fallback_rate?: Fraction;
  n_feedback_interim?: number;
  n_feedback_terminal?: number;
  pending_feedback?: number;
  n_groups?: number;
  sentinel_level?: SentinelLevel;
  rolling_reward_runner?: Usd;
}

/** GET /simulations/{id}/timeseries/?policy=<name> — rows ordered by epoch, then policy order. */
export interface Timeseries {
  simulation: Id;
  policies: PolicyName[];
  /** Completed epochs only. */
  epochs: Epoch[];
  rows: TimeseriesRow[];
}

export interface SimArmRow extends ArmRow {
  /** Posterior mean reward of the arm (group posterior when `group` is given, else decision-weighted pooled). */
  mu: Usd | null;
  /** Posterior σ (proposal scale: reward dispersion, not the standard error). */
  sigma: Usd | null;
  /** Effective observation count (interim observations weighted by interim_weight). */
  n: number;
  /** Times selected in this epoch. */
  selections: number;
  /** Times selected from the start through this epoch. */
  cum_selections: number;
  share: Fraction;
  cum_share: Fraction;
}

/**
 * GET /simulations/{id}/arms/?policy=thompson&epoch=YYYY-MM&group=<GroupKey>
 * Defaults: policy = first configured, epoch = last completed, group = pooled.
 */
export interface SimArms {
  simulation: Id;
  policy: PolicyName;
  epoch: Epoch;
  group: GroupKey | null;
  n_decisions: number;
  amounts: Usd[];
  rates: Fraction[];
  terms: number[];
  arms: SimArmRow[];
  /** Groups ordered by decision count, for the group picker. */
  groups: { group: GroupKey; n: number }[];
  /** Epoch of the posterior snapshot behind mu / sigma / n (snapshots every 3 epochs). */
  snapshot_epoch?: Epoch | null;
}

export interface PosteriorSnapshotRow {
  epoch: Epoch;
  arm: ArmId;
  mu: Usd;
  sigma: Usd;
  n: number;
}

/**
 * GET /simulations/{id}/posteriors/?policy=thompson&group=<GroupKey>&arms=1,2,3
 * Bandit posterior history (snapshots every `snapshot_every` epochs plus the last epoch).
 * Default arms: the four most-selected lending arms for the group.
 */
export interface PosteriorHistory {
  simulation: Id;
  policy: PolicyName;
  group: GroupKey | null;
  snapshot_every: number;
  arms: ArmRow[];
  rows: PosteriorSnapshotRow[];
}

export interface SimpsonGroupRow {
  group: GroupKey;
  dims: Record<string, string>;
  /** Decisions for this group so far. */
  n: number;
  group_best_arm: ArmId;
  group_best_label: string;
  group_best_mu: Usd;
  /** The pooled-best arm's posterior mean reward inside this group. */
  pooled_arm_mu: Usd;
  /** group_best_mu − pooled_arm_mu (≥ 0). */
  gap: Usd;
  agrees: boolean;
}

/**
 * GET /simulations/{id}/simpson/?policy=thompson&epoch=YYYY-MM
 * Aggregation-bias check: the arm that is best when all borrowers are pooled vs the best arm per group.
 */
export interface SimpsonPanel {
  simulation: Id;
  policy: PolicyName;
  epoch: Epoch;
  pooled_best_arm: ArmId;
  pooled_best_label: string;
  pooled_best_mu: Usd;
  n_groups: number;
  n_agree: number;
  /** Decision-weighted mean gap: reward forfeited per decision by applying the pooled arm everywhere. */
  forfeited_per_decision: Usd;
  /** Ordered by n desc. */
  groups: SimpsonGroupRow[];
}

export type SimulationEventType =
  | 'RunStarted'
  | 'BorrowerArrived'
  | 'PolicySelected'
  | 'FeedbackReleased'
  | 'PosteriorUpdated'
  | 'SentinelEvaluated'
  | 'BreakerTripped'
  | 'ReviewAcknowledged'
  | 'CheckpointWritten'
  | 'EpochCompleted'
  | 'RunPaused'
  | 'RunResumed'
  | 'WorkerKilled'
  | 'RunCompleted'
  | 'RunCancelled'
  | 'RunFailed';

/** One row of the append-only JSONL event store. */
export interface SimulationEvent {
  /** Monotonic per simulation, starting at 1. */
  seq: number;
  type: SimulationEventType;
  epoch: Epoch | null;
  ts: IsoDateTime;
  level: EventLevel;
  /** One-line human summary. */
  message: string;
  payload: Record<string, unknown>;
}

/**
 * GET /simulations/{id}/events/?type=A,B&after=<seq>&limit=100&order=asc|desc
 * Without `type`, the per-borrower types (BorrowerArrived, PolicySelected) are excluded.
 * `after` is exclusive (seq > after). order=desc returns the newest first (log tail).
 */
export interface EventPage {
  simulation: Id;
  results: SimulationEvent[];
  /** Highest seq in the store (use as the next `after`). */
  last_seq: number;
  has_more: boolean;
}

export interface PsiPoint {
  epoch: Epoch;
  feature: string;
  psi: number;
  level: SentinelLevel;
}

export interface PsiAggregatePoint {
  epoch: Epoch;
  psi_max: number;
  psi_mean: number;
  /** psi_max or psi_mean per `aggregate_mode`. */
  aggregate: number;
  level: SentinelLevel;
  breaker_state: BreakerState;
  top_feature: string;
}

export interface BreakerTransition {
  epoch: Epoch;
  ts: IsoDateTime;
  from_state: BreakerState;
  to_state: BreakerState;
  reason: string;
  psi_aggregate: number | null;
  review: Id | null;
  action: ReviewAction | null;
}

/** GET /simulations/{id}/sentinel/ */
export interface SentinelTimeline {
  simulation: Id;
  features: string[];
  thresholds: { watch: number; breaker: number };
  aggregate_mode: 'max' | 'mean';
  window_months: number;
  /** Reference window = fit training window; `rebased_at` lists epochs where resegment/rollback rebased it. */
  reference: { from: Epoch; to: Epoch; n: number; rebased_at: Epoch[] };
  /** One row per feature × completed epoch. */
  features_psi: PsiPoint[];
  /** One row per completed epoch. */
  aggregate: PsiAggregatePoint[];
  transitions: BreakerTransition[];
  current: {
    epoch: Epoch | null;
    breaker_state: BreakerState;
    level: SentinelLevel;
    psi_aggregate: number | null;
    top_feature: string | null;
  };
  /** Frequency of detected shifts (Table 11): epochs at watch / breaker level. */
  shift_counts: { watch: number; breaker: number };
}

export interface PsiBucket {
  /** Bin label, e.g. "[660, 675)" or a category level. */
  bucket: string;
  lo: number | null;
  hi: number | null;
  /** Reference share e_i. */
  expected: Fraction;
  /** Window share a_i. */
  actual: Fraction;
  /** (a_i − e_i)·ln(a_i / e_i) */
  contribution: number;
}

/** GET /simulations/{id}/sentinel/buckets/?feature=<name>&epoch=YYYY-MM */
export interface PsiBucketTable {
  simulation: Id;
  feature: string;
  epoch: Epoch;
  kind: 'numeric' | 'categorical';
  psi: number;
  level: SentinelLevel;
  n_reference: number;
  n_window: number;
  buckets: PsiBucket[];
  /** The runner's own PSI reading for the same feature and epoch. */
  psi_runner?: number | null;
  reference_rebased_at?: Epoch | null;
}

/** GET /sentinel/reviews/?simulation=<id>&status=pending|acknowledged (Paginated, newest first). */
export interface Review {
  id: Id;
  simulation: Id;
  simulation_name: string;
  epoch: Epoch;
  status: ReviewStatus;
  created_at: IsoDateTime;
  psi_aggregate: number;
  level: SentinelLevel;
  breaker_state: BreakerState;
  reason: string;
  top_features: { feature: string; psi: number }[];
  action: ReviewAction | null;
  note: string;
  acknowledged_at: IsoDateTime | null;
  acknowledged_by: string | null;
  /** Acknowledged automatically (breaker_mode = "auto_ack"). */
  auto: boolean;
  segmentation_before: SegmentationDimension[];
  /** Set when action = "resegment" (Segmenter.refine()). */
  segmentation_after: SegmentationDimension[] | null;
}

/** POST /sentinel/reviews/{id}/acknowledge/ → 200 {@link Review}; 409 if already acknowledged. */
export interface AcknowledgeRequest {
  action: ReviewAction;
  note: string;
}

// ─────────────────────────────────────────────────────────────── evaluation report
export type MetricUnit =
  | 'usd'
  | 'usd_per_decision'
  | 'fraction'
  | 'ratio'
  | 'count'
  | 'seconds'
  | 'megabytes'
  | 'nats'
  | 'score';

export interface ReportMetric {
  key: string;
  label: string;
  value: number | null;
  unit: MetricUnit;
  better: 'higher' | 'lower' | null;
  /** Reference value (the historical policy for decision metrics, base rate / climatology for predictive ones). */
  baseline: number | null;
  baseline_label: string | null;
  /** Per-policy values for policy-level metrics. */
  by_policy: Partial<Record<PolicyName, number>> | null;
  note: string | null;
}

/** The nine rows of Table 11. */
export type Table11Dimension =
  | 'default_prediction'
  | 'recovery_prediction'
  | 'credit_state_modeling'
  | 'policy_performance'
  | 'bandit_performance'
  | 'counterfactual_evaluation'
  | 'distributional_monitoring'
  | 'computational_feasibility'
  | 'infrastructure_resilience';

/** G.7 evaluation axes the Table 11 rows roll up to. */
export type EvaluationAxis = 'predictive' | 'decision' | 'adaptability' | 'feasibility';

export interface ReportDimension {
  key: Table11Dimension;
  label: string;
  axis: EvaluationAxis;
  metrics: ReportMetric[];
}

export interface RuntimeStats {
  cpu_seconds: number;
  /** Mean CPU utilisation of the worker (1.0 = one core). */
  cpu_utilization: number;
  peak_memory_mb: number;
  wall_seconds: number;
  epochs: number;
  decisions: number;
  decisions_per_second: number;
  restarts: number;
  checkpoints_written: number;
  failed_workloads: number;
  recovery_seconds: number | null;
  /** One entry per worker session (restarts add sessions). */
  sessions?: { pid: number; started_at: IsoDateTime; wall_seconds: number; cpu_seconds: number; peak_rss_mb: number }[];
}

/** GET /simulations/{id}/report/ — `status` "partial" while the run is not completed. */
export interface EvaluationReport {
  simulation: Id;
  fit: Id;
  dataset: Id;
  generated_at: IsoDateTime;
  status: 'final' | 'partial';
  dimensions: ReportDimension[];
  runtime: RuntimeStats;
  policy_summary: PolicyHeadline[];
  config: { fit: FitConfig; simulation: SimulationConfig };
  /** Raw evaluation blocks keyed by Table 11 dimension (served; typed where the console reads them). */
  details?: ReportDetails;
  config_hashes?: Record<string, string>;
}

export interface BanditPolicyDetail {
  final_cumulative_regret: Usd | null;
  average_regret: Usd | null;
  final_epoch_avg_regret: Usd | null;
  convergence_epoch: Epoch | null;
  converged: boolean;
  /** Share of the top-3 arms in the last 6 epochs. */
  selection_concentration_top3: Fraction | null;
  top_arms_recent: ArmId[];
  top_arm_labels_recent: string[];
  n_distinct_arms: number;
  /** Decisions of the last 6 epochs (`concentration_window_epochs`) by term / rate / amount bucket. */
  selection_mix: SelectionMix | null;
}

/**
 * `EvaluationReport.details.counterfactual_evaluation`. Every `uplift_pct` in here (top level,
 * `uplift_decomposition.*`, `by_risk_aversion.*`) is a **fraction**, like the portfolio endpoint.
 * The console still derives its displayed fraction from `uplift / |expected_profit_historical|`
 * (`lib/counterfactual.ts::upliftFraction`), which is unit-independent.
 */
export interface CounterfactualDetail {
  /** False when no shadow portfolio summary was available to the report. */
  available: boolean;
  note?: string | null;
  /** When available: the portfolio summary scalars (same meaning as `PortfolioSummary`). */
  n_loans?: number;
  n_lendable?: number;
  share_lendable?: Fraction;
  expected_profit_historical?: Usd;
  expected_profit_shadow?: Usd;
  uplift?: Usd;
  /** Percent of |historical expected profit| (core units; see the interface note). */
  uplift_pct?: number;
  uplift_risk_adjusted?: Usd;
  realized_profit_historical?: Usd | null;
  n_realized?: number;
  /** Outcome-class counts plus `n_<class>` flag totals (core naming). */
  counts?: Record<string, number>;
  risk_aversion?: number;
  /** First 10 rows of the portfolio's largest-gap loans. */
  top_loans?: PortfolioLoanRow[];
  recommendation_mix?: RecommendationMix | null;
  uplift_decomposition?: UpliftDecomposition | null;
  by_risk_aversion?: Record<string, RiskAversionPoint> | null;
  eligibility?: PortfolioEligibility | null;
  interest_realization?: InterestRealization | null;
}

export interface ReportDetails {
  bandit_performance?: {
    available: boolean;
    by_policy: Record<string, BanditPolicyDetail>;
    concentration_window_epochs: Epoch[];
    note?: string | null;
  };
  counterfactual_evaluation?: CounterfactualDetail;
  /** Other dimensions' detail blocks (primary / secondary subsets, per-feature PSI, phase timings…). */
  [dimension: string]: unknown;
}
