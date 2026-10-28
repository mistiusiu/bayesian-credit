/** Domain vocabulary shared by the UI and the mock API (labels, defaults, arm grids). */
import type {
  ArmRow,
  ArmSpaceConfig,
  Guardrails,
  HindsightSample,
  KnownPolicy,
  PolicyEffectConfig,
  PolicyEffectFeature,
  BreakerFallback,
  BreakerMode,
  CreditState,
  FeedbackMode,
  FitConfig,
  PolicyName,
  PsiSettings,
  ReviewAction,
  SegmentationDimension,
  Stage,
} from '@/api/types';
import { fmt } from './format';

export const GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] as const;
export const HOME_OWNERSHIP = ['MORTGAGE', 'RENT', 'OWN', 'OTHER'] as const;
export const PURPOSE_GROUPS = ['debt_consolidation', 'credit_card', 'other'] as const;
export const PURPOSES = [
  'debt_consolidation',
  'credit_card',
  'home_improvement',
  'other',
  'major_purchase',
  'medical',
  'small_business',
  'car',
  'moving',
  'vacation',
  'house',
  'wedding',
  'renewable_energy',
] as const;

export const POLICIES: KnownPolicy[] = ['thompson', 'thompson_se', 'ucb1', 'epsilon_greedy', 'greedy', 'historical', 'random'];

export const POLICY_LABEL: Record<KnownPolicy, string> = {
  thompson: 'Thompson',
  thompson_se: 'Thompson (σ/√n)',
  ucb1: 'UCB1',
  epsilon_greedy: 'ε-greedy',
  greedy: 'Greedy',
  historical: 'Historical',
  random: 'Random',
};

export const POLICY_BLURB: Record<KnownPolicy, string> = {
  thompson: 'Contextual Gaussian Thompson sampling, one bandit per borrower group, warm-started from the risk model; samples N(μ, σ²)',
  thompson_se: 'Thompson sampling from N(μ, σ²/n): the standard-error scale, so exploration shrinks as observations accumulate',
  ucb1: 'Upper confidence bound (c = 2) per group',
  epsilon_greedy: 'Greedy with 10% uniform exploration',
  greedy: 'Always the current best posterior mean (no exploration)',
  historical: 'The loan actually issued by Lending Club (reference policy, Table 8)',
  random: 'Uniform over the eligible lending arms',
};

/** Core policy kinds that are valid specs but not offered as form checkboxes. */
const EXTRA_POLICY: Record<string, { label: string; blurb: string }> = {
  thompson_flat: { label: 'Thompson (flat)', blurb: 'Non-contextual Gaussian Thompson sampling: one arm set shared by every borrower group' },
};

/** Base name of a policy spec ("epsilon_greedy(epsilon=0.05)" → "epsilon_greedy"). */
export const policyBase = (p: PolicyName): string => p.split('(')[0].trim();

const isKnown = (p: string): p is KnownPolicy => (POLICIES as string[]).includes(p);

/** Display label for any served policy spec (parameters kept in brackets). */
export function policyLabel(p: PolicyName): string {
  const base = policyBase(p);
  const params = p.includes('(') ? p.slice(p.indexOf('(')) : '';
  const name = isKnown(base) ? POLICY_LABEL[base] : EXTRA_POLICY[base]?.label;
  return name ? `${name}${params ? ` ${params}` : ''}` : p;
}

export function policyBlurb(p: PolicyName): string {
  const base = policyBase(p);
  return isKnown(base) ? POLICY_BLURB[base] : (EXTRA_POLICY[base]?.blurb ?? 'Custom policy spec');
}

/** The policy a single-policy view should default to: plain Thompson, then any Thompson variant, then the first adaptive one. */
export function leadPolicy<P extends PolicyName>(policies: readonly P[]): P | undefined {
  return (
    policies.find((p) => p === 'thompson') ??
    policies.find((p) => policyBase(p).startsWith('thompson')) ??
    policies.find((p) => !['historical', 'random'].includes(policyBase(p))) ??
    policies[0]
  );
}

export const STAGES: Stage[] = ['pre_decision', 'policy', 'disbursement', 'interim', 'terminal'];

export const STAGE_LABEL: Record<Stage, string> = {
  pre_decision: 'Pre-decision',
  policy: 'Policy',
  disbursement: 'Disbursement',
  interim: 'Interim',
  terminal: 'Terminal',
};

export const STAGE_BLURB: Record<Stage, string> = {
  pre_decision: 'Known when the lending decision is made; the only stage the bandit may see',
  policy: 'The lending-policy arm itself: amount, rate, term',
  disbursement: 'Known once the loan is funded',
  interim: 'Observed month by month while the loan is on book (interim rewards, MCTM)',
  terminal: 'Known only at payoff or charge-off; updates posteriors, never a feature',
};

export const CREDIT_STATES: CreditState[] = ['current', 'delinquent', 'recovered', 'loss'];

export const STATE_LABEL: Record<CreditState, string> = {
  current: 'Current',
  delinquent: 'Delinquent',
  recovered: 'Recovered',
  loss: 'Absorbed loss',
};

/** Validated stack order (bottom → top); the top of the delinquent band is P(Default, t) = π_D + π_L. */
export const STATE_STACK: CreditState[] = ['loss', 'delinquent', 'current', 'recovered'];

export const STATE_SHORT: Record<CreditState, string> = { current: 'C', delinquent: 'D', recovered: 'R', loss: 'L' };

export const SEGMENTATION_DIMENSIONS: { key: SegmentationDimension; label: string; source: string }[] = [
  { key: 'grade', label: 'Grade', source: 'grade' },
  { key: 'sub_grade', label: 'Sub-grade', source: 'sub_grade' },
  { key: 'purpose_group', label: 'Purpose group', source: 'purpose' },
  { key: 'home_ownership', label: 'Home ownership', source: 'home_ownership' },
  { key: 'fico_band', label: 'FICO band', source: 'fico_range_low' },
  { key: 'dti_band', label: 'DTI band', source: 'dti' },
  { key: 'income_band', label: 'Income band', source: 'annual_inc' },
  { key: 'emp_band', label: 'Employment band', source: 'emp_length' },
  { key: 'verification_status', label: 'Verification', source: 'verification_status' },
  { key: 'application_type', label: 'Application type', source: 'application_type' },
];

export const REVIEW_ACTIONS: { key: ReviewAction; label: string; blurb: string }[] = [
  { key: 'resume', label: 'Resume', blurb: 'Keep the incumbent policy and unfreeze updates. The reference window is unchanged.' },
  {
    key: 'resegment',
    label: 'Resegment',
    blurb: 'Refine the borrower grouping (Segmenter.refine), refit the risk model on data to date and re-warm-start the bandits.',
  },
  { key: 'rollback', label: 'Rollback', blurb: 'Restore the champion snapshot taken before the drift and rebase the reference window.' },
];

export const FEEDBACK_MODES: { key: FeedbackMode; label: string; blurb: string }[] = [
  { key: 'simulated', label: 'Simulated', blurb: 'Reward drawn from the environment model (posterior predictive + MCTM)' },
  { key: 'replay', label: 'Replay', blurb: 'Rejection-sampling replay: update only when the chosen arm equals the historical one' },
  { key: 'hybrid', label: 'Hybrid', blurb: 'Realized reward when the arms match, simulated otherwise' },
];

export const BREAKER_MODES: { key: BreakerMode; label: string; blurb: string }[] = [
  { key: 'pause', label: 'Pause', blurb: 'Stop and wait for a human acknowledgement' },
  { key: 'auto_ack', label: 'Auto-acknowledge', blurb: 'Apply the configured action after a delay' },
  { key: 'off', label: 'Monitor only', blurb: 'Record PSI, never freeze policies' },
];

export const FALLBACKS: { key: BreakerFallback; label: string }[] = [
  { key: 'champion_snapshot', label: 'Champion snapshot' },
  { key: 'decline', label: 'Decline all' },
  { key: 'historical', label: 'Historical policy' },
];

export const PSI_FEATURES = ['fico_range_low', 'dti', 'annual_inc', 'revol_util', 'inq_last_6mths', 'grade'];

export const DEFAULT_ARM_SPACE: ArmSpaceConfig = {
  amounts: [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
  rates: [0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28],
  terms: [36, 60],
  include_null: true,
  max_payment_to_income: 0.2,
  max_amount_to_income: 0.5,
  max_total_dti: 0.43,
  rate_corridor: 0.05,
};

export const POLICY_EFFECT_FEATURES: { key: PolicyEffectFeature; label: string; step: number; stepLabel: string }[] = [
  { key: 'int_rate', label: 'Interest rate', step: 0.01, stepLabel: '+1 pp rate' },
  { key: 'term60', label: '60-month term', step: 1, stepLabel: '60 vs 36 months' },
  { key: 'pti', label: 'Payment-to-income', step: 0.05, stepLabel: '+5 pp payment-to-income' },
  { key: 'log_amount', label: 'Log amount', step: Math.log(2), stepLabel: 'doubling the amount' },
];

export const DEFAULT_POLICY_EFFECT: PolicyEffectConfig = {
  features: ['int_rate', 'term60', 'pti', 'log_amount'],
  ridge: 0.001,
  max_abs_logit_shift: 3,
  min_rows: 5000,
  min_elapsed_rows: 20000,
  max_iter: 25,
  pti_cap: 1.5,
};

export const DEFAULT_FIT_CONFIG: FitConfig = {
  segmentation: { dimensions: ['grade', 'purpose_group', 'home_ownership'] },
  arm_space: DEFAULT_ARM_SPACE,
  hierarchy: { kappa_group: 20, kappa_leaf: 10, alpha0: 1, beta0: 1, policy_effect: true, policy_effect_cfg: DEFAULT_POLICY_EFFECT },
  recovery: { min_obs: 5, basis: 'principal', term_level: true },
  mctm: { structure: 'proposal', prior_strength: 1, chargeoff_lag_months: 4 },
  reward: { funding_rate: 0.04, risk_aversion: 0, lendable_threshold: 0.6, prepayment_adjustment: 'empirical', exposure_basis: 'full' },
  train_end: '2015-01',
};

export const DEFAULT_HINDSIGHT: HindsightSample = 'elapsed';

/** Where a recovery estimate came from: the core chain leaf → (group, term) → group → global → prior (MATH §2.9). */
export function recoverySourceLabel(r: { source: string; core_source?: string | null } | null | undefined): string {
  const s = r?.core_source ?? r?.source;
  if (!s) return '—';
  return s === 'group_term' ? 'group × term' : s;
}

/** One-line explanation per guardrail (docs/MATH.md §2.7). */
export const GUARDRAIL_INFO: { key: keyof Guardrails; label: string; short: string; unit: '%' | 'pp'; blurb: string }[] = [
  {
    key: 'max_payment_to_income',
    label: 'Payment-to-income cap',
    short: 'Payment / income',
    unit: '%',
    blurb: 'Instalment ≤ x of monthly income: lender convention for unsecured instalment credit (PTI caps of 15–20 %).',
  },
  {
    key: 'max_amount_to_income',
    label: 'Amount-to-income cap',
    short: 'Amount / income',
    unit: '%',
    blurb: 'Loan amount ≤ x of annual income: lender convention that bounds the exposure per borrower.',
  },
  {
    key: 'max_total_dti',
    label: 'Total DTI ceiling',
    short: 'Total DTI',
    unit: '%',
    blurb: 'Existing DTI + the new payment ≤ x: the 43 % General QM limit, 12 CFR 1026.43(e)(2)(vi); LC dti excludes mortgages, so a lower bound.',
  },
  {
    key: 'rate_corridor',
    label: 'Rate corridor',
    short: 'Rate corridor',
    unit: 'pp',
    blurb: '|rate − sub-grade reference rate| ≤ x: offers stay close to the market price of the borrower’s risk band.',
  },
];

/** "≤ 20%" / "± 5 pp" / "off" for one guardrail value. */
export function guardrailValue(v: number | null | undefined, unit: '%' | 'pp'): string {
  if (v == null) return 'off';
  return unit === 'pp' ? `± ${Math.round(v * 1000) / 10} pp` : `≤ ${Math.round(v * 1000) / 10}%`;
}

/** Compact summary of the active guardrails ("PTI ≤ 20 % · amount ≤ 50 % of income · …"). */
export function guardrailSummary(g: Guardrails | null | undefined): string {
  if (!g) return 'no guardrails reported';
  const parts: string[] = [];
  if (g.max_payment_to_income != null) parts.push(`PTI ≤ ${Math.round(g.max_payment_to_income * 100)}%`);
  if (g.max_amount_to_income != null) parts.push(`amount ≤ ${Math.round(g.max_amount_to_income * 100)}% of income`);
  if (g.max_total_dti != null) parts.push(`total DTI ≤ ${Math.round(g.max_total_dti * 100)}%`);
  if (g.rate_corridor != null) parts.push(`rate within ±${Math.round(g.rate_corridor * 100)} pp of reference`);
  return parts.length ? parts.join(' · ') : 'all guardrails off';
}

export const DEFAULT_PSI: PsiSettings = {
  features: PSI_FEATURES,
  window_months: 3,
  thresholds: { watch: 0.1, breaker: 0.25 },
  aggregate: 'max',
  recovery_readings: 3,
  fallback: 'champion_snapshot',
  auto_action: 'resume',
  auto_ack_epochs: 1,
};

export interface ArmGrid {
  amounts: number[];
  rates: number[];
  terms: number[];
  arms: ArmRow[];
}

/** Arm 0 = decline (if included), then term-major, amount, rate — same order as core ArmSpace.from_grid. */
export function buildArmGrid(cfg: Pick<ArmSpaceConfig, 'amounts' | 'rates' | 'terms' | 'include_null'>): ArmGrid {
  const amounts = [...new Set(cfg.amounts)].sort((a, b) => a - b);
  const rates = [...new Set(cfg.rates)].sort((a, b) => a - b);
  const terms = [...new Set(cfg.terms)].sort((a, b) => a - b);
  const arms: ArmRow[] = [];
  if (cfg.include_null) {
    arms.push({ arm: 0, amount: 0, rate: 0, term: 0, is_null: true, installment: 0, label: 'decline' });
  }
  for (const term of terms) {
    for (const amount of amounts) {
      for (const rate of rates) {
        const m = rate / 12;
        const f = (1 + m) ** term;
        const inst = (amount * m * f) / (f - 1);
        arms.push({
          arm: arms.length,
          amount,
          rate,
          term,
          is_null: false,
          installment: Math.round(inst * 100) / 100,
          label: `$${amount / 1000}k @ ${Math.round(rate * 100)}% / ${term}m`,
        });
      }
    }
  }
  return { amounts, rates, terms, arms };
}

/** Human label for an arm row (uses the shared formatter). */
export function armLabel(a: Pick<ArmRow, 'amount' | 'rate' | 'term' | 'is_null'> | null | undefined): string {
  return fmt.arm(a);
}

/** Split a group key into its dimension values using the fit's dimension order. */
export function groupDims(key: string, dimensions: readonly string[]): Record<string, string> {
  const parts = key.split('|');
  const out: Record<string, string> = {};
  dimensions.forEach((d, i) => {
    out[d] = parts[i] ?? '';
  });
  return out;
}
