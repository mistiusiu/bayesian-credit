/** Shadow lending engine mock (SPEC §2.10): counterfactual arms per borrower + portfolio evaluation. */
import type { BorrowerContext, LoanOutcome, SegmentationDimension, ShadowArmRow, ShadowReport } from '@/api/types';
import { economics, rewardMoments, timedInterest } from './finance';
import { Rng, seedOf } from './prng';
import { type FitModel, eligibility, groupArmReward, leafPosterior, leafRecovery, realizationOf, riskAdjusted } from './riskModel';
import { betaHdiApprox, round } from './stats';

const PRE_DECISION_KEYS: (keyof BorrowerContext)[] = [
  'annual_inc', 'annual_inc_joint', 'application_type', 'dti', 'dti_joint', 'emp_length', 'fico_range_low',
  'fico_range_high', 'home_ownership', 'purpose', 'verification_status', 'verified_status_joint', 'delinq_2yrs',
  'delinq_amnt', 'inq_last_6mths', 'inq_last_12m', 'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'installment',
  'mort_acc', 'pub_rec', 'pub_rec_bankruptcies', 'collections_12_mths_ex_med', 'num_accts_ever_120_pd', 'pct_tl_nvr_dlq',
  'num_sats', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'bc_util',
  'all_util', 'grade', 'sub_grade',
];

export const isPreDecisionKey = (k: string): k is keyof BorrowerContext => (PRE_DECISION_KEYS as string[]).includes(k);

/** Full context with every pre-decision key present (missing → null). */
export function completeContext(partial: Partial<BorrowerContext>): BorrowerContext {
  const out = {} as Record<keyof BorrowerContext, string | number | null>;
  for (const k of PRE_DECISION_KEYS) out[k] = partial[k] ?? null;
  return out as unknown as BorrowerContext;
}

const band = (v: number | null, edges: number[], labels: string[]) => {
  if (v === null) return 'unknown';
  for (let k = 0; k < edges.length; k++) if (v < edges[k]) return labels[k];
  return labels[labels.length - 1];
};

export function groupFromContext(c: BorrowerContext, dims: SegmentationDimension[]): string {
  const purpose = (c.purpose ?? '').toLowerCase();
  const home = (c.home_ownership ?? '').toUpperCase();
  const value = (d: SegmentationDimension): string => {
    switch (d) {
      case 'grade':
        return c.grade ?? 'unknown';
      case 'sub_grade':
        return c.sub_grade ?? 'unknown';
      case 'purpose_group':
        return purpose === 'debt_consolidation' || purpose === 'credit_card' ? purpose : 'other';
      case 'home_ownership':
        return ['RENT', 'OWN', 'MORTGAGE'].includes(home) ? home : 'OTHER';
      case 'fico_band':
        return band(c.fico_range_low, [660, 700, 740, 780], ['<660', '660-699', '700-739', '740-779', '780+']);
      case 'dti_band':
        return band(c.dti, [10, 20, 30], ['<10', '10-20', '20-30', '30+']);
      case 'income_band':
        return band(c.annual_inc, [40000, 75000, 120000], ['<40k', '40-75k', '75-120k', '120k+']);
      case 'emp_band':
        return band(c.emp_length, [2, 6, 10], ['<2', '2-5', '6-9', '10+']);
      case 'verification_status':
        return c.verification_status ?? 'unknown';
      case 'application_type':
        return c.application_type ?? 'unknown';
    }
  };
  return dims.map(value).join('|');
}

const N_SAMPLES = 1000;

function armRows(model: FitModel, group: string, mask: boolean[], rng: Rng): ShadowArmRow[] {
  const f = model.config.reward.funding_rate;
  return model.grid.arms.map((a) => {
    if (a.is_null) {
      return {
        ...a, eligible: true, realization: 1, p_default_mean: 0, p_default_hdi_low: 0, p_default_hdi_high: 0, expected_recovery: 0,
        reward_mean: 0, reward_std: 0, reward_q05: 0, reward_q50: 0, reward_q95: 0, prob_positive: 0, risk_adjusted: 0,
      };
    }
    const post = leafPosterior(model, group, a.arm);
    const rec = leafRecovery(model, group, a.arm);
    const r = groupArmReward(model, group, a.arm);
    const rho = realizationOf(model, group, a.arm);
    const e = economics(a.amount, a.rate, a.term);
    const funding = rho * f * a.amount * e.walYears;
    const base = rho * e.totalInterest - funding;
    const B = rho * timedInterest(a.amount, a.rate, a.term) + a.amount;
    // core CompoundRewardFunction.sample: Reward = ρ·a − p·((ρ·b + c) − c·R) over posterior draws of p and R
    const draws = new Float64Array(N_SAMPLES);
    let positive = 0;
    for (let s = 0; s < N_SAMPLES; s++) {
      const p = rng.beta(post.alpha, post.beta);
      const R = rng.beta(rec.alpha, rec.beta);
      const reward = base - p * (B - a.amount * R);
      draws[s] = reward;
      if (reward > 0) positive++;
    }
    draws.sort();
    const [lo, hi] = betaHdiApprox(post.alpha, post.beta);
    return {
      ...a,
      eligible: mask[a.arm],
      realization: round(rho, 6),
      p_default_mean: round(post.alpha / (post.alpha + post.beta), 5),
      p_default_hdi_low: round(lo, 5),
      p_default_hdi_high: round(hi, 5),
      expected_recovery: round(rec.mean, 4),
      reward_mean: round(r.mean, 2),
      reward_std: round(r.std, 2),
      reward_q05: round(draws[Math.floor(0.05 * N_SAMPLES)], 2),
      reward_q50: round(draws[Math.floor(0.5 * N_SAMPLES)], 2),
      reward_q95: round(draws[Math.floor(0.95 * N_SAMPLES)], 2),
      prob_positive: round(positive / N_SAMPLES, 4),
      risk_adjusted: round(riskAdjusted(model, r), 2),
    };
  });
}

export function evaluateShadow(
  model: FitModel,
  fitId: number,
  input: { loanId: number | null; issue: string | null; context: BorrowerContext; policy: { amount: number; rate: number; term: number } | null; outcome: LoanOutcome | null },
): ShadowReport {
  const group = groupFromContext(input.context, model.config.segmentation.dimensions);
  const rng = new Rng(seedOf('shadow', fitId, input.loanId ?? JSON.stringify(input.context)));
  const histArm = input.policy ? model.armOf(input.policy.amount, input.policy.rate, input.policy.term) : null;
  const elig = eligibility(model, { income: input.context.annual_inc, dti: input.context.dti, subGrade: input.context.sub_grade, grade: input.context.grade }, histArm);
  const arms = armRows(model, group, elig.mask, rng);
  let rec = 0;
  for (const a of arms) if (a.eligible && a.risk_adjusted > arms[rec].risk_adjusted) rec = a.arm;
  let histExpected: number | null = null;
  let histArmExpected: number | null = null;
  if (input.policy && histArm !== null) {
    const post = leafPosterior(model, group, histArm);
    const recov = leafRecovery(model, group, histArm);
    histExpected = rewardMoments(
      input.policy.amount, input.policy.rate, input.policy.term,
      post.alpha / (post.alpha + post.beta), recov.mean, recov.variance, model.config.reward.funding_rate, realizationOf(model, group, histArm),
    ).mean;
    histArmExpected = groupArmReward(model, group, histArm).mean;
  }
  const realized = input.outcome?.realized_reward ?? null;
  const best = arms[rec];
  const threshold = model.config.reward.lendable_threshold;
  const a = model.config.arm_space;
  return {
    fit: fitId,
    loan_id: input.loanId,
    issue_d: input.issue,
    group,
    context: input.context,
    context_columns: PRE_DECISION_KEYS.filter((k) => input.context[k] !== null),
    leakage_safe: true,
    n_samples: N_SAMPLES,
    risk_aversion: model.config.reward.risk_aversion,
    lendable_threshold: threshold,
    max_payment_to_income: a.max_payment_to_income ?? 0,
    arms,
    recommended_arm: rec,
    historical_policy: input.policy,
    historical_arm: histArm,
    historical_expected_reward: histExpected === null ? null : round(histExpected, 2),
    historical_realized_reward: realized,
    uplift_expected: histExpected === null ? null : round(best.reward_mean - histExpected, 2),
    uplift_vs_realized: realized === null ? null : round(best.reward_mean - realized, 2),
    lendable: arms.some((x) => !x.is_null && x.eligible && x.prob_positive >= threshold),
    outcome: input.outcome,
    eligibility: {
      reference_rate: elig.refRate === null ? null : round(elig.refRate, 6),
      n_eligible: elig.mask.filter(Boolean).length,
      n_eligible_lending: elig.nEligibleLending,
      historical_admitted: elig.historicalAdmitted,
      guardrails: { max_payment_to_income: a.max_payment_to_income, rate_corridor: a.rate_corridor, max_amount_to_income: a.max_amount_to_income, max_total_dti: a.max_total_dti },
      rate_corridor_applied: a.rate_corridor !== null && elig.refRate !== null,
    },
    recommended_policy: best.is_null ? null : { amount: best.amount, rate: best.rate, term: best.term },
    recommended_label: best.label,
    declined: best.is_null,
    historical_arm_expected_reward: histArmExpected === null ? null : round(histArmExpected, 2),
  };
}

