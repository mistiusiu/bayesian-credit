/** Table 3 metadata and the Lending Club-like distributions the mock generator draws from. */
import type { Stage, VariableKind } from '@/api/types';

export interface VariableMeta {
  name: string;
  stage: Stage;
  primary_use: string;
  kind: VariableKind;
  dtype: string;
  /** Typical null share in Lending Club 2012–2018. */
  null_share: number;
}

const v = (name: string, stage: Stage, primary_use: string, kind: VariableKind, null_share = 0): VariableMeta => ({
  name,
  stage,
  primary_use,
  kind,
  dtype: kind === 'categorical' ? 'string' : kind === 'date' ? 'datetime64[ns]' : name === 'term' ? 'int64' : 'float64',
  null_share,
});

/** Table 3 — mirrors core/bayesian_credit/data/schema.py RETAINED_VARIABLES. */
export const TABLE3: VariableMeta[] = [
  v('annual_inc', 'pre_decision', 'Borrower context / risk estimation', 'numeric'),
  v('annual_inc_joint', 'pre_decision', 'Joint-borrower risk estimation', 'numeric', 0.947),
  v('application_type', 'pre_decision', 'Borrower context', 'categorical'),
  v('dti', 'pre_decision', 'Risk estimation', 'numeric', 0.0008),
  v('dti_joint', 'pre_decision', 'Joint-borrower risk estimation', 'numeric', 0.947),
  v('emp_length', 'pre_decision', 'Borrower context', 'numeric', 0.065),
  v('fico_range_low', 'pre_decision', 'Credit-risk estimation', 'numeric'),
  v('fico_range_high', 'pre_decision', 'Credit-risk estimation', 'numeric'),
  v('home_ownership', 'pre_decision', 'Borrower context', 'categorical'),
  v('purpose', 'pre_decision', 'Loan context', 'categorical'),
  v('verification_status', 'pre_decision', 'Information-quality assessment', 'categorical'),
  v('verified_status_joint', 'pre_decision', 'Joint-borrower information quality', 'categorical', 0.949),
  v('delinq_2yrs', 'pre_decision', 'Historical credit risk', 'numeric'),
  v('delinq_amnt', 'pre_decision', 'Credit risk', 'numeric'),
  v('inq_last_6mths', 'pre_decision', 'Credit-seeking behavior', 'numeric'),
  v('inq_last_12m', 'pre_decision', 'Credit-seeking behavior', 'numeric', 0.383),
  v('open_acc', 'pre_decision', 'Credit profile', 'numeric'),
  v('total_acc', 'pre_decision', 'Credit history', 'numeric'),
  v('revol_bal', 'pre_decision', 'Credit exposure', 'numeric'),
  v('revol_util', 'pre_decision', 'Credit utilization', 'numeric', 0.0008),
  v('installment', 'pre_decision', 'Repayment burden', 'numeric'),
  v('mort_acc', 'pre_decision', 'Credit profile', 'numeric', 0.022),
  v('pub_rec', 'pre_decision', 'Credit risk', 'numeric'),
  v('pub_rec_bankruptcies', 'pre_decision', 'Credit risk', 'numeric', 0.0006),
  v('collections_12_mths_ex_med', 'pre_decision', 'Credit risk', 'numeric'),
  v('num_accts_ever_120_pd', 'pre_decision', 'Historical severe delinquency', 'numeric', 0.031),
  v('pct_tl_nvr_dlq', 'pre_decision', 'Credit behavior', 'numeric', 0.031),
  v('num_sats', 'pre_decision', 'Credit profile', 'numeric', 0.026),
  v('tot_coll_amt', 'pre_decision', 'Historical credit risk', 'numeric', 0.031),
  v('tot_cur_bal', 'pre_decision', 'Current credit exposure', 'numeric', 0.031),
  v('tot_hi_cred_lim', 'pre_decision', 'Credit capacity', 'numeric', 0.031),
  v('total_bal_ex_mort', 'pre_decision', 'Debt exposure', 'numeric', 0.022),
  v('total_bc_limit', 'pre_decision', 'Revolving credit capacity', 'numeric', 0.022),
  v('bc_util', 'pre_decision', 'Bankcard utilization', 'numeric', 0.033),
  v('all_util', 'pre_decision', 'Overall credit utilization', 'numeric', 0.383),
  v('grade', 'pre_decision', 'Credit-risk context', 'categorical'),
  v('sub_grade', 'pre_decision', 'Credit-risk context', 'categorical'),
  v('loan_amnt', 'policy', 'Lending-policy arm', 'numeric'),
  v('int_rate', 'policy', 'Lending-policy arm', 'numeric'),
  v('term', 'policy', 'Lending-policy characteristic', 'numeric'),
  v('funded_amnt', 'disbursement', 'Actual loan exposure', 'numeric'),
  v('issue_d', 'disbursement', 'Loan origination and time index', 'date'),
  v('last_fico_range_low', 'interim', 'Updated borrower risk', 'numeric'),
  v('last_fico_range_high', 'interim', 'Updated borrower risk', 'numeric'),
  v('last_pymnt_amnt', 'interim', 'Interim repayment reward', 'numeric'),
  v('last_pymnt_d', 'interim', 'Repayment timing', 'date', 0.001),
  v('out_prncp', 'interim', 'Remaining exposure', 'numeric'),
  v('out_prncp_inv', 'interim', 'Remaining investor exposure', 'numeric'),
  v('num_tl_30dpd', 'interim', 'Current delinquency', 'numeric', 0.031),
  v('num_tl_90g_dpd_24m', 'interim', 'Evolving credit risk', 'numeric', 0.031),
  v('mths_since_last_delinq', 'interim', 'Evolving delinquency risk', 'numeric', 0.51),
  v('next_pymnt_d', 'interim', 'Scheduled repayment information', 'date', 0.56),
  v('loan_status', 'terminal', 'Default and credit-state outcome', 'categorical'),
  v('recoveries', 'terminal', 'Recovery estimation', 'numeric'),
  v('collection_recovery_fee', 'terminal', 'Recovery/revenue calculation', 'numeric'),
  v('total_pymnt', 'terminal', 'Total repayment', 'numeric'),
  v('total_pymnt_inv', 'terminal', 'Investor repayment', 'numeric'),
  v('total_rec_prncp', 'terminal', 'Principal recovered', 'numeric'),
  v('total_rec_int', 'terminal', 'Interest revenue', 'numeric'),
  v('total_rec_late_fee', 'terminal', 'Late-fee revenue', 'numeric'),
];

export const GRADE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
export const SUB_GRADES = GRADE_LETTERS.flatMap((g) => [1, 2, 3, 4, 5].map((s) => `${g}${s}`));

/** Grade mix in 2012 and 2018; interpolated by issue year. */
export const GRADE_MIX_2012 = [0.18, 0.33, 0.24, 0.14, 0.07, 0.03, 0.01];
export const GRADE_MIX_2018 = [0.25, 0.3, 0.28, 0.12, 0.035, 0.01, 0.005];

/** Lifetime default probability anchor per grade (36-month, reference borrower). */
export const PD_GRADE = [0.03, 0.066, 0.105, 0.145, 0.183, 0.21, 0.23];

/** Interest rate as a function of sub-grade index 0..34 (A1 ≈ 5.3% → G5 ≈ 30%). */
export const rateForSub = (sub: number) => 0.053 + 0.0072 * sub;
/** Typical (mean) rate of a grade: the centre sub-grade. */
export const gradeRate = (grade: number) => rateForSub(grade * 5 + 2);

export const PURPOSE_KEYS = [
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
];
export const PURPOSE_SHARE = [0.58, 0.22, 0.065, 0.058, 0.022, 0.012, 0.011, 0.01, 0.007, 0.007, 0.005, 0.002, 0.001];
export const PURPOSE_PD_MULT = [1.0, 0.88, 1.02, 1.12, 1.0, 1.12, 1.45, 0.92, 1.18, 1.05, 1.1, 0.95, 1.2];
export const purposeGroupOf = (purpose: number) => (purpose === 0 ? 'debt_consolidation' : purpose === 1 ? 'credit_card' : 'other');

export const HOME_KEYS = ['MORTGAGE', 'RENT', 'OWN', 'OTHER'];
export const HOME_SHARE = [0.49, 0.4, 0.108, 0.002];
export const HOME_PD_MULT = [0.88, 1.12, 0.98, 1.25];
export const HOME_INC_MULT = [1.2, 0.85, 0.95, 0.8];

export const VERIF_KEYS = ['Not Verified', 'Source Verified', 'Verified'];

/** Lending Club loan_status vocabulary in display order. */
export const STATUS_KEYS = [
  'Fully Paid',
  'Current',
  'Charged Off',
  'Late (31-120 days)',
  'In Grace Period',
  'Late (16-30 days)',
  'Default',
];
export const ST = { FULLY_PAID: 0, CURRENT: 1, CHARGED_OFF: 2, LATE_31_120: 3, GRACE: 4, LATE_16_30: 5, DEFAULT: 6 } as const;
export const RESOLVED_STATUS = new Set<number>([ST.FULLY_PAID, ST.CHARGED_OFF, ST.DEFAULT]);
export const DEFAULTED_STATUS = new Set<number>([ST.CHARGED_OFF, ST.DEFAULT]);

/** Average monthly volume per issue year (full Lending Club), used for the arrival ramp. */
export const YEAR_VOLUME: Record<number, number> = {
  2007: 50,
  2008: 200,
  2009: 450,
  2010: 1000,
  2011: 1800,
  2012: 4400,
  2013: 11000,
  2014: 19600,
  2015: 35000,
  2016: 36100,
  2017: 36900,
  2018: 41300,
};

/** Snapshot month: issue_d ends 2018-12, last_pymnt_d runs to 2019-03. */
export const SNAPSHOT = '2019-03';
