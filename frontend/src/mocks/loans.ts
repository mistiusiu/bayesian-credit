/**
 * Synthetic Lending Club twin for the mock API. `buildIndex` draws every loan once into
 * compact typed arrays (deterministic in the dataset seed); dataset statistics, fits,
 * calibration and the loan browser are all computed from the same arrays, so every
 * number the console shows agrees with every other.
 */
import type { BorrowerContext, DatasetCreate, LoanOutcome, LoanRecord } from '@/api/types';
import { epochAdd, epochDiff, epochRange } from '@/lib/epoch';
import { balanceAfter, balanceSum, installment, interestThrough } from './finance';
import { Rng, seedOf } from './prng';
import { clamp } from './stats';
import {
  DEFAULTED_STATUS,
  GRADE_LETTERS,
  GRADE_MIX_2012,
  GRADE_MIX_2018,
  HOME_INC_MULT,
  HOME_KEYS,
  HOME_PD_MULT,
  HOME_SHARE,
  PD_GRADE,
  PURPOSE_KEYS,
  PURPOSE_PD_MULT,
  PURPOSE_SHARE,
  RESOLVED_STATUS,
  SNAPSHOT,
  ST,
  STATUS_KEYS,
  SUB_GRADES,
  VERIF_KEYS,
  YEAR_VOLUME,
  purposeGroupOf,
  rateForSub,
} from './vocab';

export const FICO_MEAN = [738, 706, 692, 683, 677, 672, 669];
const FICO_SD = [32, 26, 22, 20, 18, 16, 15];
export const DTI_MEAN = [15.2, 17.6, 19.1, 20.4, 21.6, 22.4, 23.0];
const INC_GRADE = [1.25, 1.12, 1.0, 0.97, 0.94, 0.92, 0.9];
const TERM60_GRADE = [0.35, 0.6, 1.0, 1.4, 2.0, 2.4, 2.6];
const EMP_W = [0.08, 0.07, 0.09, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.33];
/** Adverse-selection slope: relative default risk per unit of rate above the sub-grade price. */
export const KAPPA_RATE = 18;
export const FUNDING_RATE = 0.04;
/** Monthly prepayment hazard of performing loans. */
const PREPAY_HAZARD = 0.0125;

/** Adverse selection: relative default risk of pricing Δ above (steep) or below (mild) the market rate. */
export const rateRisk = (delta: number) => Math.exp(KAPPA_RATE * Math.max(0, delta) + 6 * Math.min(0, delta));
/** Payment-to-income stress multiplier on default risk. */
export const ptiStress = (pti: number) => Math.exp(4 * Math.max(0, pti - 0.06));
/** Relative lifetime default risk of a 60-month term. */
export const TERM60_RISK = 1.7;

export interface LoanIndex {
  n: number;
  seed: number;
  epochs: string[];
  monthCounts: number[];
  issue: Uint16Array;
  grade: Uint8Array;
  sub: Uint8Array;
  purpose: Uint8Array;
  home: Uint8Array;
  term60: Uint8Array;
  status: Uint8Array;
  amount: Float32Array;
  rate: Float32Array;
  fico: Uint16Array;
  dti: Float32Array;
  income: Float32Array;
  /** 0..10, 255 = missing */
  emp: Uint8Array;
  verif: Uint8Array;
  joint: Uint8Array;
  /** month of default (1..T), 0 = performs */
  defMonth: Uint8Array;
  /** payoff month (prepayment or maturity) */
  payMonth: Uint8Array;
  recov: Float32Array;
  lateFee: Uint8Array;
  pTrue: Float32Array;
}

export const loanIdOf = (i: number) => 1_000_000 + i * 11 + ((i * 7919) % 11);
export const termOf = (idx: LoanIndex, i: number) => (idx.term60[i] ? 60 : 36);

/** Monthly arrival counts following Lending Club's volume ramp (largest-remainder allocation to n). */
function arrivalCounts(epochs: string[], n: number, rng: Rng): number[] {
  const w = epochs.map((e) => {
    const [y, m] = e.split('-').map(Number);
    const lo = YEAR_VOLUME[y] ?? 41300;
    const hi = YEAR_VOLUME[y + 1] ?? lo * 1.05;
    const level = lo + ((hi - lo) * (m - 1)) / 12;
    const season = 1 + 0.05 * Math.sin((2 * Math.PI * (m - 3)) / 12);
    return Math.max(1, level * season * (1 + 0.04 * rng.normal()));
  });
  const total = w.reduce((a, b) => a + b, 0);
  const raw = w.map((x) => (x / total) * n);
  const counts = raw.map(Math.floor);
  let rest = n - counts.reduce((a, b) => a + b, 0);
  const order = raw.map((x, i) => [x - Math.floor(x), i] as const).sort((a, b) => b[0] - a[0]);
  for (let k = 0; rest > 0; k = (k + 1) % order.length, rest--) counts[order[k][1]]++;
  return counts;
}

export function buildIndex(params: DatasetCreate, nominalN: number): LoanIndex {
  const seed = params.seed ?? 42;
  const n = Math.max(1000, Math.min(nominalN, 2_500_000));
  const rng = new Rng(seedOf('index', params.source, seed));
  const epochs = epochRange(params.issue_from ?? '2012-01', params.issue_to ?? '2018-12');
  const monthCounts = arrivalCounts(epochs, n, rng);
  const shift = params.source === 'synthetic' ? params.shift ?? null : null;
  const d = shift?.deltas ?? {};

  const idx: LoanIndex = {
    n,
    seed,
    epochs,
    monthCounts,
    issue: new Uint16Array(n),
    grade: new Uint8Array(n),
    sub: new Uint8Array(n),
    purpose: new Uint8Array(n),
    home: new Uint8Array(n),
    term60: new Uint8Array(n),
    status: new Uint8Array(n),
    amount: new Float32Array(n),
    rate: new Float32Array(n),
    fico: new Uint16Array(n),
    dti: new Float32Array(n),
    income: new Float32Array(n),
    emp: new Uint8Array(n),
    verif: new Uint8Array(n),
    joint: new Uint8Array(n),
    defMonth: new Uint8Array(n),
    payMonth: new Uint8Array(n),
    recov: new Float32Array(n),
    lateFee: new Uint8Array(n),
    pTrue: new Float32Array(n),
  };

  let i = 0;
  for (let e = 0; e < epochs.length; e++) {
    const epoch = epochs[e];
    const year = Number(epoch.slice(0, 4));
    const frac = clamp((year - 2012) / 6, 0, 1);
    const mix = GRADE_MIX_2012.map((a, k) => a + (GRADE_MIX_2018[k] - a) * frac);
    const mob = epochDiff(epoch, SNAPSHOT);
    const shifted = shift !== null && epoch >= shift.at;
    const jointAllowed = epoch >= '2015-10';
    const maxAmount = epoch >= '2016-03' ? 40000 : 35000;
    for (let c = 0; c < monthCounts[e]; c++, i++) {
      const g = rng.weighted(mix);
      const s = rng.weighted([0.24, 0.22, 0.2, 0.18, 0.16]);
      const sub = g * 5 + s;
      const purpose = rng.weighted(PURPOSE_SHARE);
      const home = rng.weighted(HOME_SHARE);
      const t60 = rng.bernoulli(clamp((0.2 + 0.1 * frac) * TERM60_GRADE[g], 0, 0.85));
      const T = t60 ? 60 : 36;
      const rate = clamp(rateForSub(sub) + rng.normal(0, 0.0035) + (year <= 2013 ? 0.004 : 0), 0.05, 0.31);
      let income = rng.lognormal(Math.log(62000), 0.55) * INC_GRADE[g] * HOME_INC_MULT[home];
      if (shifted && d.annual_inc) income += d.annual_inc;
      income = Math.round(clamp(income, 6000, 1_500_000) / 100) * 100;
      const amtRaw = rng.lognormal(Math.log(11500) + 0.3 * Math.log(income / 62000), 0.55) * (t60 ? 1.35 : 1);
      const amount = Math.round(clamp(amtRaw, 1000, maxAmount) / 25) * 25;
      let fico = FICO_MEAN[g] + rng.normal(0, FICO_SD[g]);
      if (shifted && d.fico_range_low) fico += d.fico_range_low;
      fico = clamp(5 * Math.round(fico / 5), shifted ? 600 : 660, 845);
      let dti = rng.normal(DTI_MEAN[g], 8);
      if (shifted && d.dti) dti += d.dti;
      dti = Math.round(clamp(dti, 0, 49.9) * 100) / 100;
      const emp = rng.next() < 0.065 ? 255 : rng.weighted(EMP_W);
      const verif = rng.weighted([0.45 - 0.05 * g, 0.33 + 0.015 * g, 0.22 + 0.035 * g]);
      const joint = jointAllowed && rng.bernoulli(0.075) ? 1 : 0;

      const pay = installment(amount, rate, T);
      const pti = pay / (income / 12);
      const p = clamp(
        PD_GRADE[g] *
          Math.exp(0.12 * (s - 2)) *
          PURPOSE_PD_MULT[purpose] *
          HOME_PD_MULT[home] *
          (t60 ? TERM60_RISK : 1) *
          rateRisk(rate - rateForSub(sub)) *
          ptiStress(pti) *
          Math.exp(-0.012 * (fico - FICO_MEAN[g])) *
          Math.exp(0.02 * (dti - DTI_MEAN[g])),
        0.003,
        0.85,
      );

      const defaulted = rng.bernoulli(p);
      const u = rng.next();
      const geo = Math.ceil(Math.log(1 - u) / Math.log(1 - PREPAY_HAZARD));
      const payMonth = Math.min(Math.max(geo, 1), T);
      let defMonth = 0;
      let status: number;
      let recov = 0;
      let lateFee = 0;
      if (defaulted) {
        defMonth = Math.max(1, Math.ceil(T * rng.beta(1.4, 2.6)));
        const since = mob - defMonth;
        if (since >= 4) status = rng.next() < 0.0004 ? ST.DEFAULT : ST.CHARGED_OFF;
        else if (since >= 2) status = ST.LATE_31_120;
        else if (since === 1) status = rng.next() < 0.5 ? ST.GRACE : ST.LATE_16_30;
        else status = ST.CURRENT;
        const bal = balanceAfter(amount, rate, T, defMonth - 1);
        const recovered = status === ST.CHARGED_OFF || status === ST.DEFAULT ? rng.beta(1.2, 6) * bal * 0.82 : 0;
        recov = clamp((amount - bal + recovered) / amount, 0, 1);
      } else if (payMonth <= mob) {
        status = ST.FULLY_PAID;
        lateFee = rng.next() < 0.08 ? 1 : 0;
      } else {
        const r = rng.next();
        status = r < 0.004 ? ST.GRACE : r < 0.006 ? ST.LATE_16_30 : r < 0.01 ? ST.LATE_31_120 : ST.CURRENT;
      }

      idx.issue[i] = e;
      idx.grade[i] = g;
      idx.sub[i] = sub;
      idx.purpose[i] = purpose;
      idx.home[i] = home;
      idx.term60[i] = t60 ? 1 : 0;
      idx.status[i] = status;
      idx.amount[i] = amount;
      idx.rate[i] = rate;
      idx.fico[i] = fico;
      idx.dti[i] = dti;
      idx.income[i] = income;
      idx.emp[i] = emp;
      idx.verif[i] = verif;
      idx.joint[i] = joint;
      idx.defMonth[i] = defMonth;
      idx.payMonth[i] = payMonth;
      idx.recov[i] = recov;
      idx.lateFee[i] = lateFee;
      idx.pTrue[i] = p;
    }
  }
  return idx;
}

export const isResolved = (idx: LoanIndex, i: number) => RESOLVED_STATUS.has(idx.status[i]);
export const isDefault = (idx: LoanIndex, i: number) => DEFAULTED_STATUS.has(idx.status[i]);
export const groupKeyDefault = (idx: LoanIndex, i: number) =>
  `${GRADE_LETTERS[idx.grade[i]]}|${purposeGroupOf(idx.purpose[i])}|${HOME_KEYS[idx.home[i]]}`;

export interface LoanQuery {
  grade?: string[];
  purpose?: string[];
  home_ownership?: string[];
  resolved?: boolean;
  search?: string;
}

/** Indices matching the filters, ascending (= loan_id order). */
export function filterLoans(idx: LoanIndex, q: LoanQuery): number[] {
  const grades = q.grade?.length ? new Set(q.grade.map((g) => GRADE_LETTERS.indexOf(g))) : null;
  const purposes = q.purpose?.length ? new Set(q.purpose.map((p) => PURPOSE_KEYS.indexOf(p))) : null;
  const homes = q.home_ownership?.length ? new Set(q.home_ownership.map((h) => HOME_KEYS.indexOf(h))) : null;
  const search = q.search?.trim() ?? '';
  const out: number[] = [];
  for (let i = 0; i < idx.n; i++) {
    if (grades && !grades.has(idx.grade[i])) continue;
    if (purposes && !purposes.has(idx.purpose[i])) continue;
    if (homes && !homes.has(idx.home[i])) continue;
    if (q.resolved !== undefined && isResolved(idx, i) !== q.resolved) continue;
    if (search && !String(loanIdOf(i)).includes(search)) continue;
    out.push(i);
  }
  return out;
}

export function indexOfLoanId(idx: LoanIndex, loanId: number): number | null {
  // ids are increasing: base + 11 i + r, r ∈ [0, 10]
  const i = Math.floor((loanId - 1_000_000) / 11);
  return i >= 0 && i < idx.n && loanIdOf(i) === loanId ? i : null;
}

/** Full LC-schema record of loan i; secondary bureau fields drawn from a per-loan stream. */
export function loanRecord(idx: LoanIndex, i: number, nearest: (a: number, r: number, t: number) => number): LoanRecord {
  const rng = new Rng(seedOf('row', idx.seed, i));
  const g = idx.grade[i];
  const T = termOf(idx, i);
  const epoch = idx.epochs[idx.issue[i]];
  const amount = idx.amount[i];
  const rate = Math.round(idx.rate[i] * 10000) / 10000;
  const income = idx.income[i];
  const joint = idx.joint[i] === 1;
  const home = idx.home[i];
  const recentBureau = epoch >= '2015-12';
  const openAcc = rng.poisson(10) + 2;
  const curBal = home === 0 ? rng.lognormal(Math.log(220000), 0.7) : rng.lognormal(Math.log(40000), 0.9);
  const pubRec = rng.next() < 0.16 ? rng.poisson(0.5) + 1 : 0;
  const r2 = (x: number) => Math.round(x * 100) / 100;
  const r4 = (x: number) => Math.round(x * 10000) / 10000;

  const context: BorrowerContext = {
    annual_inc: income,
    annual_inc_joint: joint ? Math.round((income * rng.uniform(1.4, 2.1)) / 100) * 100 : null,
    application_type: joint ? 'Joint App' : 'Individual',
    dti: r2(idx.dti[i]),
    dti_joint: joint ? r2(idx.dti[i] * rng.uniform(0.7, 0.95)) : null,
    emp_length: idx.emp[i] === 255 ? null : idx.emp[i],
    fico_range_low: idx.fico[i],
    fico_range_high: idx.fico[i] + 4,
    home_ownership: HOME_KEYS[home],
    purpose: PURPOSE_KEYS[idx.purpose[i]],
    verification_status: VERIF_KEYS[idx.verif[i]],
    verified_status_joint: joint ? rng.pick(VERIF_KEYS) : null,
    delinq_2yrs: Math.min(rng.poisson(0.25 + 0.05 * g), 12),
    delinq_amnt: rng.next() < 0.003 ? Math.round(rng.lognormal(Math.log(900), 1)) : 0,
    inq_last_6mths: Math.min(rng.poisson(0.5 + 0.12 * g), 8),
    inq_last_12m: recentBureau ? rng.poisson(1.8 + 0.3 * g) : null,
    open_acc: openAcc,
    total_acc: openAcc + rng.poisson(13),
    revol_bal: Math.round(rng.lognormal(Math.log(11000) + 0.5 * Math.log(income / 62000), 0.9)),
    revol_util: r4(clamp(rng.normal(0.5 + 0.04 * g, 0.24), 0, 1.2)),
    installment: r2(installment(amount, rate, T)),
    mort_acc: home === 0 ? rng.poisson(2.2) : rng.poisson(0.5),
    pub_rec: pubRec,
    pub_rec_bankruptcies: pubRec > 0 && rng.next() < 0.7 ? 1 : 0,
    collections_12_mths_ex_med: rng.next() < 0.015 ? 1 : 0,
    num_accts_ever_120_pd: rng.next() < 0.2 ? rng.poisson(1) + 1 : 0,
    pct_tl_nvr_dlq: r4(clamp(rng.normal(0.94, 0.07), 0.4, 1)),
    num_sats: Math.max(1, openAcc - (rng.next() < 0.1 ? 1 : 0)),
    tot_coll_amt: rng.next() < 0.13 ? Math.round(rng.lognormal(Math.log(800), 1)) : 0,
    tot_cur_bal: Math.round(curBal),
    tot_hi_cred_lim: Math.round(curBal * rng.uniform(1.15, 1.6)),
    total_bal_ex_mort: Math.round(rng.lognormal(Math.log(38000), 0.8)),
    total_bc_limit: Math.round(rng.lognormal(Math.log(21000), 0.85)),
    bc_util: rng.next() < 0.033 ? null : r4(clamp(rng.normal(0.58 + 0.03 * g, 0.27), 0, 1.2)),
    all_util: recentBureau ? r4(clamp(rng.normal(0.6, 0.2), 0, 1.2)) : null,
    grade: GRADE_LETTERS[g],
    sub_grade: SUB_GRADES[idx.sub[i]],
  };

  return {
    loan_id: loanIdOf(i),
    issue_d: epoch,
    funded_amnt: amount,
    group: groupKeyDefault(idx, i),
    historical_arm: nearest(amount, rate, T),
    context,
    policy: { loan_amnt: amount, int_rate: rate, term: T },
    outcome: loanOutcome(idx, i),
  };
}

/** Derived y_* outcome of loan i (SPEC §1.2) including the realized historical reward. */
export function loanOutcome(idx: LoanIndex, i: number): LoanOutcome {
  const T = termOf(idx, i);
  const epoch = idx.epochs[idx.issue[i]];
  const amount = idx.amount[i];
  const rate = idx.rate[i];
  const st = idx.status[i];
  const resolved = RESOLVED_STATUS.has(st);
  const defaulted = DEFAULTED_STATUS.has(st);
  const mobSnapshot = epochDiff(epoch, SNAPSHOT);
  let months = Math.min(mobSnapshot, T);
  let revenue: number | null = null;
  let loss: number | null = null;
  let reward: number | null = null;
  let resolution: string | null = null;
  if (resolved) {
    months = defaulted ? idx.defMonth[i] - 1 : idx.payMonth[i];
    const fee = idx.lateFee[i] ? 15 + ((i * 37) % 46) : 0;
    revenue = interestThrough(amount, rate, T, months) + fee;
    loss = defaulted ? Math.max(0, amount * (1 - idx.recov[i])) : 0;
    const funding = (FUNDING_RATE * balanceSum(amount, rate, T, Math.max(months, 1))) / 12;
    reward = revenue - loss - funding;
    resolution = epochAdd(epoch, defaulted ? months + 4 : months);
  }
  const r2 = (x: number | null) => (x === null ? null : Math.round(x * 100) / 100);
  return {
    loan_status: STATUS_KEYS[st],
    resolved,
    defaulted: resolved ? defaulted : null,
    recovery_rate: resolved && defaulted ? Math.round(idx.recov[i] * 10000) / 10000 : null,
    realized_revenue: r2(revenue),
    realized_loss: r2(loss),
    realized_reward: r2(reward),
    months_on_book: Math.max(0, months),
    resolution_epoch: resolution,
  };
}
