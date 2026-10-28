/** Amortisation and the compound reward R = Revenue − Expected Loss − Funding (SPEC §2.7), closed form. */
import { betaCdf } from './stats';

export function installment(amount: number, rate: number, term: number): number {
  if (term <= 0 || amount <= 0) return 0;
  const m = rate / 12;
  if (m <= 0) return amount / term;
  const f = (1 + m) ** term;
  return (amount * m * f) / (f - 1);
}

export interface ArmEconomics {
  installment: number;
  /** Interest paid in month k (k = 1..T), index k − 1. */
  interest: Float64Array;
  /** Opening balance of month k, index k − 1. */
  balance: Float64Array;
  totalInterest: number;
  /** Weighted-average life in years: Σ balance_{k−1} / 12 / amount. */
  walYears: number;
}

const cache = new Map<string, ArmEconomics>();

export function economics(amount: number, rate: number, term: number): ArmEconomics {
  const key = `${amount}|${rate}|${term}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const pay = installment(amount, rate, term);
  const interest = new Float64Array(term);
  const balance = new Float64Array(term);
  let bal = amount;
  let total = 0;
  let balSum = 0;
  for (let k = 0; k < term; k++) {
    balance[k] = bal;
    balSum += bal;
    const i = bal * (rate / 12);
    interest[k] = i;
    total += i;
    bal = Math.max(0, bal - (pay - i));
  }
  const econ = { installment: pay, interest, balance, totalInterest: total, walYears: amount > 0 ? balSum / 12 / amount : 0 };
  cache.set(key, econ);
  return econ;
}

export interface RewardMoments {
  mean: number;
  std: number;
  revenue: number;
  expectedLoss: number;
  funding: number;
  /** Reward if the loan performs. */
  noDefault: number;
  /** Mean reward conditional on default. */
  onDefault: number;
  /** Contractual interest before default and prepayment adjustments. */
  grossInterest: number;
  /** Realised share ρ applied to interest and funding. */
  realization: number;
}

/** Default-timing CDF F(m) given eventual default: Beta(1.4, 2.6) on m/T (front-loaded, ≈ the MCTM timing). */
const timingCache = new Map<number, Float64Array>();
export function defaultTiming(term: number): Float64Array {
  let f = timingCache.get(term);
  if (!f) {
    f = new Float64Array(term);
    for (let k = 0; k < term; k++) f[k] = betaCdf((k + 1) / term, 1.4, 2.6);
    timingCache.set(term, f);
  }
  return f;
}

/**
 * Closed form of the core's CompoundRewardFunction (docs/MATH.md §2.4, §2.8) with front-loaded
 * default timing F(m) (see defaultTiming):
 *   Reward(p, R) = ρ·a − p·h(R),  a = gross interest − funding,  h(R) = (ρ·b + c) − c·R,
 *   b = Σ I_m·F(m),  c = L,  E = ρ·a − (ρ·b + c)·E[p] + c·E[p]·E[R],
 *   Var = E[p²]·E[h²] − (E[p]·E[h])²  (posterior uncertainty of p and R, p ⫫ R).
 * `noDefault` / `onDefault` are the reward of a performing / defaulting loan (lendability checks).
 */
export function rewardMoments(
  amount: number,
  rate: number,
  term: number,
  pMean: number,
  rMean: number,
  rVar: number,
  fundingRate: number,
  realization = 1,
  /** Posterior variance of p (Var of the leaf Beta); 0 treats p as known. */
  pVar = 0,
): RewardMoments {
  if (amount <= 0 || term <= 0) {
    return { mean: 0, std: 0, revenue: 0, expectedLoss: 0, funding: 0, noDefault: 0, onDefault: 0, grossInterest: 0, realization: 1 };
  }
  const rho = realization;
  const e = economics(amount, rate, term);
  const b = timedInterest(amount, rate, term);
  const revenue = rho * (e.totalInterest - pMean * b);
  const expectedLoss = amount * pMean * (1 - rMean);
  const funding = rho * fundingRate * amount * e.walYears;
  const mean = revenue - expectedLoss - funding;
  const noDefault = rho * e.totalInterest - funding;
  const onDefault = rho * (e.totalInterest - b) - amount * (1 - rMean) - funding;
  const B = rho * b + amount;
  const eh = B - amount * rMean;
  const eh2 = B * B - 2 * B * amount * rMean + amount * amount * (rVar + rMean * rMean);
  const ep2 = pVar + pMean * pMean;
  const variance = Math.max(0, ep2 * eh2 - (pMean * eh) ** 2);
  return { mean, std: Math.sqrt(variance), revenue, expectedLoss, funding, noDefault, onDefault, grossInterest: e.totalInterest, realization: rho };
}

/** b = Σ_m I_m·F(m): contractual interest at risk of being lost to a default (cached per arm). */
const timedCache = new Map<string, number>();
export function timedInterest(amount: number, rate: number, term: number): number {
  const key = `${amount}|${rate}|${term}`;
  let b = timedCache.get(key);
  if (b === undefined) {
    const e = economics(amount, rate, term);
    const timing = defaultTiming(term);
    b = 0;
    for (let k = 0; k < term; k++) b += e.interest[k] * timing[k];
    timedCache.set(key, b);
  }
  return b;
}

/** Balance after k scheduled payments (closed form, no arrays — used per loan). */
export function balanceAfter(amount: number, rate: number, term: number, k: number): number {
  if (k <= 0) return amount;
  if (k >= term) return 0;
  const m = rate / 12;
  if (m <= 0) return amount * (1 - k / term);
  const pay = installment(amount, rate, term);
  const q = (1 + m) ** k;
  return Math.max(0, amount * q - (pay * (q - 1)) / m);
}

/** Interest paid over the first k scheduled payments. */
export function interestThrough(amount: number, rate: number, term: number, k: number): number {
  const kk = Math.min(Math.max(k, 0), term);
  return installment(amount, rate, term) * kk - (amount - balanceAfter(amount, rate, term, kk));
}

/** Σ_{j=0}^{k−1} balance_j — exposure-months for the funding cost on the realized schedule. */
export function balanceSum(amount: number, rate: number, term: number, k: number): number {
  const kk = Math.min(Math.max(k, 0), term);
  let s = 0;
  // k ≤ 60, a short loop is cheap and exact
  for (let j = 0; j < kk; j++) s += balanceAfter(amount, rate, term, j);
  return s;
}
