/** Shadow portfolio evaluation mock (SPEC §2.10, API §7.2 / §13.1). */
import type {
  ArmRow,
  InclusionRow,
  PortfolioGradeRow,
  PortfolioLoanRow,
  PortfolioOutcomeClass,
  PortfolioRequest,
  PortfolioResult,
  RiskAversionPoint,
  SelectionMix,
} from '@/api/types';
import { rewardMoments } from './finance';
import { type LoanIndex, filterLoans, loanIdOf, loanOutcome, termOf } from './loans';
import { Rng, seedOf } from './prng';
import { type FitModel, eligibility, groupArmReward, leafPosterior, leafRecovery, realizationOf } from './riskModel';
import { round } from './stats';
import { GRADE_LETTERS, SUB_GRADES } from './vocab';

const CLASSES: PortfolioOutcomeClass[] = ['rescued', 'upsized', 'downsized', 'repriced_down', 'repriced_up', 'declined', 'unchanged'];
const emptyCounts = () => Object.fromEntries(CLASSES.map((c) => [c, 0])) as Record<PortfolioOutcomeClass, number>;
const LAMBDAS = [0, 0.5, 1];

const rateBucket = (r: number) => (r < 0.1 ? '<10%' : r < 0.15 ? '10-15%' : r < 0.2 ? '15-20%' : r < 0.25 ? '20-25%' : '>=25%');
const amountBucket = (a: number) => (a < 10000 ? '<$10k' : a < 20000 ? '$10k-20k' : a < 30000 ? '$20k-30k' : '>=$30k');

/** Distribution of decisions over term / rate / amount buckets (lending decisions). */
export function selectionMix(picks: { arm: ArmRow; weight: number }[], maxRate: number, maxAmount: number): SelectionMix {
  const n = picks.reduce((a, p) => a + p.weight, 0);
  const lend = picks.filter((p) => !p.arm.is_null);
  const nl = lend.reduce((a, p) => a + p.weight, 0);
  const share = (key: (a: ArmRow) => string, keys: string[]) => {
    const out: Record<string, number> = Object.fromEntries(keys.map((k) => [k, 0]));
    for (const p of lend) out[key(p.arm)] += p.weight / (nl || 1);
    return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, round(v, 6)]));
  };
  const mean = (f: (a: ArmRow) => number) => (nl ? round(lend.reduce((a, p) => a + p.weight * f(p.arm), 0) / nl, 6) : null);
  return {
    n: Math.round(n),
    n_lending: Math.round(nl),
    decline_share: round(n ? 1 - nl / n : 0, 6),
    by_term: share((a) => String(a.term), ['36', '60']),
    by_rate: share((a) => rateBucket(a.rate), ['<10%', '10-15%', '15-20%', '20-25%', '>=25%']),
    by_amount: share((a) => amountBucket(a.amount), ['<$10k', '$10k-20k', '$20k-30k', '>=$30k']),
    mean_rate: mean((a) => a.rate),
    mean_amount: mean((a) => a.amount),
    share_36m: mean((a) => (a.term === 36 ? 1 : 0)),
    share_at_max_rate: mean((a) => (a.rate >= maxRate - 1e-12 ? 1 : 0)),
    share_at_max_amount: mean((a) => (a.amount >= maxAmount ? 1 : 0)),
  };
}

export function portfolioResult(model: FitModel, idx: LoanIndex, fitId: number, req: PortfolioRequest): Omit<PortfolioResult, 'job'> {
  const t0 = performance.now();
  const f = req.filters ?? {};
  let pool = filterLoans(idx, { grade: f.grade, purpose: f.purpose, home_ownership: f.home_ownership, resolved: f.resolved_only ? true : undefined });
  pool = pool.filter((i) => {
    const e = idx.epochs[idx.issue[i]];
    return e >= (f.issue_from ?? model.config.train_end) && (!f.issue_to || e <= f.issue_to);
  });
  const rng = new Rng(seedOf('portfolio', fitId, req.seed));
  const take = Math.min(req.max_loans, pool.length);
  for (let k = 0; k < take; k++) {
    const j = k + Math.floor(rng.next() * (pool.length - k));
    [pool[k], pool[j]] = [pool[j], pool[k]];
  }
  const sample = pool.slice(0, take).sort((a, b) => a - b);
  const arms = model.grid.arms;
  const lambda0 = model.config.reward.risk_aversion;
  const threshold = model.config.reward.lendable_threshold;
  const corridor = model.config.arm_space.rate_corridor;
  const maxRate = Math.max(...model.grid.rates);
  const maxAmount = Math.max(...model.grid.amounts);
  const byGrade = GRADE_LETTERS.map((grade) => ({ grade, n: 0, hist: 0, shadow: 0, realized: 0, realizedN: 0, counts: emptyCounts(), lendable: 0, rateH: 0, rateS: 0, amtH: 0, amtS: 0, nS: 0 }));
  const counts = emptyCounts();
  const loans: PortfolioLoanRow[] = [];
  const recPicks: { arm: ArmRow; weight: number }[] = [];
  const histPicks: { arm: ArmRow; weight: number }[] = [];
  const sweep = LAMBDAS.map(() => ({ profit: 0, picks: [] as { arm: ArmRow; weight: number }[] }));
  const part = { reprice: 0, resize: 0 };
  let histTotal = 0;
  let shadowTotal = 0;
  let upliftRisk = 0;
  let lendable = 0;
  let nRealized = 0;
  const refStats = { n: 0, ref: 0, diff: 0, above: 0, within: 0, histDiff: 0 };
  let nEligSum = 0;
  let nSingle = 0;
  let withRef = 0;
  for (const i of sample) {
    const group = model.groupOf(i);
    const T = termOf(idx, i);
    const histArm = model.armOf(idx.amount[i], idx.rate[i], T);
    const el = eligibility(model, { income: idx.income[i], dti: idx.dti[i], subGrade: SUB_GRADES[idx.sub[i]], grade: GRADE_LETTERS[idx.grade[i]] }, histArm);
    nEligSum += el.nEligibleLending;
    if (el.nEligibleLending <= 1) nSingle++;
    if (el.refRate !== null) withRef++;
    const reward = (k: number) => (arms[k].is_null ? { mean: 0, std: 0, noDefault: 0 } : groupArmReward(model, group, k));
    const best = (lambda: number, keep: (k: number) => boolean) => {
      let pick = arms[0].is_null ? 0 : -1;
      let v = pick === 0 ? 0 : -Infinity;
      for (const a of arms) {
        if (a.is_null || !el.mask[a.arm] || !keep(a.arm)) continue;
        const r = reward(a.arm);
        const s = r.mean - lambda * r.std;
        if (s > v) {
          v = s;
          pick = a.arm;
        }
      }
      return pick < 0 ? histArm : pick;
    };
    const rec = best(lambda0, () => true);
    const ha = arms[histArm];
    const ra = arms[rec];
    const post = leafPosterior(model, group, histArm);
    const recov = leafRecovery(model, group, histArm);
    const histExp = rewardMoments(idx.amount[i], idx.rate[i], T, post.alpha / (post.alpha + post.beta), recov.mean, recov.variance, model.config.reward.funding_rate, realizationOf(model, group, histArm)).mean;
    const recR = reward(rec);
    const shadowExp = recR.mean;
    histTotal += histExp;
    shadowTotal += shadowExp;
    upliftRisk += recR.mean - lambda0 * recR.std - (reward(histArm).mean - lambda0 * reward(histArm).std);
    part.reprice += reward(best(lambda0, (k) => arms[k].term === ha.term && arms[k].amount === ha.amount)).mean;
    part.resize += reward(best(lambda0, (k) => arms[k].term === ha.term)).mean;
    LAMBDAS.forEach((lam, s) => {
      const pick = best(lam, () => true);
      sweep[s].profit += reward(pick).mean;
      sweep[s].picks.push({ arm: arms[pick], weight: 1 });
    });
    recPicks.push({ arm: ra, weight: 1 });
    histPicks.push({ arm: ha, weight: 1 });
    if (el.refRate !== null && !ra.is_null) {
      refStats.n++;
      refStats.ref += el.refRate;
      refStats.diff += ra.rate - el.refRate;
      refStats.above += ra.rate > el.refRate ? 1 : 0;
      refStats.within += corridor === null || Math.abs(ra.rate - el.refRate) <= corridor + 1e-9 ? 1 : 0;
      refStats.histDiff += idx.rate[i] - el.refRate;
    }
    const isLendable = arms.some((a) => {
      if (a.is_null || !el.mask[a.arm]) return false;
      const r = reward(a.arm);
      const pd = leafPosterior(model, group, a.arm);
      return r.noDefault > 0 && r.mean > 0 && 1 - pd.alpha / (pd.alpha + pd.beta) >= threshold;
    });
    const realized = loanOutcome(idx, i).realized_reward;
    if (realized !== null) nRealized++;
    const declined = ra.is_null;
    let cls: PortfolioOutcomeClass = 'unchanged';
    if (declined) cls = 'declined';
    else if (realized !== null && realized < 0 && shadowExp > 0) cls = 'rescued';
    else if (ra.amount > ha.amount && shadowExp > 0) cls = 'upsized';
    else if (ra.amount < ha.amount) cls = 'downsized';
    else if (ra.rate < ha.rate) cls = 'repriced_down';
    else if (ra.rate > ha.rate) cls = 'repriced_up';
    counts[cls]++;
    const g = byGrade[idx.grade[i]];
    g.n++;
    g.hist += histExp;
    g.shadow += shadowExp;
    g.counts[cls]++;
    g.rateH += idx.rate[i];
    g.amtH += idx.amount[i];
    if (!declined) {
      g.rateS += ra.rate;
      g.amtS += ra.amount;
      g.nS++;
    }
    if (realized !== null) {
      g.realized += realized;
      g.realizedN++;
    }
    if (isLendable) {
      g.lendable++;
      lendable++;
    }
    loans.push({
      loan_id: loanIdOf(i), grade: GRADE_LETTERS[idx.grade[i]], group, historical_arm: histArm, recommended_arm: rec,
      historical_label: ha.label, recommended_label: ra.label, historical_expected_reward: round(histExp, 2),
      shadow_expected_reward: round(shadowExp, 2), realized_reward: realized, class: cls,
    });
  }
  const n = sample.length;
  const pct = (v: number) => round((v - histTotal) / Math.max(Math.abs(histTotal), 1), 6);
  const step = (profit: number) => ({ expected_profit: round(profit, 2), uplift: round(profit - histTotal, 2), uplift_pct: pct(profit) });
  const mixRec = selectionMix(recPicks, maxRate, maxAmount);
  const mixHist = selectionMix(histPicks, maxRate, maxAmount);
  const byRiskAversion: Record<string, RiskAversionPoint> = {};
  LAMBDAS.forEach((lam, s) => {
    const m = selectionMix(sweep[s].picks, maxRate, maxAmount);
    byRiskAversion[lam.toFixed(1)] = {
      risk_aversion: lam,
      expected_profit_shadow: round(sweep[s].profit, 2),
      expected_profit_historical: round(histTotal, 2),
      uplift: round(sweep[s].profit - histTotal, 2),
      uplift_pct: pct(sweep[s].profit),
      mean_amount_recommended: m.mean_amount,
      mean_rate_recommended: m.mean_rate,
      share_36m_recommended: m.share_36m,
      share_at_max_amount_recommended: m.share_at_max_amount,
      share_at_max_rate_recommended: m.share_at_max_rate,
      decline_share: m.decline_share,
    };
  });
  const gradeRows: PortfolioGradeRow[] = byGrade
    .filter((g) => g.n > 0)
    .map((g) => ({ grade: g.grade, n: g.n, expected_profit_historical: round(g.hist, 2), expected_profit_shadow: round(g.shadow, 2), realized_profit_historical: g.realizedN > 0 ? round(g.realized, 2) : null, counts: g.counts }));
  const inclusion: InclusionRow[] = byGrade
    .filter((g) => g.n > 0)
    .map((g) => ({
      grade: g.grade, n: g.n, share_lendable: round(g.lendable / g.n, 4), decline_rate_shadow: round(g.counts.declined / g.n, 4),
      mean_rate_historical: round(g.rateH / g.n, 4), mean_rate_shadow: g.nS > 0 ? round(g.rateS / g.nS, 4) : null,
      mean_amount_historical: round(g.amtH / g.n, 0), mean_amount_shadow: g.nS > 0 ? round(g.amtS / g.nS, 0) : null,
    }));
  loans.sort((a, b) => Math.abs(b.shadow_expected_reward - b.historical_expected_reward) - Math.abs(a.shadow_expected_reward - a.historical_expected_reward));
  const realizedTotal = byGrade.reduce((a, g) => a + g.realized, 0);
  const a = model.config.arm_space;
  const guardrails = { max_payment_to_income: a.max_payment_to_income, rate_corridor: a.rate_corridor, max_amount_to_income: a.max_amount_to_income, max_total_dti: a.max_total_dti };
  return {
    fit: fitId,
    request: req,
    summary: {
      n_loans: n,
      n_lendable: lendable,
      share_lendable: round(lendable / Math.max(n, 1), 4),
      expected_profit_historical: round(histTotal, 2),
      expected_profit_shadow: round(shadowTotal, 2),
      uplift: round(shadowTotal - histTotal, 2),
      uplift_pct: pct(shadowTotal),
      realized_profit_historical: nRealized > 0 ? round(realizedTotal, 2) : null,
      counts,
      uplift_risk_adjusted: round(upliftRisk, 2),
      n_realized: nRealized,
      seconds: round((performance.now() - t0) / 1000, 3),
      recommendation_mix: {
        recommended: mixRec,
        historical: mixHist,
        mean_rate_recommended: mixRec.mean_rate,
        mean_rate_historical: mixHist.mean_rate,
        mean_amount_recommended: mixRec.mean_amount,
        mean_amount_historical: mixHist.mean_amount,
        share_36m_recommended: mixRec.share_36m,
        share_36m_historical: mixHist.share_36m,
        share_at_max_rate_recommended: mixRec.share_at_max_rate,
        share_at_max_amount_recommended: mixRec.share_at_max_amount,
        reference: refStats.n
          ? {
              n: refStats.n,
              mean_reference_rate: round(refStats.ref / refStats.n, 6),
              mean_rate_minus_reference: round(refStats.diff / refStats.n, 6),
              share_above_reference: round(refStats.above / refStats.n, 6),
              share_within_corridor: round(refStats.within / refStats.n, 6),
              historical_mean_rate_minus_reference: round(refStats.histDiff / refStats.n, 6),
            }
          : null,
      },
      uplift_decomposition: {
        expected_profit_historical: round(histTotal, 2),
        repricing_only: step(part.reprice),
        repricing_and_resizing: step(part.resize),
        unrestricted: step(shadowTotal),
        note: 'best admissible arm keeping the historical term and amount / the historical term / unrestricted',
      },
      by_risk_aversion: byRiskAversion,
      eligibility: {
        guardrails,
        reference_rates: true,
        share_with_reference_rate: round(withRef / Math.max(n, 1), 6),
        mean_n_eligible_lending: round(nEligSum / Math.max(n, 1), 4),
        share_single_fallback_arm: round(nSingle / Math.max(n, 1), 6),
      },
    },
    by_grade: gradeRows,
    inclusion,
    top_loans: loans.slice(0, 50),
  };
}
