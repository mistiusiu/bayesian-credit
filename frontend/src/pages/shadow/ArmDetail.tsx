import type { ShadowArmRow, ShadowReport } from '@/api/types';
import { IntervalBar } from '@/charts/IntervalBar';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Meter } from '@/components/ui/Progress';
import { fmt } from '@/lib/format';
import { armViolations } from '@/lib/guardrails';
import { niceDomain } from '@/charts/kit';
import { useElementWidth } from '@/lib/useElementWidth';

/** Reward percentiles (5 / 50 / 95) and mean on one axis, zero marked. */
function PercentileStrip({ a }: { a: ShadowArmRow }) {
  const [lo, hi] = niceDomain([a.reward_q05, a.reward_q95, a.reward_mean, 0]);
  const [box, w] = useElementWidth<HTMLDivElement>(320);
  const x = (v: number) => 8 + ((v - lo) / (hi - lo || 1)) * (w - 16);
  return (
    <div ref={box} className="w-full">
    <svg viewBox={`0 0 ${w} 46`} width={w} height={46} role="img" aria-label={`Reward 5th percentile ${fmt.usd(a.reward_q05)}, median ${fmt.usd(a.reward_q50)}, 95th ${fmt.usd(a.reward_q95)}, mean ${fmt.usd(a.reward_mean)}`}>
      <line x1={8} x2={w - 8} y1={18} y2={18} stroke="var(--grid)" />
      <line x1={x(0)} x2={x(0)} y1={8} y2={28} stroke="var(--axis)" />
      <line x1={x(a.reward_q05)} x2={x(a.reward_q95)} y1={18} y2={18} stroke="var(--series-1)" strokeWidth={6} strokeLinecap="round" opacity={0.35} />
      <line x1={x(a.reward_q50)} x2={x(a.reward_q50)} y1={10} y2={26} stroke="var(--series-1)" strokeWidth={2} />
      <circle cx={x(a.reward_mean)} cy={18} r={4.5} fill="var(--text-primary)" stroke="var(--surface-1)" strokeWidth={2} />
      <text x={x(lo)} y={42} fontSize={10} fill="var(--text-axis)" className="num">
        {fmt.usdCompact(lo)}
      </text>
      <text x={x(0)} y={42} fontSize={10} textAnchor="middle" fill="var(--text-axis)">
        0
      </text>
      <text x={x(hi)} y={42} fontSize={10} textAnchor="end" fill="var(--text-axis)" className="num">
        {fmt.usdCompact(hi)}
      </text>
    </svg>
    </div>
  );
}

export function ArmDetail({ report, arm: a }: { report: ShadowReport; arm: ShadowArmRow }) {
  const fails = armViolations(a, { annual_inc: report.context.annual_inc, dti: report.context.dti }, report.eligibility.guardrails, report.eligibility.reference_rate);
  const admittedDespite = a.eligible && fails.length > 0;
  const tags = [
    a.arm === report.recommended_arm && 'recommended',
    a.arm === report.historical_arm && 'historical arm',
    !a.eligible && 'not admissible',
    admittedDespite && (a.arm === report.historical_arm ? 'admitted for comparison' : 'fallback arm'),
  ].filter(Boolean) as string[];
  const tone = (t: string) => (t === 'not admissible' ? 'critical' : t === 'recommended' ? 'good' : t === 'admitted for comparison' || t === 'fallback arm' ? 'warning' : 'neutral');
  return (
    <Card title={`Arm ${a.arm} · ${fmt.arm(a)}`} subtitle={a.is_null ? 'Decline: reward 0 by definition' : `instalment ${fmt.usd2(a.installment)} / month`} actions={tags.map((t) => <Badge key={t} tone={tone(t)}>{t}</Badge>)}>
      {a.is_null ? (
        <p className="text-xs text-muted">The null arm lets the policy choose not to lend (§1.5). It is always admissible and is recommended when no admissible arm has a positive risk-adjusted reward.</p>
      ) : (
        <div className="space-y-4 text-xs">
          {fails.length > 0 && (
            <div className={`rounded-md border px-3 py-2 ${a.eligible ? 'border-warning/40 bg-warning/10' : 'border-critical/40 bg-critical/10'}`}>
              <div className="mb-0.5 font-medium text-ink">{a.eligible ? 'Fails a guardrail but kept' : 'Not admissible for this borrower'}</div>
              <ul className="list-disc pl-4 text-ink-2">
                {fails.map((v) => (
                  <li key={v.key}>{v.text}</li>
                ))}
              </ul>
            </div>
          )}
          <div>
            <div className="mb-1 flex justify-between text-2xs text-muted">
              <span>P(default) posterior mean and 95% HDI</span>
              <span className="num text-ink">
                {fmt.pct(a.p_default_mean)} [{fmt.pct(a.p_default_hdi_low)}, {fmt.pct(a.p_default_hdi_high)}]
              </span>
            </div>
            <IntervalBar mean={a.p_default_mean} low={a.p_default_hdi_low} high={a.p_default_hdi_high} max={Math.max(0.5, Math.ceil(a.p_default_hdi_high * 10) / 10)} width={320} label="P(default)" />
          </div>
          <div className="num grid grid-cols-2 gap-x-4 gap-y-1.5">
            <span className="text-muted">E[R | default]</span>
            <span className="text-right text-ink">{fmt.pct(a.expected_recovery)}</span>
            <span className="text-muted" title="Realised share of the scheduled interest for this grade and term (prepayment); revenue and funding cost are multiplied by ρ">
              Realised interest ρ
            </span>
            <span className="text-right text-ink">{fmt.pct(a.realization, 1)}</span>
            <span className="text-muted">Expected reward</span>
            <span className="text-right font-semibold text-ink">{fmt.usd(a.reward_mean)}</span>
            <span className="text-muted">Std. deviation</span>
            <span className="text-right text-ink">{fmt.usd(a.reward_std)}</span>
            <span className="text-muted">Risk-adjusted (mean − λ·std)</span>
            <span className="text-right text-ink">{fmt.usd(a.risk_adjusted)}</span>
            <span className="text-muted">5% · 50% · 95%</span>
            <span className="text-right text-ink">
              {fmt.usdCompact(a.reward_q05)} · {fmt.usdCompact(a.reward_q50)} · {fmt.usdCompact(a.reward_q95)}
            </span>
          </div>
          <div>
            <div className="mb-1 text-2xs text-muted">Reward distribution: band 5–95%, tick median, dot mean</div>
            <PercentileStrip a={a} />
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-muted">P(reward &gt; 0)</span>
            <Meter value={a.prob_positive} label="Probability the reward is positive" width={140} />
          </div>
        </div>
      )}
    </Card>
  );
}
