import { useState } from 'react';
import type { ShadowArmRow, ShadowReport } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Heatmap } from '@/charts/Heatmap';
import { ScaleLegend } from '@/charts/ScaleLegend';
import { Segmented } from '@/components/ui/Segmented';
import { fmt } from '@/lib/format';
import { armViolations, corridorColumns } from '@/lib/guardrails';
import { divColor, divStops } from '@/lib/palette';
import { useTheme } from '@/lib/theme';

const short = (v: number) => {
  const a = Math.abs(v);
  const s = a >= 1000 ? `${(a / 1000).toFixed(a >= 10000 ? 0 : 1)}k` : a.toFixed(0);
  return `${v < 0 ? '−' : ''}$${s}`;
};

/** Expected risk-adjusted reward of every arm (diverging: loss red · 0 gray · profit blue); guardrails hatch the rest. */
export function ArmRewardHeatmap({ report, selected, onSelect }: { report: ShadowReport; selected: number; onSelect: (arm: number) => void }) {
  const { theme } = useTheme();
  const e = report.eligibility;
  const ctx = { annual_inc: report.context.annual_inc, dti: report.context.dti };
  const terms = [...new Set(report.arms.filter((a) => !a.is_null).map((a) => a.term))].sort((a, b) => a - b);
  const rec = report.arms[report.recommended_arm];
  const [termState, setTerm] = useState<number | null>(null);
  const term = termState ?? (rec && !rec.is_null ? rec.term : terms[0]);
  const lend = report.arms.filter((a) => !a.is_null);
  const amounts = [...new Set(lend.map((a) => a.amount))].sort((a, b) => b - a);
  const rates = [...new Set(lend.map((a) => a.rate))].sort((a, b) => a - b);
  const eligible = lend.filter((a) => a.eligible).map((a) => a.risk_adjusted);
  const maxPos = Math.max(...eligible, 1);
  const maxNeg = Math.max(...eligible.map((v) => -v), 1);
  const find = (amount: number, rate: number): ShadowArmRow | undefined => lend.find((a) => a.term === term && a.amount === amount && a.rate === rate);
  const decline = report.arms.find((a) => a.is_null);
  const selArm = report.arms[selected];
  const corridor = corridorColumns(rates, e.reference_rate, e.rate_corridor_applied ? e.guardrails.rate_corridor : null);
  const why = (a: ShadowArmRow) => armViolations(a, ctx, e.guardrails, e.reference_rate);
  return (
    <ChartFrame
      title="Expected risk-adjusted reward by arm"
      caption={`Posterior-predictive mean − λ·std (λ = ${report.risk_aversion}) per arm for this borrower, ${report.n_samples.toLocaleString()} draws, interest scaled by the prepayment factor ρ. Ring = recommended arm, H = the arm Lending Club issued, hatched = not admissible under the guardrails (value dimmed), frame = rate corridor around the reference rate.`}
      actions={<Segmented<number> size="sm" label="Term" value={term} onChange={setTerm} options={terms.map((t) => ({ value: t, label: `${t}m` }))} />}
      legend={
        <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
          <ScaleLegend title="USD per loan (each side scaled to its own extreme)" stops={divStops(theme)} labels={[short(-maxNeg), '$0', short(maxPos)]} />
          <span className="inline-flex items-center gap-1.5 text-2xs text-ink-2">
            <span aria-hidden="true" className="hatch-na inline-block h-3 w-4 rounded-[2px] border border-line" />
            Not admissible
          </span>
          {corridor && (
            <span className="inline-flex items-center gap-1.5 text-2xs text-ink-2">
              <span aria-hidden="true" className="inline-block h-3 w-4 rounded-[3px] border-[1.5px] border-[var(--text-secondary)]" />
              Rate corridor
            </span>
          )}
        </div>
      }
      table={{
        columns: [
          { key: 'arm', label: 'Arm' },
          { key: 'e', label: 'Admissible' },
          { key: 'r', label: 'Risk-adjusted', align: 'right' },
          { key: 'p', label: 'P(reward > 0)', align: 'right' },
          { key: 'pd', label: 'P(default)', align: 'right' },
          { key: 'rho', label: 'ρ', align: 'right' },
          { key: 'w', label: 'Guardrails failed' },
        ],
        rows: report.arms.map((a) => ({
          arm: fmt.arm(a),
          e: a.eligible ? 'yes' : 'no',
          r: fmt.usd(a.risk_adjusted),
          p: fmt.pct(a.prob_positive),
          pd: fmt.pct(a.p_default_mean),
          rho: a.is_null ? '—' : fmt.pct(a.realization, 1),
          w: why(a).map((v) => v.text).join('; ') || '—',
        })),
      }}
    >
      <div className="grid gap-4 px-2 pb-1 lg:grid-cols-[minmax(0,1fr)_11rem]">
        <Heatmap
          label={`Risk-adjusted reward, ${term}-month term`}
          rowTitle="Amount"
          colTitle="Interest rate"
          cellHeight={32}
          band={corridor && e.reference_rate !== null ? { ...corridor, label: `reference ${fmt.pct(e.reference_rate, 1)} ± ${fmt.num((e.guardrails.rate_corridor ?? 0) * 100, 0)} pp` } : null}
          selected={selArm && !selArm.is_null && selArm.term === term ? [amounts.indexOf(selArm.amount), rates.indexOf(selArm.rate)] : null}
          onSelect={(ri, ci) => {
            const a = find(amounts[ri], rates[ci]);
            if (a) onSelect(a.arm);
          }}
          rows={amounts.map((a) => ({ key: String(a), label: fmt.amountShort(a) }))}
          cols={rates.map((r) => ({ key: String(r), label: fmt.rate(r) }))}
          color={(v) => divColor(v >= 0 ? v / maxPos : v / maxNeg, theme)}
          cell={(ri, ci) => {
            const a = find(amounts[ri], rates[ci]);
            if (!a) return { value: null, na: true, ariaLabel: 'not in grid' };
            const isHist = a.arm === report.historical_arm;
            const fails = why(a);
            const label = `${fmt.arm(a)}: risk-adjusted ${fmt.usd(a.risk_adjusted)}, P(reward > 0) ${fmt.pct(a.prob_positive)}${a.eligible ? '' : `, not admissible (${fails.map((v) => v.text).join('; ') || 'guardrails'})`}${a.arm === report.recommended_arm ? ', recommended' : ''}${isHist ? ', historical arm' : ''}`;
            if (!a.eligible) {
              return {
                value: null,
                na: true,
                hatched: true,
                text: short(a.risk_adjusted),
                marker: isHist ? 'H' : null,
                ariaLabel: label,
                tooltip: (
                  <div className="num space-y-0.5">
                    <div className="font-medium text-ink">{fmt.arm(a)} · not admissible</div>
                    {(fails.length ? fails : [{ key: 'x', text: 'excluded by the guardrails' }]).map((v) => (
                      <div key={v.key}>{v.text}</div>
                    ))}
                    <div className="text-muted">would score {fmt.usd(a.risk_adjusted)} · instalment {fmt.usd(a.installment)}</div>
                  </div>
                ),
              };
            }
            return {
              value: a.risk_adjusted,
              text: short(a.risk_adjusted),
              outline: a.arm === report.recommended_arm,
              marker: isHist ? 'H' : null,
              ariaLabel: label,
              tooltip: (
                <div className="num space-y-0.5">
                  <div className="font-medium text-ink">{fmt.arm(a)}</div>
                  <div>
                    E[reward] {fmt.usd(a.reward_mean)} · σ {fmt.usd(a.reward_std)}
                  </div>
                  <div className="text-muted">
                    P(default) {fmt.pct(a.p_default_mean)} · P(reward &gt; 0) {fmt.pct(a.prob_positive)} · ρ {fmt.pct(a.realization, 0)}
                  </div>
                  {fails.length > 0 && <div className="text-muted">{isHist ? 'Admitted for comparison' : 'Fallback arm'}: {fails.map((v) => v.text).join('; ')}</div>}
                </div>
              ),
            };
          }}
        />
        <div className="space-y-2 text-xs">
          <button
            type="button"
            onClick={() => decline && onSelect(decline.arm)}
            aria-pressed={selected === decline?.arm}
            className="w-full rounded-md border border-line p-2.5 text-left hover:bg-surface-2"
          >
            <span className="block text-2xs text-muted">Decline (arm 0)</span>
            <span className="num block text-base font-semibold text-ink">$0</span>
            <span className="block text-2xs text-muted">{report.recommended_arm === decline?.arm ? 'recommended: no admissible arm beats 0' : 'always admissible: not lending'}</span>
          </button>
          <p className="text-2xs leading-snug text-muted">Select a cell for the arm's posterior predictive detail. Use the term toggle to compare 36- and 60-month offers.</p>
        </div>
      </div>
    </ChartFrame>
  );
}
