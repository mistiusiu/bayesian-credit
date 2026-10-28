import { useState } from 'react';
import { useSimArms, useTimeseries } from '@/api/client';
import type { PolicyName, SimArmRow, Simulation } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Heatmap } from '@/charts/Heatmap';
import { ScaleLegend } from '@/charts/ScaleLegend';
import { Select } from '@/components/ui/Field';
import { Segmented } from '@/components/ui/Segmented';
import { QueryState } from '@/components/ui/States';
import { policyLabel } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { seqColor, seqStops } from '@/lib/palette';
import { useTheme } from '@/lib/theme';

/** Arm-selection heatmap (amount × rate per term) with an epoch scrubber. */
export function ArmSelection({ sim }: { sim: Simulation }) {
  const { theme } = useTheme();
  const epochs = useTimeseries(sim.id, sim).data?.epochs ?? [];
  const [policy, setPolicy] = useState<PolicyName>(sim.config.policies[0]);
  const [termPick, setTerm] = useState<number | null>(null);
  const [mode, setMode] = useState<'share' | 'cum_share'>('share');
  const [idx, setIdx] = useState<number | null>(null);
  const last = Math.max(0, epochs.length - 1);
  const at = idx === null ? last : Math.min(idx, last);
  const epoch = epochs[at] ?? null;
  const q = useSimArms(sim.id, sim, { policy, epoch });
  return (
    <ChartFrame
      title="Arm selection"
      caption={`Share of ${policyLabel(policy)} decisions per arm ${mode === 'share' ? 'in the selected epoch' : 'from the start to the selected epoch'}; Thompson concentrates on few arms as posteriors sharpen. Decline is reported separately.`}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <Select className="h-7 w-36 text-xs" aria-label="Policy" value={policy} onValue={setPolicy} options={sim.config.policies.map((p) => ({ value: p, label: policyLabel(p) }))} />
          <Segmented<'share' | 'cum_share'> size="sm" label="Window" value={mode} onChange={setMode} options={[{ value: 'share', label: 'Epoch' }, { value: 'cum_share', label: 'Cumulative' }]} />
        </div>
      }
    >
      <div className="space-y-3 px-2 pb-1">
        <label className="flex items-center gap-3 text-xs text-ink-2">
          <span className="shrink-0">Epoch</span>
          <input type="range" min={0} max={last} step={1} value={at} onChange={(e) => setIdx(Number(e.target.value))} aria-valuetext={fmt.epoch(epoch)} className="flex-1 accent-[var(--accent-strong)]" disabled={epochs.length < 2} />
          <span className="num w-20 text-right font-medium text-ink">{fmt.epoch(epoch)}</span>
        </label>
        <QueryState query={q} loadingLabel="Loading arm statistics">
          {(d) => {
            const lead = [...d.arms].filter((a) => !a.is_null).sort((a, b) => b[mode] - a[mode])[0];
            const term = termPick ?? lead?.term ?? d.terms[0];
            const rows = d.arms.filter((a) => !a.is_null && a.term === term);
            const decline = d.arms.find((a) => a.is_null);
            const value = (a: SimArmRow) => a[mode];
            const max = Math.max(...d.arms.filter((a) => !a.is_null).map(value), 1e-6);
            const amounts = [...d.amounts].reverse();
            const top = [...d.arms].filter((a) => !a.is_null).sort((a, b) => value(b) - value(a)).slice(0, 5);
            return (
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_14rem]">
                <div className="space-y-2">
                <Segmented<number> size="sm" label="Term" value={term} onChange={setTerm} options={d.terms.map((t) => ({ value: t, label: `${t}-month term` }))} />
                <Heatmap
                  label={`Arm selection share, ${term}-month term`}
                  rowTitle="Amount"
                  colTitle="Interest rate"
                  cellHeight={30}
                  rows={amounts.map((a) => ({ key: String(a), label: fmt.amountShort(a) }))}
                  cols={d.rates.map((r) => ({ key: String(r), label: fmt.rate(r) }))}
                  color={(v) => seqColor(Math.sqrt(v / max), theme)}
                  cell={(ri, ci) => {
                    const arm = rows.find((a) => a.amount === amounts[ri] && a.rate === d.rates[ci]);
                    if (!arm) return { value: null, na: true, ariaLabel: 'not in grid' };
                    const v = value(arm);
                    return {
                      value: v,
                      text: v >= 0.005 ? fmt.pct(v, v >= 0.1 ? 0 : 1) : undefined,
                      ariaLabel: `${arm.label}: ${fmt.pct(v, 1)} of decisions`,
                      tooltip: (
                        <div className="num space-y-0.5">
                          <div className="font-medium text-ink">{fmt.arm(arm)}</div>
                          <div>
                            {fmt.pct(arm.share, 2)} this epoch · {fmt.pct(arm.cum_share, 2)} cumulative
                          </div>
                          <div className="text-muted">
                            {fmt.int(arm.selections)} selections · {fmt.int(arm.cum_selections)} total
                          </div>
                          {arm.mu !== null && (
                            <div className="text-muted">
                              posterior μ {fmt.usd(arm.mu)} · σ {fmt.usd(arm.sigma)} · n {fmt.num(arm.n, 0)}
                            </div>
                          )}
                        </div>
                      ),
                    };
                  }}
                />
                </div>
                <div className="space-y-3 text-xs">
                  <div className="rounded-md border border-line p-3">
                    <div className="text-2xs text-muted">Decline (arm 0)</div>
                    <div className="text-lg font-semibold text-ink">{fmt.pct(decline ? value(decline) : 0, 1)}</div>
                    <div className="num text-2xs text-muted">{fmt.int(decline?.selections)} borrowers this epoch</div>
                  </div>
                  <div>
                    <div className="mb-1 text-2xs text-muted">Top arms ({mode === 'share' ? 'epoch' : 'cumulative'})</div>
                    <ol className="num space-y-1">
                      {top.map((a) => (
                        <li key={a.arm} className="flex justify-between gap-2">
                          <span className="text-ink-2">{fmt.arm(a)}</span>
                          <span className="font-medium text-ink">{fmt.pct(value(a), 1)}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                  <ScaleLegend title="Share of decisions (√ scale)" stops={seqStops(theme)} labels={['0', fmt.pct(max, 0)]} />
                </div>
              </div>
            );
          }}
        </QueryState>
      </div>
    </ChartFrame>
  );
}
