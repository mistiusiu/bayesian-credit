import { useState } from 'react';
import { useTimeseries } from '@/api/client';
import type { BreakerState, PolicyName, Simulation, Timeseries, TimeseriesRow } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Legend } from '@/charts/Legend';
import { PolicyLines } from '@/charts/PolicyLines';
import { QueryState } from '@/components/ui/States';
import { policyLabel } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { policyColor } from '@/lib/palette';

type Metric = 'cum_reward' | 'cum_regret' | 'rolling_reward' | 'decline_rate';

const SPECS: { metric: Metric; title: string; caption: string; y: string; format: (v: number) => string; tick: (v: number) => string }[] = [
  { metric: 'cum_reward', title: 'Cumulative risk-adjusted reward', caption: 'Σ expected reward of the arms each policy chose; the slope is reward per epoch.', y: 'USD, cumulative', format: fmt.usd, tick: fmt.usdCompact },
  { metric: 'cum_regret', title: 'Cumulative regret', caption: 'Σ pseudo-regret against the hindsight oracle (best arm per group). Learning policies flatten (concave); fixed ones stay linear.', y: 'USD, cumulative', format: fmt.usd, tick: fmt.usdCompact },
  { metric: 'rolling_reward', title: 'Rolling reward per decision', caption: 'Mean reward per decision over the trailing 6 epochs: convergence of each policy.', y: 'USD per decision', format: fmt.usd, tick: fmt.usdCompact },
  { metric: 'decline_rate', title: 'Decline rate', caption: 'Share of borrowers assigned the null arm (decline). The historical policy approved every loan it saw.', y: 'Share of decisions', format: (v) => fmt.pct(v, 2), tick: (v) => fmt.pct(v, v > 0 && v < 0.1 ? 1 : 0) },
];

function pivot(ts: Timeseries, metric: Metric) {
  const by = new Map<string, Map<PolicyName, TimeseriesRow>>();
  for (const r of ts.rows) {
    const m = by.get(r.epoch) ?? new Map<PolicyName, TimeseriesRow>();
    m.set(r.policy, r);
    by.set(r.epoch, m);
  }
  return ts.policies.map((p) => ({ policy: p, values: ts.epochs.map((e) => by.get(e)?.get(p)?.[metric] ?? null) }));
}

function Charts({ ts }: { ts: Timeseries }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const breaker = ts.epochs.map<BreakerState>((e) => ts.rows.find((r) => r.epoch === e)?.breaker_state ?? 'closed');
  const toggle = (k: string) =>
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Legend items={ts.policies.map((p) => ({ key: p, label: policyLabel(p), color: policyColor(p) }))} hidden={hidden} onToggle={toggle} />
        <Legend
          items={[
            { key: 'open', label: 'Breaker open (policies frozen)', color: 'var(--band-open)', kind: 'band' },
            { key: 'half', label: 'Half-open', color: 'var(--band-half)', kind: 'band' },
          ]}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {SPECS.map((s) => {
          const series = pivot(ts, s.metric);
          return (
            <ChartFrame
              key={s.metric}
              title={s.title}
              caption={s.caption}
              yLabel={s.y}
              xLabel="Epoch (issue month)"
              table={{
                columns: [{ key: 'e', label: 'Epoch' }, ...series.map((x) => ({ key: x.policy, label: policyLabel(x.policy), align: 'right' as const }))],
                rows: ts.epochs.map((e, i) => ({ e: fmt.epoch(e), ...Object.fromEntries(series.map((x) => [x.policy, x.values[i] === null ? '—' : s.format(x.values[i])])) })),
              }}
            >
              <PolicyLines epochs={ts.epochs} series={series} hidden={hidden} breaker={breaker} format={s.format} tick={s.tick} height={250} priority={['thompson', 'historical', 'random', 'ucb1', 'epsilon_greedy', 'greedy']} />
            </ChartFrame>
          );
        })}
      </div>
    </div>
  );
}

export function RunCharts({ sim }: { sim: Simulation }) {
  const q = useTimeseries(sim.id, sim);
  if (sim.epochs_done === 0) return <p className="py-8 text-center text-xs text-muted">Waiting for the first epoch…</p>;
  return <QueryState query={q} loadingLabel="Loading time series">{(ts) => <Charts ts={ts} />}</QueryState>;
}
