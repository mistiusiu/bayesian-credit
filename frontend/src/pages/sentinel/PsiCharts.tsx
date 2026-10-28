import type { SentinelTimeline } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { epochTick, yearTicks } from '@/charts/kit';
import { Legend } from '@/charts/Legend';
import { SeriesLines } from '@/charts/SeriesLines';
import { Button } from '@/components/ui/Button';
import { fmt } from '@/lib/format';

export function PsiCharts({ t, selected, onSelect }: { t: SentinelTimeline; selected: string; onSelect: (f: string) => void }) {
  const epochs = t.aggregate.map((a) => a.epoch);
  const xs = epochs.map((_, i) => i);
  const top = Math.max(0.35, ...t.features_psi.map((p) => p.psi), ...t.aggregate.map((a) => a.aggregate)) * 1.08;
  const domain: [number, number] = [0, Math.ceil(top * 20) / 20];
  const bands = [
    { y1: t.thresholds.watch, y2: t.thresholds.breaker, fill: 'var(--band-watch)' },
    { y1: t.thresholds.breaker, y2: domain[1], fill: 'var(--band-breaker)' },
  ];
  const tick = epochTick(epochs);
  const ticks = yearTicks(epochs);
  return (
    <div className="space-y-4">
      <ChartFrame
        title="Aggregate PSI"
        caption={`${t.aggregate_mode === 'max' ? 'Maximum' : 'Mean'} PSI across ${t.features.length} features per epoch (window ${t.window_months} months). Amber band = watch (${t.thresholds.watch}–${t.thresholds.breaker}), red band = breaker (> ${t.thresholds.breaker}).`}
        legend={
          <Legend
            items={[
              { key: 'agg', label: `Aggregate (${t.aggregate_mode})`, color: 'var(--series-1)' },
              { key: 'mean', label: 'Mean across features', color: 'var(--text-muted)' },
              { key: 'w', label: 'Watch band', color: 'var(--band-watch)', kind: 'band' },
              { key: 'b', label: 'Breaker band', color: 'var(--band-breaker)', kind: 'band' },
            ]}
          />
        }
        yLabel="PSI = Σ (a − e)·ln(a / e)"
        xLabel="Epoch"
        table={{
          columns: [{ key: 'e', label: 'Epoch' }, { key: 'a', label: 'Aggregate', align: 'right' }, { key: 'm', label: 'Mean', align: 'right' }, { key: 'l', label: 'Level' }, { key: 'b', label: 'Breaker' }, { key: 'f', label: 'Top feature' }],
          rows: t.aggregate.map((a) => ({ e: fmt.epoch(a.epoch), a: fmt.num(a.aggregate, 3), m: fmt.num(a.psi_mean, 3), l: a.level, b: a.breaker_state, f: a.top_feature })),
        }}
      >
        <SeriesLines
          xs={xs}
          ticks={ticks}
          xTick={tick}
          xLabelFor={(i) => fmt.epoch(epochs[i])}
          yTick={(v) => fmt.num(v, 2)}
          format={(v) => fmt.num(v, 3)}
          height={240}
          yDomain={domain}
          bands={bands}
          thresholds={[
            { y: t.thresholds.watch, label: `watch ${t.thresholds.watch}` },
            { y: t.thresholds.breaker, label: `breaker ${t.thresholds.breaker}` },
          ]}
          lines={[
            { key: 'agg', label: 'Aggregate', color: 'var(--series-1)', values: t.aggregate.map((a) => a.aggregate) },
            { key: 'mean', label: 'Mean', color: 'var(--text-muted)', values: t.aggregate.map((a) => a.psi_mean), width: 1.5 },
          ]}
        />
      </ChartFrame>
      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
        {t.features.map((f) => {
          const pts = t.features_psi.filter((p) => p.feature === f);
          const last = pts[pts.length - 1];
          return (
            <ChartFrame
              key={f}
              className={f === selected ? 'ring-2 ring-[var(--focus)]' : undefined}
              title={<code>{f}</code>}
              caption={`PSI per epoch on a shared scale; latest ${fmt.num(last?.psi, 3)} (${last?.level ?? '—'}).`}
              actions={
                <Button size="sm" variant={f === selected ? 'secondary' : 'ghost'} onClick={() => onSelect(f)} aria-pressed={f === selected}>
                  Buckets
                </Button>
              }
              table={{ columns: [{ key: 'e', label: 'Epoch' }, { key: 'p', label: 'PSI', align: 'right' }, { key: 'l', label: 'Level' }], rows: pts.map((p) => ({ e: fmt.epoch(p.epoch), p: fmt.num(p.psi, 3), l: p.level })) }}
            >
              <SeriesLines
                xs={xs}
                ticks={ticks}
                xTick={tick}
                xLabelFor={(i) => fmt.epoch(epochs[i])}
                yTick={(v) => fmt.num(v, 1)}
                format={(v) => fmt.num(v, 3)}
                height={120}
                yDomain={domain}
                bands={bands}
                lines={[{ key: 'psi', label: f, color: 'var(--series-1)', values: pts.map((p) => p.psi) }]}
              />
            </ChartFrame>
          );
        })}
      </div>
    </div>
  );
}
