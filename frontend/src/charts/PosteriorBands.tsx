import { Area, CartesianGrid, ComposedChart, Line, Tooltip, XAxis, YAxis } from 'recharts';
import { fmt } from '@/lib/format';
import { ChartTooltip } from './ChartTooltip';
import { ACTIVE_DOT, AXIS, CURSOR_LINE, epochTick, GRID, indexTicks } from './kit';

export interface BandSeries {
  key: string;
  label: string;
  color: string;
  mu: (number | null)[];
  lo: (number | null)[];
  hi: (number | null)[];
}

/** Posterior μ per arm (2px line) with a μ ± σ band (10% wash), over the snapshot epochs. */
export function PosteriorBands({ epochs, series, height = 260 }: { epochs: string[]; series: BandSeries[]; height?: number }) {
  const data = epochs.map((epoch, i) => {
    const row: Record<string, number | string | [number, number] | null> = { i, epoch };
    for (const s of series) {
      row[s.key] = s.mu[i];
      const lo = s.lo[i];
      const hi = s.hi[i];
      row[`${s.key}_band`] = lo !== null && hi !== null ? [lo, hi] : null;
    }
    return row;
  });
  const names = new Map(series.map((s) => [s.key, s]));
  return (
    <ComposedChart responsive width="100%" height={height} data={data} margin={{ top: 10, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} />
      <XAxis type="number" dataKey="i" domain={[0, Math.max(epochs.length - 1, 1)]} ticks={indexTicks(epochs.length, 7)} tickFormatter={epochTick(epochs)} {...AXIS} />
      <YAxis tickFormatter={(v: number) => fmt.usdCompact(v)} width={52} {...AXIS} />
      <Tooltip
        cursor={CURSOR_LINE}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={p.payload?.filter((e) => !String(e.dataKey).endsWith('_band'))}
            label={p.label}
            format={{
              label: (l) => fmt.epoch(epochs[Number(l)]),
              value: (v) => `μ ${fmt.usd(v)}`,
              name: (k) => names.get(k)?.label ?? k,
              color: (k) => names.get(k)?.color ?? 'var(--text-axis)',
            }}
          />
        )}
      />
      {series.map((s) => (
        <Area key={`${s.key}_band`} type="linear" dataKey={`${s.key}_band`} stroke="none" fill={s.color} fillOpacity={0.1} isAnimationActive={false} connectNulls activeDot={false} />
      ))}
      {series.map((s) => (
        <Line key={s.key} type="linear" dataKey={s.key} stroke={s.color} strokeWidth={2} dot={{ r: 2.5, strokeWidth: 0, fill: s.color }} activeDot={ACTIVE_DOT} isAnimationActive={false} connectNulls />
      ))}
    </ComposedChart>
  );
}
