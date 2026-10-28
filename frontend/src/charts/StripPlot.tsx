import { CartesianGrid, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { fmt } from '@/lib/format';
import { ChartTooltip } from './ChartTooltip';
import { AXIS, GRID } from './kit';

export interface StripPoint {
  key: string;
  category: number;
  value: number;
  n: number;
  label: string;
}

/**
 * Strip plot: one dot per group at its category (jittered deterministically), plus the
 * category median as a short ink tick. Hit areas are enlarged beyond the 8px marks.
 */
export function StripPlot({ points, categories, height = 260, yTick, format }: { points: StripPoint[]; categories: string[]; height?: number; yTick: (v: number) => string; format: (v: number) => string }) {
  const jittered = points.map((p, i) => ({ ...p, x: p.category + (((i * 0.6180339) % 1) - 0.5) * 0.5 }));
  const medians = categories.map((_, c) => {
    const vals = points.filter((p) => p.category === c).map((p) => p.value).sort((a, b) => a - b);
    return vals.length ? { x: c, value: vals[Math.floor(vals.length / 2)], key: `m${c}`, label: `median ${categories[c]}`, n: vals.length } : null;
  }).filter((m) => m !== null);
  return (
    <ScatterChart responsive width="100%" height={height} margin={{ top: 10, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} />
      <XAxis type="number" dataKey="x" domain={[-0.6, categories.length - 0.4]} ticks={categories.map((_, i) => i)} tickFormatter={(v: number) => categories[Math.round(v)] ?? ''} {...AXIS} />
      <YAxis type="number" dataKey="value" tickFormatter={yTick} width={44} {...AXIS} />
      <ZAxis range={[64, 64]} />
      <Tooltip
        cursor={false}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={p.payload?.filter((e) => e.dataKey === 'value')}
            format={{
              value: (v) => format(v),
              name: () => 'posterior mean',
              extra: (d) => `${String(d.label)} · n ${fmt.int(Number(d.n))}`,
            }}
          />
        )}
      />
      <Scatter data={jittered} dataKey="value" fill="var(--series-1)" fillOpacity={0.85} stroke="var(--surface-1)" strokeWidth={1.5} isAnimationActive={false} />
      <Scatter
        data={medians}
        dataKey="value"
        isAnimationActive={false}
        shape={(props: { cx?: number; cy?: number }) => (
          <line x1={(props.cx ?? 0) - 16} x2={(props.cx ?? 0) + 16} y1={props.cy} y2={props.cy} stroke="var(--text-primary)" strokeWidth={2} strokeLinecap="round" />
        )}
      />
    </ScatterChart>
  );
}
