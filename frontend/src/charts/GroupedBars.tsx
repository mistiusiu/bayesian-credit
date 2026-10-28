import { Bar, BarChart, CartesianGrid, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts';
import { ChartTooltip } from './ChartTooltip';
import { AXIS, CURSOR_BAND, GRID } from './kit';

/** Grouped columns for ≤ 3 series sharing one axis (2px gap inside a group). */
export function GroupedBars({
  data,
  series,
  height = 240,
  format,
  yTick,
  domain,
  yWidth = 56,
}: {
  data: Record<string, number | string>[];
  series: { key: string; label: string; color: string }[];
  height?: number;
  format: (v: number) => string;
  yTick: (v: number) => string;
  /** Fixed y domain (e.g. shared across small multiples). */
  domain?: [number, number];
  yWidth?: number;
}) {
  const names = new Map(series.map((s) => [s.key, s]));
  return (
    <BarChart responsive width="100%" height={height} data={data} margin={{ top: 10, right: 12, bottom: 2, left: 2 }} barGap={2} barCategoryGap="22%">
      <CartesianGrid {...GRID} />
      <XAxis dataKey="x" {...AXIS} />
      <YAxis tickFormatter={yTick} width={yWidth} domain={domain} {...AXIS} />
      <ReferenceLine y={0} stroke="var(--axis)" />
      <Tooltip
        cursor={CURSOR_BAND}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip active={p.active} payload={p.payload} label={p.label} format={{ value: format, name: (k) => names.get(k)?.label ?? k, color: (k) => names.get(k)?.color ?? '' }} />
        )}
      />
      {series.map((s) => (
        <Bar key={s.key} dataKey={s.key} fill={s.color} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
      ))}
    </BarChart>
  );
}
