import type { ReactNode } from 'react';
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from 'recharts';
import { ChartTooltip } from './ChartTooltip';
import { AXIS, CURSOR_BAND, GRID } from './kit';

export interface BarDatum {
  x: string;
  value: number;
  [extra: string]: string | number;
}

/**
 * Vertical columns for one measure (slot-1 hue, ≤ 24px, 4px rounded data end, 2px surface
 * gap between neighbours). `xTicks` picks which category labels to print.
 */
export function Columns({
  data,
  height = 200,
  format,
  yTick,
  xTick,
  xTicks,
  label,
  extra,
  color = 'var(--series-1)',
}: {
  data: BarDatum[];
  height?: number;
  format: (v: number) => string;
  yTick: (v: number) => string;
  xTick?: (x: string) => string;
  xTicks?: string[];
  label: (x: string) => string;
  extra?: (d: Record<string, unknown>) => ReactNode;
  color?: string;
}) {
  return (
    <BarChart responsive width="100%" height={height} data={data} margin={{ top: 8, right: 8, bottom: 2, left: 2 }} barCategoryGap={2}>
      <CartesianGrid {...GRID} />
      <XAxis dataKey="x" ticks={xTicks} interval={xTicks ? 0 : 'preserveStartEnd'} tickFormatter={xTick} {...AXIS} />
      <YAxis tickFormatter={yTick} width={48} {...AXIS} />
      <Tooltip
        cursor={CURSOR_BAND}
        isAnimationActive={false}
        content={(p) => <ChartTooltip active={p.active} payload={p.payload} label={p.label} format={{ label: (l) => label(String(l)), value: format, name: () => '', extra }} />}
      />
      <Bar dataKey="value" fill={color} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
    </BarChart>
  );
}

/** Horizontal bars in plain HTML: label · bar (value at the tip) — every value directly labelled. */
export function HBars({ rows, format, max, color = 'var(--series-1)' }: { rows: { label: string; value: number; note?: string }[]; format: (v: number) => string; max?: number; color?: string }) {
  const top = max ?? Math.max(...rows.map((r) => r.value), 1);
  return (
    <ul className="space-y-1.5">
      {rows.map((r) => (
        <li key={r.label} className="grid grid-cols-[9.5rem_1fr] items-center gap-2 text-xs" title={r.note}>
          <span className="truncate text-ink-2">{r.label}</span>
          <span className="flex items-center gap-2">
            <span className="h-3 rounded-r-[4px]" style={{ width: `${Math.max(0.5, (r.value / top) * 82)}%`, background: color }} />
            <span className="num shrink-0 text-ink">{format(r.value)}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}
