import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts';
import type { DensityPoint } from '@/api/types';
import { fmt } from '@/lib/format';
import { ChartTooltip } from './ChartTooltip';
import { ACTIVE_DOT, AXIS, CURSOR_LINE, GRID } from './kit';

/** Histogram bin fill: a muted ink wash that reads in both themes. */
export const HIST_FILL = 'var(--text-axis)';
/** Legend swatch matching the translucent bins. */
export const HIST_SWATCH = 'color-mix(in srgb, var(--text-axis) 30%, transparent)';
const HIST_OPACITY = 0.3;

/**
 * Beta densities on [0, 1]: the posterior as a 2px line over a 10% wash, an optional prior
 * (gray line) and an optional empirical histogram drawn as exact bin rectangles
 * (density-scaled, same axis, 2px surface gap between bins).
 */
export function DensityChart({
  posterior,
  prior,
  histogram,
  marks,
  height = 200,
  xMax = 1,
  color = 'var(--series-1)',
  names = { posterior: 'Posterior', prior: 'Prior' },
}: {
  posterior: DensityPoint[];
  prior?: DensityPoint[];
  histogram?: { lo: number; hi: number; count: number }[];
  marks?: { x: number; label: string }[];
  height?: number;
  xMax?: number;
  color?: string;
  names?: { posterior: string; prior: string };
}) {
  const total = histogram?.reduce((a, b) => a + b.count, 0) ?? 0;
  const bins = (histogram ?? []).map((b) => ({ ...b, density: total ? b.count / total / (b.hi - b.lo) : 0 }));
  const rows = new Map<number, Record<string, number>>();
  const put = (x: number, k: string, v: number) => {
    const key = Math.round(x * 1000) / 1000;
    const row = rows.get(key) ?? { x: key };
    row[k] = v;
    rows.set(key, row);
  };
  posterior.filter((p) => p.x <= xMax).forEach((p) => put(p.x, 'posterior', p.density));
  prior?.filter((p) => p.x <= xMax).forEach((p) => put(p.x, 'prior', p.density));
  const data = [...rows.values()].sort((a, b) => a.x - b.x);
  const top = Math.max(...data.map((d) => Math.max(d.posterior ?? 0, d.prior ?? 0)), ...bins.map((b) => b.density), 0.1) * 1.08;
  const gap = 0.0025 * xMax;
  return (
    <ComposedChart responsive width="100%" height={height} data={data} margin={{ top: 16, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} />
      <XAxis type="number" dataKey="x" domain={[0, xMax]} tickFormatter={(v: number) => fmt.pct(v, 0)} {...AXIS} />
      <YAxis domain={[0, top]} tickFormatter={(v: number) => fmt.num(v, v >= 10 ? 0 : 1)} width={36} {...AXIS} />
      {bins.map((b) => (
        <ReferenceArea key={b.lo} x1={b.lo + gap} x2={b.hi - gap} y1={0} y2={b.density} fill={HIST_FILL} fillOpacity={HIST_OPACITY} stroke="none" ifOverflow="hidden" />
      ))}
      <Tooltip
        cursor={CURSOR_LINE}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={p.payload}
            label={p.label}
            format={{
              label: (l) => `x = ${fmt.pct(Number(l), 1)}`,
              value: (v) => fmt.num(v, 2),
              name: (k) => names[k as keyof typeof names] ?? k,
              extra: (d) => {
                const x = Number(d.x);
                const bin = bins.find((b) => x >= b.lo && x < b.hi);
                return bin ? `holdout bin ${fmt.pct(bin.lo, 0)}–${fmt.pct(bin.hi, 0)}: ${fmt.int(bin.count)} loans` : null;
              },
            }}
          />
        )}
      />
      <Area type="monotone" dataKey="posterior" stroke={color} strokeWidth={2} fill={color} fillOpacity={0.1} activeDot={ACTIVE_DOT} isAnimationActive={false} connectNulls />
      {prior && <Line type="monotone" dataKey="prior" stroke="var(--text-muted)" strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls />}
      {marks?.map((m) => (
        <ReferenceLine key={m.label} x={m.x} stroke="var(--text-axis)" label={{ value: m.label, position: 'insideTopRight', fill: 'var(--text-muted)', fontSize: 10.5 }} />
      ))}
    </ComposedChart>
  );
}
