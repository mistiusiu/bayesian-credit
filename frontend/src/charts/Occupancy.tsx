import { Area, AreaChart, CartesianGrid, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts';
import type { CreditState, MctmTrajectoryRow } from '@/api/types';
import { STATE_LABEL, STATE_STACK } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { stateColor } from '@/lib/palette';
import { ChartTooltip } from './ChartTooltip';
import { AXIS, CURSOR_LINE, GRID, indexTicks } from './kit';

/** State occupancy π_t = π₀Pᵗ as a stacked area; band edges are 2px lines in the state colour. */
export function OccupancyChart({ rows, height = 260, maturity }: { rows: MctmTrajectoryRow[]; height?: number; maturity?: number }) {
  const ts = rows.map((r) => r.t);
  return (
    <AreaChart responsive width="100%" height={height} data={rows} margin={{ top: 10, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} />
      <XAxis type="number" dataKey="t" domain={[0, ts[ts.length - 1] ?? 1]} ticks={indexTicks(ts.length, 7).map((i) => ts[i])} {...AXIS} />
      <YAxis domain={[0, 1]} tickFormatter={(v: number) => fmt.pct(v, 0)} width={40} {...AXIS} />
      <Tooltip
        cursor={CURSOR_LINE}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={[...(p.payload ?? [])].reverse()}
            label={p.label}
            format={{ label: (l) => `Month ${String(l)}`, value: (v) => fmt.pct(v, 1), name: (k) => STATE_LABEL[k as CreditState] ?? k, color: (k) => stateColor(k as CreditState) }}
          />
        )}
      />
      {STATE_STACK.map((s) => (
        <Area key={s} type="monotone" dataKey={s} stackId="pi" stroke={stateColor(s)} strokeWidth={2} fill={stateColor(s)} fillOpacity={0.28} isAnimationActive={false} />
      ))}
      {maturity !== undefined && <ReferenceLine x={maturity} stroke="var(--text-axis)" label={{ value: 'maturity', position: 'insideTopRight', fill: 'var(--text-muted)', fontSize: 10 }} />}
    </AreaChart>
  );
}
