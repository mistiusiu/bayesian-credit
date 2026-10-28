import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts';
import type { BreakerState, PolicyName } from '@/api/types';
import { policyLabel } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { policyColor } from '@/lib/palette';
import { ChartTooltip } from './ChartTooltip';
import { EndLabels } from './EndLabels';
import { ACTIVE_DOT, AXIS, breakerRuns, CURSOR_LINE, epochTick, GRID, niceDomain, RUN_FILL, yearTicks } from './kit';

export interface PolicySeries {
  policy: PolicyName;
  values: (number | null)[];
}

/** One line per policy over epochs; breaker-open (red wash) and half-open (amber wash) epochs shaded. */
export function PolicyLines({
  epochs,
  series,
  hidden,
  breaker,
  format,
  tick,
  height = 240,
  includeZero = true,
  priority,
}: {
  epochs: string[];
  series: PolicySeries[];
  hidden?: Set<string>;
  breaker?: BreakerState[];
  format: (v: number) => string;
  tick: (v: number) => string;
  height?: number;
  includeZero?: boolean;
  priority?: PolicyName[];
}) {
  const visible = series.filter((s) => !hidden?.has(s.policy));
  const data = epochs.map((epoch, i) => {
    const row: Record<string, number | string | null> = { i, epoch };
    for (const s of series) row[s.policy] = s.values[i] ?? null;
    return row;
  });
  const domain = niceDomain(visible.flatMap((s) => s.values.filter((v): v is number => v !== null)), { includeZero });
  const last = epochs.length - 1;
  const order = priority ?? visible.map((s) => s.policy);
  const ends = [...visible]
    .sort((a, b) => order.indexOf(a.policy) - order.indexOf(b.policy))
    .flatMap((s) => (s.values[last] != null ? [{ key: s.policy, label: policyLabel(s.policy), x: last, y: s.values[last] }] : []));
  return (
    <LineChart responsive width="100%" height={height} data={data} margin={{ top: 8, right: 76, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} />
      {breaker &&
        breakerRuns(breaker.slice(0, epochs.length)).map((r) => (
          <ReferenceArea key={`${r.state}-${r.from}`} x1={r.from - 0.5} x2={r.to + 0.5} fill={RUN_FILL[r.state]} fillOpacity={1} ifOverflow="hidden" />
        ))}
      <XAxis type="number" dataKey="i" domain={[-0.5, Math.max(last, 0) + 0.5]} ticks={yearTicks(epochs)} tickFormatter={epochTick(epochs)} {...AXIS} />
      <YAxis domain={domain} tickFormatter={tick} width={56} {...AXIS} />
      {domain[0] < 0 && <ReferenceLine y={0} stroke="var(--axis)" />}
      <Tooltip
        cursor={CURSOR_LINE}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={p.payload}
            label={p.label}
            format={{
              label: (l) => fmt.epoch(epochs[Number(l)]),
              value: (v) => format(v),
              name: (k) => policyLabel(k) ?? k,
              include: (k) => !hidden?.has(k),
            }}
          />
        )}
      />
      {visible.map((s) => (
        <Line
          key={s.policy}
          type="linear"
          dataKey={s.policy}
          stroke={policyColor(s.policy)}
          strokeWidth={2}
          dot={false}
          activeDot={ACTIVE_DOT}
          isAnimationActive={false}
          connectNulls
        />
      ))}
      <EndLabels items={ends} />
    </LineChart>
  );
}
