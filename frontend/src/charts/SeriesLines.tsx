import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceDot, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts';
import { ChartTooltip } from './ChartTooltip';
import { ACTIVE_DOT, AXIS, CURSOR_LINE, GRID, indexTicks, niceDomain } from './kit';

export interface LineSpec {
  key: string;
  label: string;
  color: string;
  values: (number | null)[];
  width?: number;
}

export interface Band {
  y1: number;
  y2?: number;
  fill: string;
  label?: string;
}

/**
 * Generic numeric-x line chart (t = 0…n, epochs as indices) with optional horizontal
 * threshold bands, x-range shading and point annotations. Single axis only.
 */
export function SeriesLines({
  xs,
  lines,
  xTick,
  xLabelFor,
  yTick,
  format,
  height = 220,
  bands,
  xRanges,
  thresholds,
  dots,
  yDomain,
  ticks,
  includeZero = true,
}: {
  xs: number[];
  lines: LineSpec[];
  xTick: (x: number) => string;
  xLabelFor?: (x: number) => string;
  yTick: (v: number) => string;
  format: (v: number) => string;
  height?: number;
  bands?: Band[];
  xRanges?: { x1: number; x2: number; fill: string }[];
  thresholds?: { y: number; label: string }[];
  dots?: { x: number; y: number; label: string }[];
  yDomain?: [number, number];
  ticks?: number[];
  includeZero?: boolean;
}) {
  const data = xs.map((x, i) => {
    const row: Record<string, number | null> = { x };
    for (const l of lines) row[l.key] = l.values[i] ?? null;
    return row;
  });
  const values = lines.flatMap((l) => l.values.filter((v): v is number => v !== null));
  const domain = yDomain ?? niceDomain([...values, ...(thresholds?.map((t) => t.y) ?? [])], { includeZero });
  const labels = new Map(lines.map((l) => [l.key, l]));
  return (
    <LineChart responsive width="100%" height={height} data={data} margin={{ top: 10, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} />
      {bands?.map((b) => (
        <ReferenceArea key={`${b.y1}-${b.y2}`} y1={b.y1} y2={b.y2 ?? domain[1]} fill={b.fill} fillOpacity={1} ifOverflow="hidden" />
      ))}
      {xRanges?.map((r) => <ReferenceArea key={`${r.x1}-${r.x2}`} x1={r.x1} x2={r.x2} fill={r.fill} fillOpacity={1} ifOverflow="hidden" />)}
      <XAxis
        type="number"
        dataKey="x"
        domain={[xs[0] ?? 0, xs[xs.length - 1] ?? 1]}
        ticks={ticks ?? indexTicks(xs.length, 7).map((i) => xs[i])}
        tickFormatter={xTick}
        {...AXIS}
      />
      <YAxis domain={domain} tickFormatter={yTick} width={52} {...AXIS} />
      {thresholds?.map((t) => (
        <ReferenceLine
          key={t.y}
          y={t.y}
          stroke="var(--text-axis)"
          strokeWidth={1}
          label={{ value: t.label, position: 'insideTopLeft', fill: 'var(--text-muted)', fontSize: 10 }}
        />
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
              label: (l) => (xLabelFor ? xLabelFor(Number(l)) : xTick(Number(l))),
              value: (v) => format(v),
              name: (k) => labels.get(k)?.label ?? k,
              color: (k) => labels.get(k)?.color ?? 'var(--text-axis)',
            }}
          />
        )}
      />
      {lines.map((l) => (
        <Line key={l.key} type="linear" dataKey={l.key} stroke={l.color} strokeWidth={l.width ?? 2} dot={false} activeDot={ACTIVE_DOT} isAnimationActive={false} connectNulls />
      ))}
      {dots?.map((d) => (
        <ReferenceDot
          key={`${d.x}-${d.y}`}
          x={d.x}
          y={d.y}
          r={4.5}
          fill="var(--text-primary)"
          stroke="var(--surface-1)"
          strokeWidth={2}
          label={{ value: d.label, position: 'left', fill: 'var(--text-secondary)', fontSize: 11 }}
        />
      ))}
    </LineChart>
  );
}
