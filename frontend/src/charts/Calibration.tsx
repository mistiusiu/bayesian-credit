import { CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import type { ReliabilityBin, RocPoint } from '@/api/types';
import { fmt } from '@/lib/format';
import { ChartTooltip } from './ChartTooltip';
import { ACTIVE_DOT, AXIS, CURSOR_LINE, GRID } from './kit';

/** Reliability diagram: observed default rate vs mean predicted P(default) per decile bin; diagonal = perfect calibration. */
export function ReliabilityChart({ bins, height = 240 }: { bins: ReliabilityBin[]; height?: number }) {
  const top = Math.min(1, Math.ceil(Math.max(...bins.flatMap((b) => [b.mean_predicted, b.observed_rate]), 0.05) * 20) / 20 + 0.05);
  const data = bins.map((b) => ({ x: b.mean_predicted, observed: b.observed_rate, n: b.n, bin: b.bin }));
  return (
    <ComposedChart responsive width="100%" height={height} data={data} margin={{ top: 10, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} vertical />
      <XAxis type="number" dataKey="x" domain={[0, top]} tickFormatter={(v: number) => fmt.pct(v, 0)} {...AXIS} />
      <YAxis type="number" domain={[0, top]} tickFormatter={(v: number) => fmt.pct(v, 0)} width={40} {...AXIS} />
      <ReferenceLine segment={[{ x: 0, y: 0 }, { x: top, y: top }]} stroke="var(--text-axis)" strokeWidth={1} ifOverflow="hidden" />
      <Tooltip
        cursor={false}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={p.payload}
            format={{
              value: (v) => fmt.pct(v, 1),
              name: () => 'observed',
              extra: (d) => `bin ${Number(d.bin) + 1} · predicted ${fmt.pct(Number(d.x), 1)} · n ${fmt.int(Number(d.n))}`,
            }}
          />
        )}
      />
      <Line dataKey="observed" stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} />
      <Scatter dataKey="observed" fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} isAnimationActive={false} />
    </ComposedChart>
  );
}

/** ROC curve on the holdout; the diagonal is a random ranking (AUC 0.5). */
export function RocChart({ roc, auc, height = 240 }: { roc: RocPoint[]; auc: number; height?: number }) {
  return (
    <LineChart responsive width="100%" height={height} data={roc} margin={{ top: 10, right: 16, bottom: 2, left: 2 }}>
      <CartesianGrid {...GRID} vertical />
      <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickFormatter={(v: number) => fmt.pct(v, 0)} {...AXIS} />
      <YAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => fmt.pct(v, 0)} width={40} {...AXIS} />
      <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-axis)" strokeWidth={1} />
      <Tooltip
        cursor={CURSOR_LINE}
        isAnimationActive={false}
        content={(p) => (
          <ChartTooltip
            active={p.active}
            payload={p.payload}
            label={p.label}
            format={{
              label: (l) => `FPR ${fmt.pct(Number(l), 1)}`,
              value: (v) => fmt.pct(v, 1),
              name: () => `TPR (AUC ${fmt.num(auc, 3)})`,
              extra: (d) => `threshold P(default) ≥ ${fmt.prob(Number(d.threshold))}`,
            }}
          />
        )}
      />
      <Line type="stepAfter" dataKey="tpr" stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={ACTIVE_DOT} isAnimationActive={false} />
    </LineChart>
  );
}
