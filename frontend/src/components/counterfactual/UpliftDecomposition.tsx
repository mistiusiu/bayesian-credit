import type { UpliftDecomposition as Decomposition } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Legend } from '@/charts/Legend';
import { Waterfall, type WaterfallBar } from '@/charts/Waterfall';
import { decompositionSteps, upliftFraction } from '@/lib/counterfactual';
import { fmt } from '@/lib/format';
import { divColor, policyColor } from '@/lib/palette';
import { useTheme } from '@/lib/theme';

const UP = 'color-mix(in oklab, var(--series-1) 58%, var(--surface-1))';

/** Historical → + repricing → + resizing → + term choice → shadow, as a waterfall of expected profit. */
export function UpliftDecomposition({ d, className }: { d: Decomposition; className?: string }) {
  const { theme } = useTheme();
  const steps = decompositionSteps(d);
  const hist = d.expected_profit_historical;
  const total = d.unrestricted.expected_profit - hist;
  const down = divColor(-1, theme);
  const bars: WaterfallBar[] = steps.map((s) => {
    const change = s.to - s.from;
    const color = s.key === 'historical' ? policyColor('historical') : s.key === 'shadow' ? 'var(--series-1)' : change >= 0 ? UP : down;
    const cumulative = upliftFraction(s.to - hist, hist);
    const share = s.kind === 'delta' && Math.abs(total) > 0 ? change / total : null;
    return {
      key: s.key,
      label: s.label,
      kind: s.kind,
      from: s.from,
      to: s.to,
      color,
      valueLabel: s.kind === 'total' ? fmt.usdCompact(s.to) : fmt.usdCompactSigned(change),
      ariaLabel: `${s.label}: ${s.kind === 'total' ? fmt.usd(s.to) : `${fmt.usdSigned(change)}, reaching ${fmt.usd(s.to)}`}`,
      tooltip: (
        <div className="num space-y-0.5">
          <div className="font-medium text-ink">{s.label.replace('+ ', '')}</div>
          <div>Expected profit {fmt.usd(s.to)}</div>
          {s.kind === 'delta' && <div>Step {fmt.usdSigned(change)}{share !== null && ` · ${fmt.pct(share, 0)} of the total uplift`}</div>}
          {s.key !== 'historical' && <div>Cumulative uplift {fmt.pctSigned(cumulative, 0)} of historical</div>}
          <div className="text-muted">{s.note}</div>
        </div>
      ),
    };
  });
  const hasDown = bars.some((b) => b.kind === 'delta' && b.to < b.from);
  return (
    <ChartFrame
      className={className}
      title="Where the uplift comes from"
      caption={`Expected profit when the shadow engine may change only the rate (historical amount and term kept), then rate and amount (term kept), then everything; each step takes the best admissible arm. Total uplift ${fmt.usdCompactSigned(total)} (${fmt.pctSigned(upliftFraction(total, hist), 0)} of historical).`}
      legend={
        <Legend
          items={[
            { key: 'h', label: 'Historical total', color: policyColor('historical'), kind: 'rect' },
            { key: 'up', label: 'Step adds profit', color: UP, kind: 'rect' },
            ...(hasDown ? [{ key: 'down', label: 'Step loses profit', color: down, kind: 'rect' as const }] : []),
            { key: 's', label: 'Shadow total', color: 'var(--series-1)', kind: 'rect' },
          ]}
        />
      }
      yLabel="USD, expected profit of the portfolio"
      table={{
        columns: [
          { key: 's', label: 'Step' },
          { key: 'p', label: 'Expected profit', align: 'right' },
          { key: 'c', label: 'Change', align: 'right' },
          { key: 'u', label: 'Uplift vs historical', align: 'right' },
        ],
        rows: steps.map((s) => ({
          s: s.label,
          p: fmt.usd(s.to),
          c: s.kind === 'delta' ? fmt.usdSigned(s.to - s.from) : '—',
          u: s.key === 'historical' ? '—' : `${fmt.usdSigned(s.to - hist)} (${fmt.pctSigned(upliftFraction(s.to - hist, hist), 1)})`,
        })),
      }}
    >
      <Waterfall bars={bars} tick={fmt.usdCompact} label="Uplift decomposition: historical, repricing, resizing, term choice, shadow" height={250} />
    </ChartFrame>
  );
}
