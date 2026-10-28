import type { Mctm } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Heatmap } from '@/charts/Heatmap';
import { ScaleLegend } from '@/charts/ScaleLegend';
import { STATE_LABEL } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { seqColor, seqStops } from '@/lib/palette';
import { useTheme } from '@/lib/theme';

const p3 = (v: number) => (v > 0 && v < 0.001 ? '<0.001' : fmt.num(v, 3));

/** 4×4 posterior transition matrix: probability (√-scaled sequential colour) and observed loan-month counts. */
export function TransitionMatrix({ m }: { m: Mctm }) {
  const { theme } = useTheme();
  const states = m.states;
  return (
    <ChartFrame
      title="Transition matrix P"
      caption={`Row = state this month, column = next month. Posterior mean of Dirichlet rows (prior ${m.prior_strength} per allowed cell); counts are training loan-months. Gray cells are structural zeros (proposal) or absorbing rows.`}
      legend={<ScaleLegend title="P(next | current), √ scale" stops={seqStops(theme)} labels={['0', '0.25', '1']} />}
      table={{
        columns: [{ key: 'from', label: 'From \\ to' }, ...states.map((s) => ({ key: s, label: STATE_LABEL[s], align: 'right' as const }))],
        rows: states.map((from, i) => ({
          from: STATE_LABEL[from],
          ...Object.fromEntries(states.map((to, j) => [to, m.structural_zeros[i][j] ? '0 (structural)' : `${p3(m.matrix[i][j])} (${fmt.int(m.counts[i][j])})`])),
        })),
      }}
    >
      <div className="px-2 pb-1">
        <Heatmap
          label="Transition probabilities"
          rowTitle="Current state"
          colTitle="Next state"
          cellHeight={58}
          rowHeaderWidth="6rem"
          rows={states.map((s) => ({ key: s, label: STATE_LABEL[s] }))}
          cols={states.map((s) => ({ key: s, label: STATE_LABEL[s] }))}
          color={(v) => seqColor(Math.sqrt(v), theme)}
          cell={(i, j) => {
            const zero = m.structural_zeros[i][j];
            const absorbing = i >= 2;
            const v = m.matrix[i][j];
            const label = `${STATE_LABEL[states[i]]} to ${STATE_LABEL[states[j]]}`;
            if (zero)
              return { value: null, na: true, text: '0', sub: absorbing ? 'absorbing' : 'structural', ariaLabel: `${label}: structurally zero` };
            return {
              value: v,
              text: p3(v),
              sub: absorbing ? 'absorbing' : fmt.compact(m.counts[i][j]),
              ariaLabel: `${label}: probability ${p3(v)}, ${fmt.int(m.counts[i][j])} loan-months`,
              tooltip: (
                <div className="space-y-0.5">
                  <div className="font-medium text-ink">{label}</div>
                  <div className="num">
                    P = {p3(v)} <span className="text-muted">[{p3(m.matrix_low[i][j])}, {p3(m.matrix_high[i][j])}]</span>
                  </div>
                  {!absorbing && <div className="num text-muted">{fmt.int(m.counts[i][j])} loan-months</div>}
                </div>
              ),
            };
          }}
        />
      </div>
    </ChartFrame>
  );
}
