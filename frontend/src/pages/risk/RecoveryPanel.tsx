import { useCalibration } from '@/api/client';
import { HBars } from '@/charts/Bars';
import { ChartFrame } from '@/charts/ChartFrame';
import { DensityChart, HIST_SWATCH } from '@/charts/Density';
import { Legend } from '@/charts/Legend';
import { QueryState } from '@/components/ui/States';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';

/** Stage 2: R | D ~ Beta(α^R, β^R) by method of moments, against holdout realized recovery rates. */
export function RecoveryPanel({ fitId }: { fitId: number }) {
  const q = useCalibration(fitId, true);
  return (
    <QueryState query={q} loadingLabel="Loading recovery stage">
      {(c) => {
        const r = c.recovery;
        return (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <ChartFrame
              title="Recovery given default"
              caption={`Global Beta(${fmt.num(r.alpha, 2)}, ${fmt.num(r.beta, 2)}) fitted by method of moments on ${fmt.int(r.n_defaults)} defaulted training loans (line), over holdout realized recovery rates (columns, density-scaled).`}
              legend={<Legend items={[{ key: 'pdf', label: 'Fitted Beta density', color: 'var(--series-1)' }, { key: 'h', label: 'Holdout realized (density)', color: HIST_SWATCH, kind: 'rect' }]} />}
              yLabel="Density"
              xLabel="Recovery rate R = principal recovered / funded amount"
              table={{
                columns: [{ key: 'b', label: 'Bin' }, { key: 'n', label: 'Holdout loans', align: 'right' }],
                rows: r.histogram.map((h) => ({ b: `${fmt.pct(h.lo, 0)}–${fmt.pct(h.hi, 0)}`, n: fmt.int(h.count) })),
              }}
            >
              <DensityChart posterior={r.pdf} histogram={r.histogram} marks={[{ x: r.mean, label: `E[R] ${fmt.pct(r.mean, 0)}` }]} height={250} names={{ posterior: 'Beta density', prior: 'Prior' }} />
            </ChartFrame>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <StatTile label="E[R | D]" value={fmt.pct(r.mean)} hint={`α ${fmt.num(r.alpha, 2)} · β ${fmt.num(r.beta, 2)}`} />
                <StatTile label="Defaults (train)" value={fmt.int(r.n_defaults)} />
                <StatTile label="MAE" value={fmt.num(r.mae, 3)} hint="holdout" />
                <StatTile label="RMSE" value={fmt.num(r.rmse, 3)} hint="holdout" />
              </div>
              <ChartFrame
                title="Realized recovery by grade"
                caption="Mean holdout recovery rate of defaulted loans per grade."
                table={{ columns: [{ key: 'g', label: 'Grade' }, { key: 'm', label: 'Mean', align: 'right' }, { key: 'n', label: 'n', align: 'right' }], rows: r.by_grade.map((g) => ({ g: g.grade, m: fmt.pct(g.mean), n: fmt.int(g.n) })) }}
              >
                <div className="px-2 pb-2">
                  <HBars rows={r.by_grade.map((g) => ({ label: `Grade ${g.grade}`, value: g.mean, note: `n ${g.n}` }))} format={(v) => fmt.pct(v, 1)} max={1} />
                </div>
              </ChartFrame>
            </div>
          </div>
        );
      }}
    </QueryState>
  );
}
