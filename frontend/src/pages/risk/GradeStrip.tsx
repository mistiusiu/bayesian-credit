import { useGroups } from '@/api/client';
import type { Fit } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { StripPlot } from '@/charts/StripPlot';
import { Empty, QueryState } from '@/components/ui/States';
import { GRADES } from '@/lib/domain';
import { fmt } from '@/lib/format';

export function GradeStrip({ fit }: { fit: Fit }) {
  const q = useGroups(fit.id, true, { level: 'group' });
  if (!fit.config.segmentation.dimensions.includes('grade')) return <Empty title="Grade is not a segmentation dimension of this fit" />;
  return (
    <QueryState query={q}>
      {(rows) => {
        const points = rows.map((r) => ({ key: r.key, category: GRADES.indexOf(r.dims.grade as (typeof GRADES)[number]), value: r.mean, n: r.n_obs, label: fmt.group(r.key) })).filter((p) => p.category >= 0);
        return (
          <ChartFrame
            title="Posterior default rate per group, by grade"
            caption="Each dot is a borrower group (grade × other dimensions); the bar marks the grade median. Risk rises A → G; sparse groups sit near the global mean because of shrinkage."
            yLabel="Posterior mean P(default)"
            xLabel="Grade"
            table={{
              columns: [{ key: 'g', label: 'Group' }, { key: 'n', label: 'n', align: 'right' }, { key: 'm', label: 'Posterior mean', align: 'right' }],
              rows: [...points].sort((a, b) => a.category - b.category || b.n - a.n).map((p) => ({ g: p.label, n: fmt.int(p.n), m: fmt.pct(p.value) })),
            }}
          >
            <StripPlot points={points} categories={[...GRADES]} height={300} yTick={(v) => fmt.pct(v, 0)} format={(v) => fmt.pct(v, 1)} />
          </ChartFrame>
        );
      }}
    </QueryState>
  );
}
