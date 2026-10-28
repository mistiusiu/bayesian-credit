import type { EvaluationAxis, EvaluationReport, PolicyName, ReportMetric } from '@/api/types';
import { Card } from '@/components/ui/Card';
import { policyLabel } from '@/lib/domain';
import { cn } from '@/lib/cn';
import { fmt, formatByUnit } from '@/lib/format';
import { policyColor } from '@/lib/palette';

const AXES: { key: EvaluationAxis; label: string }[] = [
  { key: 'predictive', label: 'Predictive performance' },
  { key: 'decision', label: 'Decision-making performance' },
  { key: 'adaptability', label: 'Adaptability' },
  { key: 'feasibility', label: 'Computational feasibility' },
];

function Delta({ m }: { m: ReportMetric }) {
  if (m.value === null || m.baseline === null) return <span className="text-muted">—</span>;
  const d = m.value - m.baseline;
  const good = m.better === null ? null : m.better === 'higher' ? d > 0 : d < 0;
  const text = m.unit === 'usd' || m.unit === 'usd_per_decision' ? fmt.usdCompactSigned(d) : m.unit === 'fraction' ? fmt.pctSigned(d, 2) : fmt.numSigned(d, 4);
  return <span className={cn('num font-medium', good === true ? 'text-delta-good' : good === false ? 'text-delta-bad' : 'text-ink-2')}>{text}</span>;
}

function ByPolicy({ m }: { m: ReportMetric }) {
  if (!m.by_policy) return null;
  const entries = Object.entries(m.by_policy) as [PolicyName, number][];
  return (
    <ul className="num mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-2xs text-muted">
      {entries.map(([p, v]) => (
        <li key={p} className="inline-flex items-center gap-1">
          <span aria-hidden="true" className="inline-block h-0.5 w-2.5 rounded-full" style={{ background: policyColor(p) }} />
          {policyLabel(p)} {formatByUnit(v, m.unit === 'usd_per_decision' ? 'usd' : m.unit)}
        </li>
      ))}
    </ul>
  );
}

/** Table 11 grid: evaluation axis → dimension → metrics with value, reference and Δ (coloured by better direction). */
export function Table11Grid({ report }: { report: EvaluationReport }) {
  return (
    <div className="space-y-5">
      {AXES.map((axis) => {
        const dims = report.dimensions.filter((d) => d.axis === axis.key);
        if (!dims.length) return null;
        return (
          <section key={axis.key} aria-label={axis.label} className="space-y-2">
            <h2 className="text-xs font-semibold tracking-wide text-muted uppercase">{axis.label}</h2>
            <div className="grid gap-3 xl:grid-cols-2 2xl:grid-cols-3">
              {dims.map((d) => (
                <Card key={d.key} title={d.label} bodyClassName="p-0">
                  <table className="num w-full text-xs">
                    <caption className="sr-only">{d.label} metrics</caption>
                    <thead className="bg-surface-2 text-2xs text-muted">
                      <tr>
                        <th scope="col" className="px-3 py-1.5 text-left font-medium">Metric</th>
                        <th scope="col" className="px-3 py-1.5 text-right font-medium">Value</th>
                        <th scope="col" className="px-3 py-1.5 text-right font-medium">Reference</th>
                        <th scope="col" className="px-3 py-1.5 text-right font-medium">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.metrics.map((m) => (
                        <tr key={m.key} className="border-t border-line align-top">
                          <td className="px-3 py-1.5 text-ink-2">
                            {m.label}
                            {m.better && <span className="ml-1 text-2xs text-muted">({m.better} is better)</span>}
                            {m.note && <span className="block text-2xs text-muted">{m.note}</span>}
                            <ByPolicy m={m} />
                          </td>
                          <td className="px-3 py-1.5 text-right font-semibold whitespace-nowrap text-ink">{formatByUnit(m.value, m.unit)}</td>
                          <td className="px-3 py-1.5 text-right whitespace-nowrap text-ink-2">
                            {m.baseline === null ? '—' : formatByUnit(m.baseline, m.unit)}
                            {m.baseline_label && <span className="block text-2xs text-muted">{m.baseline_label}</span>}
                          </td>
                          <td className="px-3 py-1.5 text-right whitespace-nowrap">
                            <Delta m={m} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
