import { useDatasetSummary } from '@/api/client';
import type { DatasetSummary, FeatureSummary } from '@/api/types';
import { Columns, HBars } from '@/charts/Bars';
import { ChartFrame } from '@/charts/ChartFrame';
import { StageBadge } from '@/components/ui/Badge';
import { QueryState } from '@/components/ui/States';
import { fmt, tickByUnit } from '@/lib/format';

const binLabel = (f: Extract<FeatureSummary, { kind: 'numeric' }>, lo: number, hi: number) =>
  f.unit === 'usd' ? `${fmt.usdCompact(lo)}–${fmt.usdCompact(hi)}` : f.unit === 'fraction' ? `${fmt.pct(lo, 0)}–${fmt.pct(hi, 0)}` : `${fmt.num(lo, lo % 1 ? 1 : 0)}–${fmt.num(hi, hi % 1 ? 1 : 0)}`;

function FeatureCard({ f }: { f: FeatureSummary }) {
  if (f.kind === 'categorical') {
    const total = f.levels.reduce((a, l) => a + l.count, 0);
    return (
      <ChartFrame
        title={<span className="inline-flex items-center gap-2"><code>{f.feature}</code><StageBadge stage={f.stage} /></span>}
        caption={`Share of loans by level (top ${Math.min(6, f.levels.length)}).`}
        table={{ columns: [{ key: 'level', label: 'Level' }, { key: 'count', label: 'Loans', align: 'right' }, { key: 'share', label: 'Share', align: 'right' }], rows: f.levels.map((l) => ({ level: l.level, count: fmt.int(l.count), share: fmt.pct(l.count / total) })) }}
      >
        <div className="px-2 pb-1">
          <HBars rows={f.levels.slice(0, 6).map((l) => ({ label: fmt.label(l.level), value: l.count / total }))} format={(v) => fmt.pct(v, 1)} />
        </div>
      </ChartFrame>
    );
  }
  const data = f.bins.map((b) => ({ x: binLabel(f, b.lo, b.hi), value: b.count }));
  return (
    <ChartFrame
      title={<span className="inline-flex items-center gap-2"><code>{f.feature}</code><StageBadge stage={f.stage} /></span>}
      caption={`Median ${tickByUnit(f.p50, f.unit)} · 5–95% ${tickByUnit(f.p05, f.unit)}–${tickByUnit(f.p95, f.unit)} · null ${fmt.pct(f.null_share, 1)}`}
      yLabel="Loans"
      table={{ columns: [{ key: 'bin', label: 'Bin' }, { key: 'n', label: 'Loans', align: 'right' }], rows: data.map((d) => ({ bin: d.x, n: fmt.int(d.value) })) }}
    >
      <Columns data={data} height={120} format={fmt.int} yTick={fmt.compact} xTicks={[data[0]?.x, data[Math.floor(data.length / 2)]?.x, data[data.length - 1]?.x].filter((x): x is string => !!x)} xTick={(x) => x.split('–')[0]} label={(x) => x} />
    </ChartFrame>
  );
}

function SummaryView({ s }: { s: DatasetSummary }) {
  const arrivals = s.monthly_arrivals.map((a) => ({ x: a.epoch, value: a.n, resolved: a.n_resolved, defaults: a.n_default }));
  const statusRows = Object.entries(s.status_counts)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value: value / s.n_loans, note: `${fmt.int(value)} loans` }));
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <ChartFrame
          title="Monthly arrivals"
          caption="Loans issued per month (the simulation's decision epochs); volume ramps up from 2012 as in Lending Club."
          yLabel="Loans issued"
          xLabel="Issue month"
          table={{
            columns: [{ key: 'm', label: 'Month' }, { key: 'n', label: 'Issued', align: 'right' }, { key: 'r', label: 'Resolved', align: 'right' }, { key: 'd', label: 'Defaulted', align: 'right' }],
            rows: s.monthly_arrivals.map((a) => ({ m: fmt.epoch(a.epoch), n: fmt.int(a.n), r: fmt.int(a.n_resolved), d: fmt.int(a.n_default) })),
          }}
        >
          <Columns
            data={arrivals}
            height={220}
            format={fmt.int}
            yTick={fmt.compact}
            xTicks={arrivals.filter((a) => a.x.endsWith('-01')).map((a) => a.x)}
            xTick={(x) => x.slice(0, 4)}
            label={fmt.epoch}
            extra={(d) => `${fmt.int(Number(d.resolved))} resolved · ${fmt.int(Number(d.defaults))} defaulted`}
          />
        </ChartFrame>
        <ChartFrame
          title="Loan status"
          caption="Share of loans by loan_status at the snapshot; recent vintages are still Current (delayed outcomes)."
          table={{ columns: [{ key: 's', label: 'Status' }, { key: 'n', label: 'Share', align: 'right' }], rows: statusRows.map((r) => ({ s: r.label, n: fmt.pct(r.value, 2) })) }}
        >
          <div className="px-2 pt-1 pb-2">
            <HBars rows={statusRows} format={(v) => fmt.pct(v, 1)} />
          </div>
        </ChartFrame>
      </div>
      <h3 className="text-xs font-semibold tracking-wide text-muted uppercase">Feature distributions</h3>
      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-4">
        {s.features.map((f) => (
          <FeatureCard key={f.feature} f={f} />
        ))}
      </div>
    </div>
  );
}

export function Distributions({ datasetId }: { datasetId: number }) {
  const q = useDatasetSummary(datasetId, true);
  return <QueryState query={q} loadingLabel="Summarising dataset">{(s) => <SummaryView s={s} />}</QueryState>;
}
