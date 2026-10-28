import { useReport, useSimulation } from '@/api/client';
import type { EvaluationReport, Simulation } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Empty, Loading, QueryState } from '@/components/ui/States';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';
import { useSelection } from '@/lib/selection';
import { CounterfactualSection } from './CounterfactualSection';
import { SelectionMixSection } from './SelectionMixSection';
import { Table11Grid } from './Table11Grid';

function download(report: EvaluationReport) {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `evaluation-report-sim${report.simulation}-${report.generated_at.slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function ReportView({ r, sim }: { r: EvaluationReport; sim: Simulation }) {
  const rt = r.runtime;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="num flex flex-wrap items-center gap-2 text-xs text-ink-2">
          <Badge tone={r.status === 'final' ? 'good' : 'info'}>{r.status === 'final' ? 'Final report' : 'Partial (run not finished)'}</Badge>
          simulation #{r.simulation} · fit #{r.fit} · dataset #{r.dataset} · generated {fmt.dateTime(r.generated_at)}
        </p>
        <Button icon="download" onClick={() => download(r)}>
          Download JSON
        </Button>
      </div>
      <Table11Grid report={r} />
      <SelectionMixSection report={r} sim={sim} />
      <CounterfactualSection report={r} />
      <section aria-label="Runtime and feasibility" className="space-y-2">
        <h2 className="text-xs font-semibold tracking-wide text-muted uppercase">Runtime &amp; feasibility</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 2xl:grid-cols-8">
          <StatTile label="Wall time" value={fmt.duration(rt.wall_seconds)} hint={`${rt.epochs} epochs`} />
          <StatTile label="CPU time" value={fmt.duration(rt.cpu_seconds)} hint={`${fmt.num(rt.cpu_utilization, 2)} cores avg`} />
          <StatTile label="Peak memory" value={fmt.megabytes(rt.peak_memory_mb)} hint="worker RSS" />
          <StatTile label="Decisions" value={fmt.compact(rt.decisions)} hint={`${fmt.int(rt.decisions_per_second)} policy decisions / s`} />
          <StatTile label="Checkpoints" value={fmt.int(rt.checkpoints_written)} />
          <StatTile label="Restarts" value={fmt.int(rt.restarts)} hint="resume from checkpoint" />
          <StatTile label="Failed workloads" value={fmt.int(rt.failed_workloads)} />
          <StatTile label="Recovery time" value={fmt.duration(rt.recovery_seconds)} hint="kill → back past the kill point" />
        </div>
      </section>
    </div>
  );
}

export function EvaluationPage() {
  const sel = useSelection();
  const sim = useSimulation(sel.simId).data ?? sel.sim ?? undefined;
  const q = useReport(sel.simId, sim);
  return (
    <>
      <PageHeader
        title="Evaluation"
        description="Table 11 of the proposal: predictive performance, decision-making performance, adaptability and computational feasibility, with the historical policy (or the base rate) as the reference."
      />
      {!sim && sel.loading ? (
        <Loading label="Resolving selection" height={240} />
      ) : !sim ? (
        <Empty title="No simulation selected" />
      ) : sim.epochs_done === 0 ? (
        <Empty title="The report appears after the first epoch" />
      ) : (
        <QueryState query={q} loadingLabel="Building the evaluation report">
          {(r) => <ReportView r={r} sim={sim} />}
        </QueryState>
      )}
    </>
  );
}
