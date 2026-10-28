import { useSimulation } from '@/api/client';
import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { Empty, Loading } from '@/components/ui/States';
import { useSelection } from '@/lib/selection';
import { ArmSelection } from './ArmSelection';
import { EventLog } from './EventLog';
import { PosteriorPanel } from './PosteriorPanel';
import { RunCharts } from './RunCharts';
import { RunHeader } from './RunHeader';
import { RunsTable } from './RunsTable';
import { SelectionMixCard } from './SelectionMixCard';
import { SimpsonPanel } from './SimpsonPanel';
import { SimulationForm } from './SimulationForm';

export function BanditPage() {
  const sel = useSelection();
  const simQ = useSimulation(sel.simId);
  const sim = simQ.data ?? sel.sim;
  return (
    <>
      <PageHeader
        title="Bandit lab"
        description="Historical borrowers replayed month by month; contextual Thompson sampling competes with UCB1, ε-greedy, random and the historical policy under delayed interim and terminal feedback."
      />
      <SimulationForm fitId={sel.fit?.status === 'ready' ? sel.fitId : null} onCreated={(s) => sel.setSim(s.id)} />
      <div className="mt-4">
        <Card title="Runs" subtitle={sel.fit ? `with fit #${sel.fit.id} ${sel.fit.name}` : undefined} bodyClassName="p-0">
          <RunsTable sims={sel.sims} selectedId={sel.simId} onSelect={sel.setSim} />
        </Card>
      </div>
      {sim ? (
        <section className="mt-5 space-y-4" aria-label={`Simulation ${sim.name}`}>
          <RunHeader sim={sim} />
          <RunCharts sim={sim} />
          {sim.epochs_done > 0 && (
            <>
              <ArmSelection sim={sim} />
              <SelectionMixCard sim={sim} />
              <div className="grid gap-4 2xl:grid-cols-2">
                <PosteriorPanel sim={sim} />
                <SimpsonPanel sim={sim} />
              </div>
            </>
          )}
          <EventLog sim={sim} />
        </section>
      ) : sel.loading ? (
        <Loading label="Resolving selection" height={200} />
      ) : (
        <Empty title="No simulation selected">Start one with the form above.</Empty>
      )}
    </>
  );
}
