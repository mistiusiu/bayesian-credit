import { useActiveJobs, useDataset, useEvents, useFit, useSimulation } from '@/api/client';
import { PageHeader } from '@/components/layout/PageHeader';
import { StatusBadge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { ProgressBar } from '@/components/ui/Progress';
import { Empty, Loading } from '@/components/ui/States';
import { leadPolicy } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { BREAKER_LABEL, BREAKER_TONE, LEVEL_TONE } from '@/lib/palette';
import { useSelection } from '@/lib/selection';
import { EventFeed } from './EventFeed';
import { KpiRow } from './KpiRow';
import { Leaderboard } from './Leaderboard';
import { LoopDiagram, type StageInfo } from './LoopDiagram';

export function OverviewPage() {
  const sel = useSelection();
  const ds = useDataset(sel.datasetId).data ?? sel.dataset;
  const fit = useFit(sel.fitId).data ?? sel.fit;
  const simQ = useSimulation(sel.simId);
  const sim = simQ.data ?? sel.sim;
  const events = useEvents(sel.simId, sim ?? undefined, { order: 'desc', limit: 14 });
  const jobs = useActiveJobs().data?.results ?? [];
  const m = fit?.metrics;
  const leadName = sim ? leadPolicy(sim.headline.map((h) => h.policy)) : undefined;
  const lead = sim?.headline.find((h) => h.policy === leadName) ?? sim?.headline[0];
  const q = (cond: boolean, good = true) => (cond ? (good ? 'good' : 'critical') : null) as StageInfo['tone'];

  const stages: StageInfo[] = [
    { key: 'ctx', title: 'Borrower context', to: '/data', metric: ds?.n_loans ? `${fmt.compact(ds.n_loans)} loans · 37 pre-decision vars` : 'no dataset', tone: ds ? (ds.status === 'ready' ? 'good' : 'info') : null },
    { key: 'risk', title: 'Bayesian risk estimation', to: '/risk', metric: m ? `Brier ${fmt.num(m.brier, 3)} · AUC ${fmt.num(m.auc, 3)}` : fit ? fmt.label(fit.status) : 'no fit', tone: fit ? (fit.status === 'ready' ? 'good' : 'info') : null },
    { key: 'mctm', title: 'Credit-state prediction', to: '/mctm', metric: fit ? `MCTM ${fit.config.mctm.structure} · 4 states` : '—', tone: q(!!m) },
    { key: 'reward', title: 'Risk-adjusted reward', to: '/shadow', metric: fit ? `R − E[L] − funding ${fmt.pct(fit.config.reward.funding_rate, 0)}` : '—' },
    { key: 'policy', title: 'Policy selection', to: '/bandit', metric: sim ? `Thompson · ${sim.epochs_done}/${sim.epochs_total} epochs` : 'no simulation', tone: sim ? (sim.status === 'running' ? 'info' : sim.status === 'completed' ? 'good' : sim.status === 'awaiting_review' ? 'warning' : sim.status === 'interrupted' ? 'critical' : null) : null },
    { key: 'outcomes', title: 'Observed outcomes', to: '/bandit', metric: sim ? `interim ${sim.config.interim_months}m · weight ${sim.config.interim_weight}` : '—' },
    { key: 'post', title: 'Posterior update', to: '/bandit', metric: lead ? `${fmt.compact(lead.n_decisions)} decisions learned` : '—' },
    { key: 'adapt', title: 'Policy adaptation', to: '/bandit', metric: lead ? `decline ${fmt.pct(lead.decline_rate)} · rate ${fmt.pct(lead.mean_rate)}` : '—' },
  ];

  return (
    <>
      <PageHeader
        title="Overview"
        description="The closed loop of the shadow lending engine: borrower context → hierarchical risk → credit-state prediction → risk-adjusted reward → Thompson sampling → delayed outcomes → posterior update, governed by the PSI Sentinel."
      />
      <KpiRow dataset={ds} fit={fit} sim={sim} />
      <div className="mt-4">
        <Card title="Closed loop" subtitle="Live status per stage; select a node to open its page. Dashed = counterfactual (shadow) path; the frame is the PSI Sentinel spanning every stage.">
          {sel.loading && !ds ? (
            <Loading height={360} />
          ) : (
            <div className="mx-auto max-w-[1280px]">
              <LoopDiagram
                stages={stages}
                sentinel={{
                  to: '/sentinel',
                  tone: sim ? BREAKER_TONE[sim.breaker_state] : 'neutral',
                  label: sim ? `breaker ${BREAKER_LABEL[sim.breaker_state].toLowerCase()} · PSI ${fmt.num(sim.psi_aggregate, 3)} ${sim.sentinel_level}` : 'no simulation selected',
                  breakerOpen: sim?.breaker_state === 'open',
                }}
                historical={{ key: 'hist', title: 'Historical policy', to: '/shadow', metric: 'LC arms · reference' }}
                shadow={{ key: 'shadow', title: 'Shadow engine', to: '/shadow', metric: sim?.uplift_vs_historical != null ? `uplift ${fmt.usdCompactSigned(sim.uplift_vs_historical)}` : '129 arms per borrower' }}
                center={sim ? [`Epoch ${fmt.epoch(sim.current_epoch)}`, `${fmt.label(sim.status)} · ${fmt.pct(sim.progress, 0)} · ${sim.config.policies.length} policies`] : ['No simulation', 'create one in the Bandit lab']}
              />
            </div>
          )}
        </Card>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Card title="Latest events" subtitle={sim ? `Simulation #${sim.id} event store (newest first)` : undefined} bodyClassName="p-0">
          {sel.loading && !sim ? <Loading /> : !sim ? <Empty title="No simulation selected" /> : <EventFeed query={events} max={10} />}
        </Card>
        <Card title="Active jobs" subtitle="Background work polled every few seconds" bodyClassName="p-0">
          {jobs.length === 0 ? (
            <Empty title="No running jobs" height={72} />
          ) : (
            <ul className="divide-y divide-line">
              {jobs.map((j) => (
                <li key={j.id} className="px-4 py-2.5">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="font-medium text-ink">
                      {fmt.label(j.kind)} #{j.target?.id ?? j.id}
                    </span>
                    <StatusBadge status={j.status} />
                  </div>
                  <p className="mt-1 truncate text-2xs text-muted">{j.message}</p>
                  <ProgressBar value={j.progress} label={`${j.kind} progress`} className="mt-1.5" />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
      <div className="mt-4">
        <Card title="Policy leaderboard" subtitle={sim ? `${sim.name} · through ${fmt.epoch(sim.current_epoch)}` : undefined}>
          {sim ? <Leaderboard sim={sim} /> : sel.loading ? <Loading /> : <Empty title="No simulation" />}
        </Card>
      </div>
      {sim && sim.sentinel_level !== 'normal' && (
        <p className="mt-3 text-2xs text-muted">
          Sentinel level is <strong className="text-ink-2">{sim.sentinel_level}</strong> ({LEVEL_TONE[sim.sentinel_level]}): see the Sentinel page for the PSI breakdown and review queue.
        </p>
      )}
    </>
  );
}
