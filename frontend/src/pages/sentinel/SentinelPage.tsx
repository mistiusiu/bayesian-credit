import { useSentinel, useSimulation } from '@/api/client';
import type { SentinelTimeline } from '@/api/types';
import { BreakerStrip } from '@/charts/BreakerStrip';
import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { Empty, Loading, QueryState } from '@/components/ui/States';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';
import { BREAKER_LABEL, BREAKER_TONE, LEVEL_TONE } from '@/lib/palette';
import { useParam } from '@/lib/params';
import { useSelection } from '@/lib/selection';
import { BucketTable } from './BucketTable';
import { PsiCharts } from './PsiCharts';
import { ReviewQueue } from './ReviewQueue';

function Timeline({ t, simId }: { t: SentinelTimeline; simId: number }) {
  const epochs = t.aggregate.map((a) => a.epoch);
  const trip = t.transitions.find((x) => x.to_state === 'open')?.epoch ?? null;
  const [feature, setFeature] = useParam('feature', t.current.top_feature ?? t.features[0]);
  const [epoch, setEpoch] = useParam('epoch', trip ?? t.current.epoch ?? epochs[epochs.length - 1] ?? '');
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
        <StatTile label="Breaker" value={BREAKER_LABEL[t.current.breaker_state]} status={{ tone: BREAKER_TONE[t.current.breaker_state], label: t.current.breaker_state === 'open' ? 'frozen' : 'adapting' }} hint={t.current.epoch ? `at ${fmt.epoch(t.current.epoch)}` : undefined} />
        <StatTile label={`PSI now (${t.aggregate_mode})`} value={fmt.num(t.current.psi_aggregate, 3)} status={{ tone: LEVEL_TONE[t.current.level], label: t.current.level }} hint={`top: ${t.current.top_feature ?? '—'}`} />
        <StatTile label="Thresholds" value={`${t.thresholds.watch} / ${t.thresholds.breaker}`} hint="watch ≥ · breaker >" />
        <StatTile label="Epochs at watch" value={fmt.int(t.shift_counts.watch)} hint={`of ${epochs.length}`} />
        <StatTile label="Epochs at breaker" value={fmt.int(t.shift_counts.breaker)} hint={`${t.transitions.filter((x) => x.to_state === 'open').length} trip(s)`} />
        <StatTile label="Reference window" value={`${t.reference.from.slice(0, 4)}–${t.reference.to.slice(0, 4)}`} hint={`${fmt.compact(t.reference.n)} loans${t.reference.rebased_at.length ? ` · rebased ${t.reference.rebased_at.join(', ')}` : ''}`} />
      </div>
      <Card title="Breaker state by epoch" subtitle="Closed → open when aggregate PSI exceeds the breaker threshold; a human acknowledgement moves it to half-open; consecutive normal readings close it. Select an epoch to inspect its buckets.">
        <BreakerStrip epochs={epochs} states={t.aggregate.map((a) => a.breaker_state)} levels={t.aggregate.map((a) => a.level)} selected={epoch} onSelect={setEpoch} />
        {t.transitions.length > 0 && (
          <ol className="num mt-3 space-y-0.5 text-xs text-ink-2">
            {t.transitions.map((x) => (
              <li key={`${x.epoch}-${x.to_state}`}>
                <span className="text-muted">{fmt.epoch(x.epoch)}</span> · {BREAKER_LABEL[x.from_state]} → <strong className="text-ink">{BREAKER_LABEL[x.to_state]}</strong> · {x.reason}
                {x.review !== null && <span className="text-muted"> · review #{x.review}</span>}
              </li>
            ))}
          </ol>
        )}
      </Card>
      <PsiCharts t={t} selected={feature} onSelect={setFeature} />
      <BucketTable simId={simId} features={t.features} epochs={epochs} feature={feature} epoch={epoch} onFeature={setFeature} onEpoch={setEpoch} />
    </div>
  );
}

export function SentinelPage() {
  const sel = useSelection();
  const simQ = useSimulation(sel.simId);
  const sim = simQ.data ?? sel.sim ?? undefined;
  const q = useSentinel(sel.simId, sim);
  return (
    <>
      <PageHeader
        title="Sentinel"
        description="Population Stability Index per monitored feature against the training reference (Table 9: < 0.10 normal, 0.10–0.25 watch, > 0.25 circuit breaker). Breaker trips require human review: resume, resegment or roll back."
      />
      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_26rem]">
        <div className="min-w-0">
          {!sim && sel.loading ? (
            <Loading label="Resolving selection" height={240} />
          ) : !sim ? (
            <Empty title="No simulation selected">PSI is computed per epoch while a simulation runs.</Empty>
          ) : sim.epochs_done === 0 ? (
            <Empty title="No epochs yet" />
          ) : (
            <QueryState query={q} loadingLabel="Loading PSI timeline">{(t) => <Timeline t={t} simId={sim.id} />}</QueryState>
          )}
        </div>
        <ReviewQueue simId={sel.simId} />
      </div>
    </>
  );
}
