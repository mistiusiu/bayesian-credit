import { useFit } from '@/api/client';
import { PageHeader } from '@/components/layout/PageHeader';
import { StatusBadge } from '@/components/ui/Badge';
import { InterestRealizationCard } from '@/components/fit/InterestRealizationCard';
import { PolicyEffectCard } from '@/components/fit/PolicyEffectCard';
import { Card } from '@/components/ui/Card';
import { ProgressBar } from '@/components/ui/Progress';
import { Empty } from '@/components/ui/States';
import { TabPanel, Tabs } from '@/components/ui/Tabs';
import { useParam } from '@/lib/params';
import { useSelection } from '@/lib/selection';
import { CalibrationPanel } from './CalibrationPanel';
import { FitForm } from './FitForm';
import { FitList } from './FitList';
import { FitMetrics } from './FitMetrics';
import { GradeStrip } from './GradeStrip';
import { HierarchyExplorer } from './HierarchyExplorer';
import { RecoveryPanel } from './RecoveryPanel';

type Tab = 'hierarchy' | 'calibration' | 'recovery' | 'grades';

export function RiskPage() {
  const sel = useSelection();
  const fitQ = useFit(sel.fitId);
  const fit = fitQ.data ?? sel.fit;
  const [tab, setTab] = useParam('tab', 'hierarchy');
  return (
    <>
      <PageHeader
        title="Risk models"
        description="Two-Stage Beta-Beta: hierarchical Beta-Bernoulli default (global → group → leaf, κ shrinkage) and Beta method-of-moments recovery given default; validated on a temporal holdout."
      />
      <FitForm datasetId={sel.datasetId} onCreated={(f) => sel.setFit(f.id)} />
      <div className="mt-4">
        <Card title="Fits" subtitle={sel.dataset ? `on dataset #${sel.dataset.id} ${sel.dataset.name}` : undefined} bodyClassName="p-0">
          <FitList fits={sel.fits} selectedId={sel.fitId} onSelect={sel.setFit} loading={sel.loading} />
        </Card>
      </div>
      {fit && fit.status !== 'ready' && (
        <Card className="mt-4" title={`#${fit.id} ${fit.name}`} actions={<StatusBadge status={fit.status} />}>
          <ProgressBar value={fit.job?.progress ?? 0} label="Fit progress" />
          <p className="mt-2 text-xs text-muted">{fit.job?.message}</p>
        </Card>
      )}
      {fit?.status === 'ready' && fit.metrics && (
        <section className="mt-5 space-y-4" aria-label={`Fit ${fit.name}`}>
          <FitMetrics fit={fit} metrics={fit.metrics} />
          <div className="grid gap-4 xl:grid-cols-2">
            <PolicyEffectCard pe={fit.metrics.policy_effect} />
            <InterestRealizationCard ir={fit.metrics.interest_realization} />
          </div>
          <Tabs<Tab>
            scope="risk"
            label="Fit views"
            value={tab as Tab}
            onChange={setTab}
            tabs={[
              { id: 'hierarchy', label: 'Hierarchy explorer' },
              { id: 'calibration', label: 'Calibration & ROC' },
              { id: 'recovery', label: 'Recovery' },
              { id: 'grades', label: 'Groups by grade' },
            ]}
          />
          <TabPanel scope="risk" id={tab}>
            {tab === 'hierarchy' && <HierarchyExplorer fit={fit} />}
            {tab === 'calibration' && <CalibrationPanel fitId={fit.id} />}
            {tab === 'recovery' && <RecoveryPanel fitId={fit.id} />}
            {tab === 'grades' && <GradeStrip fit={fit} />}
          </TabPanel>
        </section>
      )}
      {!fit && !sel.loading && <Empty title="No fit for this dataset">Fit the Two-Stage Beta-Beta model with the form above.</Empty>}
    </>
  );
}
