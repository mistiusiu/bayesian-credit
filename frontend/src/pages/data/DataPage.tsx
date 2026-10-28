import { PageHeader } from '@/components/layout/PageHeader';
import { StatusBadge } from '@/components/ui/Badge';
import { TabPanel, Tabs } from '@/components/ui/Tabs';
import { Empty } from '@/components/ui/States';
import { useParam } from '@/lib/params';
import { useSelection } from '@/lib/selection';
import { DatasetCards } from './DatasetCards';
import { Distributions } from './Distributions';
import { IngestForm } from './IngestForm';
import { LoanBrowser } from './LoanBrowser';
import { StageTable } from './StageTable';

type Tab = 'stages' | 'distributions' | 'loans';

export function DataPage() {
  const sel = useSelection();
  const [tab, setTab] = useParam('tab', 'stages');
  const ds = sel.dataset;
  const ready = ds?.status === 'ready';
  return (
    <>
      <PageHeader
        title="Data"
        description="Lending Club 2007–2018 (or its synthetic twin) reduced to the 60 retained Table 3 variables. The lifecycle stage of each variable is the leakage firewall: the bandit only ever sees pre-decision columns."
      />
      <IngestForm onCreated={(d) => sel.setDataset(d.id)} />
      <DatasetCards datasets={sel.datasets} selectedId={sel.datasetId} onSelect={sel.setDataset} loading={sel.loading} />
      {ds && (
        <section className="mt-5" aria-label={`Dataset ${ds.name}`}>
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-sm font-semibold text-ink">
              #{ds.id} {ds.name}
            </h2>
            <StatusBadge status={ds.status} />
          </div>
          {!ready ? (
            <Empty title="Dataset is still being built">Variables, distributions and loans appear once the ingest job finishes. Progress is polled automatically.</Empty>
          ) : (
            <>
              <Tabs<Tab>
                scope="data"
                label="Dataset views"
                value={tab as Tab}
                onChange={setTab}
                tabs={[
                  { id: 'stages', label: 'Table 3 stages' },
                  { id: 'distributions', label: 'Distributions' },
                  { id: 'loans', label: 'Loan browser' },
                ]}
              />
              <div className="pt-4">
                {tab === 'stages' && (
                  <TabPanel scope="data" id="stages">
                    <StageTable datasetId={ds.id} />
                  </TabPanel>
                )}
                {tab === 'distributions' && (
                  <TabPanel scope="data" id="distributions">
                    <Distributions datasetId={ds.id} />
                  </TabPanel>
                )}
                {tab === 'loans' && (
                  <TabPanel scope="data" id="loans">
                    <LoanBrowser datasetId={ds.id} />
                  </TabPanel>
                )}
              </div>
            </>
          )}
        </section>
      )}
    </>
  );
}
