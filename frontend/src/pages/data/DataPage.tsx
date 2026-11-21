import { useState } from 'react';
import type { Dataset } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { Pill, StatusBadge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
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
  // "New dataset from #n": the ingest form re-mounts pre-filled with that dataset's parameters
  const [template, setTemplate] = useState<{ ds: Dataset; nonce: number } | null>(null);
  const excluded = ds?.params.exclude_variables ?? [];
  return (
    <>
      <PageHeader
        eyebrow="Pipeline · Data"
        title="Data"
        description="Lending Club 2007–2018 (or its synthetic twin) reduced to the 60 retained Table 3 variables, or any subset of them for ablations. The lifecycle stage of each variable is the leakage firewall: the bandit only ever sees pre-decision columns."
      />
      <IngestForm
        key={template ? `from-${template.ds.id}-${template.nonce}` : 'new'}
        template={template?.ds ?? null}
        onCreated={(d) => {
          setTemplate(null);
          sel.setDataset(d.id);
        }}
      />
      <DatasetCards datasets={sel.datasets} selectedId={sel.datasetId} onSelect={sel.setDataset} loading={sel.loading} />
      {ds && (
        <section className="mt-5" aria-label={`Dataset ${ds.name}`}>
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-sm font-semibold text-ink">
              #{ds.id} {ds.name}
            </h2>
            <StatusBadge status={ds.status} />
            {excluded.length > 0 && <Pill className="max-w-md truncate">without {excluded.join(', ')}</Pill>}
            <Button size="sm" variant="ghost" icon="plus" className="ml-auto" onClick={() => {
                setTemplate({ ds, nonce: Date.now() });
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}>
              New dataset from #{ds.id}
            </Button>
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
