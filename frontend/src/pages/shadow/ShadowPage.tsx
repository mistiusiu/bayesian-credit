import { useFit } from '@/api/client';
import { GuardrailStrip } from '@/components/fit/GuardrailStrip';
import { PageHeader } from '@/components/layout/PageHeader';
import { Empty, Loading } from '@/components/ui/States';
import { TabPanel, Tabs } from '@/components/ui/Tabs';
import { useParam } from '@/lib/params';
import { useSelection } from '@/lib/selection';
import { BorrowerPanel } from './BorrowerPanel';
import { PortfolioPanel } from './PortfolioPanel';

type Tab = 'borrower' | 'portfolio';

export function ShadowPage() {
  const sel = useSelection();
  const fit = useFit(sel.fitId).data ?? sel.fit;
  const [tab, setTab] = useParam('view', 'borrower');
  const ready = fit?.status === 'ready';
  return (
    <>
      <PageHeader
        title="Shadow engine"
        description="Counterfactual policy evaluation (G.4, Table 8): for a historical borrower, every candidate arm is scored with the posterior predictive (default, recovery, credit transitions) and compared with the loan Lending Club actually issued. Nothing simulated is presented as an observed outcome."
      />
      {!fit && sel.loading ? (
        <Loading label="Resolving selection" height={240} />
      ) : !fit || !ready ? (
        <Empty title={fit ? 'Fit is still running' : 'No fit selected'}>The shadow engine needs a ready Two-Stage Beta-Beta fit.</Empty>
      ) : (
        <>
          <div className="mb-4">
            <GuardrailStrip
              guardrails={fit.metrics?.guardrails ?? {
                max_payment_to_income: fit.config.arm_space.max_payment_to_income ?? null,
                max_amount_to_income: fit.config.arm_space.max_amount_to_income ?? null,
                max_total_dti: fit.config.arm_space.max_total_dti ?? null,
                rate_corridor: fit.config.arm_space.rate_corridor ?? null,
              }}
              source={fit.metrics?.guardrails ? 'metrics' : 'config'}
              reference={fit.metrics?.reference_rates}
            />
          </div>
          <Tabs<Tab>
            scope="shadow"
            label="Shadow engine views"
            value={tab as Tab}
            onChange={setTab}
            tabs={[
              { id: 'borrower', label: 'Borrower' },
              { id: 'portfolio', label: 'Portfolio' },
            ]}
          />
          <div className="pt-4">
            <TabPanel scope="shadow" id={tab}>
              {tab === 'borrower' ? <BorrowerPanel fit={fit} datasetId={fit.dataset} /> : <PortfolioPanel fit={fit} />}
            </TabPanel>
          </div>
        </>
      )}
    </>
  );
}
