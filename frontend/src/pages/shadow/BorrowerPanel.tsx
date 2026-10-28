import { useState } from 'react';
import { useLoans, useShadowReport } from '@/api/client';
import type { Fit, LoanRecord } from '@/api/types';
import { LoanFilters } from '@/components/LoanFilters';
import { Card } from '@/components/ui/Card';
import { DataTable } from '@/components/ui/DataTable';
import { Pagination } from '@/components/ui/Pagination';
import { Empty, QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import { emptyLoanFilters, toLoanQuery, type LoanFilterState } from '@/lib/loanFilters';
import { useParam } from '@/lib/params';
import { ArmDetail } from './ArmDetail';
import { ArmRewardHeatmap } from './ArmRewardHeatmap';
import { ContextCard } from './ContextCard';
import { EligibilityStrip } from './EligibilityStrip';
import { RecommendationCard } from './RecommendationCard';

const LIMIT = 12;

export function BorrowerPanel({ fit, datasetId }: { fit: Fit; datasetId: number }) {
  const [filters, setFilters] = useState<LoanFilterState>(emptyLoanFilters);
  const [offset, setOffset] = useState(0);
  const [loanParam, setLoan] = useParam('loan', null);
  const [armParam, setArm] = useParam('arm', null);
  const loans = useLoans(datasetId, toLoanQuery(filters, offset, LIMIT), true);
  const loanId = loanParam === null ? (loans.data?.results[0]?.loan_id ?? null) : Number(loanParam);
  const report = useShadowReport(fit.id, true, loanId === null ? null : { loan_id: loanId });
  const arm = armParam === null ? null : Number(armParam);
  return (
    <div className="space-y-4">
      <Card title="Borrowers" subtitle="Historical loans; select one to evaluate every arm for that borrower">
        <LoanFilters
          value={filters}
          onChange={(f) => {
            setFilters(f);
            setOffset(0);
          }}
        />
        <div className="mt-3">
          <QueryState query={loans} loadingLabel="Loading borrowers">
            {(page) => (
              <div className="space-y-2">
                <DataTable<LoanRecord>
                  caption="Borrowers"
                  rows={page.results}
                  rowKey={(r) => r.loan_id}
                  selectedKey={loanId}
                  onRowClick={(r) => {
                    setLoan(String(r.loan_id));
                    setArm(null);
                  }}
                  empty="No borrowers match these filters"
                  columns={[
                    { key: 'id', header: 'Loan id', render: (r) => <code className="text-xs">{r.loan_id}</code> },
                    { key: 'issue', header: 'Issued', render: (r) => fmt.epochShort(r.issue_d) },
                    { key: 'g', header: 'Grade', render: (r) => r.context.sub_grade },
                    { key: 'group', header: 'Group', render: (r) => <span className="text-ink-2">{fmt.group(r.group)}</span> },
                    { key: 'inc', header: 'Income', align: 'right', render: (r) => fmt.usdCompact(r.context.annual_inc) },
                    { key: 'fico', header: 'FICO', align: 'right', render: (r) => r.context.fico_range_low ?? '—' },
                    { key: 'pol', header: 'Historical arm', render: (r) => fmt.arm({ amount: r.policy.loan_amnt, rate: r.policy.int_rate, term: r.policy.term }) },
                    { key: 'st', header: 'Outcome', render: (r) => <span className="text-ink-2">{r.outcome.loan_status}</span> },
                  ]}
                />
                <Pagination offset={offset} limit={LIMIT} count={page.count} onChange={setOffset} />
              </div>
            )}
          </QueryState>
        </div>
      </Card>
      {loanId === null ? (
        <Empty title="Select a borrower" />
      ) : (
        <QueryState query={report} loadingLabel={`Evaluating ${fit.n_arms ?? ''} arms for loan ${loanId}`} height={260}>
          {(r) => {
            const selected = r.arms.find((a) => a.arm === (arm ?? r.recommended_arm)) ?? r.arms[r.recommended_arm];
            return (
              <div className="space-y-4">
                <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
                  <ContextCard report={r} />
                  <RecommendationCard report={r} />
                </div>
                <EligibilityStrip report={r} />
                <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
                  <ArmRewardHeatmap report={r} selected={selected.arm} onSelect={(a) => setArm(String(a))} />
                  <ArmDetail report={r} arm={selected} />
                </div>
              </div>
            );
          }}
        </QueryState>
      )}
    </div>
  );
}
