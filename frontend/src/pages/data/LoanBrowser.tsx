import { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { useLoans } from '@/api/client';
import type { LoanRecord } from '@/api/types';
import { LoanFilters } from '@/components/LoanFilters';
import { StatusBadge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { DataTable } from '@/components/ui/DataTable';
import { Pagination } from '@/components/ui/Pagination';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import { emptyLoanFilters, toLoanQuery, type LoanFilterState } from '@/lib/loanFilters';
import { selectionSearch } from '@/lib/selection';

const LIMIT = 25;

const outcomeStatus = (r: LoanRecord) => (!r.outcome.resolved ? 'running' : r.outcome.defaulted ? 'failed' : 'succeeded');

export function LoanBrowser({ datasetId }: { datasetId: number }) {
  const [filters, setFilters] = useState<LoanFilterState>(emptyLoanFilters);
  const [offset, setOffset] = useState(0);
  const q = useLoans(datasetId, toLoanQuery(filters, offset, LIMIT), true);
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const open = (r: LoanRecord) => {
    const s = new URLSearchParams(selectionSearch(params));
    s.set('loan', String(r.loan_id));
    void navigate(`/shadow?${s.toString()}`);
  };
  return (
    <Card title="Loan browser" subtitle="Context, policy and outcome columns per loan. Select a row to evaluate it in the shadow engine.">
      <LoanFilters
        value={filters}
        onChange={(f) => {
          setFilters(f);
          setOffset(0);
        }}
      />
      <div className="mt-3">
        <QueryState query={q} loadingLabel="Loading loans">
          {(page) => (
            <div className="space-y-2">
              <DataTable<LoanRecord>
                caption="Loans"
                rows={page.results}
                rowKey={(r) => r.loan_id}
                onRowClick={open}
                empty="No loans match these filters"
                columns={[
                  { key: 'id', header: 'Loan id', render: (r) => <code className="text-xs">{r.loan_id}</code> },
                  { key: 'issue', header: 'Issued', render: (r) => fmt.epochShort(r.issue_d) },
                  { key: 'grade', header: 'Grade', render: (r) => r.context.sub_grade },
                  { key: 'purpose', header: 'Purpose', render: (r) => <span className="text-ink-2">{fmt.label(r.context.purpose)}</span> },
                  { key: 'home', header: 'Home', render: (r) => <span className="text-ink-2">{fmt.label(r.context.home_ownership?.toLowerCase())}</span> },
                  { key: 'inc', header: 'Income', align: 'right', render: (r) => fmt.usdCompact(r.context.annual_inc) },
                  { key: 'dti', header: 'DTI', align: 'right', render: (r) => fmt.num(r.context.dti, 1) },
                  { key: 'fico', header: 'FICO', align: 'right', render: (r) => r.context.fico_range_low ?? '—' },
                  { key: 'amt', header: 'Amount', align: 'right', render: (r) => fmt.usd(r.policy.loan_amnt) },
                  { key: 'rate', header: 'Rate', align: 'right', render: (r) => fmt.pct(r.policy.int_rate, 2) },
                  { key: 'term', header: 'Term', align: 'right', render: (r) => `${r.policy.term}m` },
                  { key: 'status', header: 'Status', render: (r) => <StatusBadge status={outcomeStatus(r)} label={r.outcome.loan_status} /> },
                  { key: 'reward', header: 'Realized reward', align: 'right', render: (r) => fmt.usdSigned(r.outcome.realized_reward) },
                ]}
              />
              <Pagination offset={offset} limit={LIMIT} count={page.count} onChange={setOffset} />
            </div>
          )}
        </QueryState>
      </div>
    </Card>
  );
}
