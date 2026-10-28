import type { BorrowerContext, ShadowReport } from '@/api/types';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { fmt } from '@/lib/format';

type Key = keyof BorrowerContext;
const FIELDS: { key: Key; label: string; f?: (v: number) => string }[] = [
  { key: 'grade', label: 'Grade' },
  { key: 'sub_grade', label: 'Sub-grade' },
  { key: 'fico_range_low', label: 'FICO (low)' },
  { key: 'annual_inc', label: 'Annual income', f: fmt.usd },
  { key: 'dti', label: 'DTI', f: (v) => `${fmt.num(v, 1)}%` },
  { key: 'emp_length', label: 'Employment', f: (v) => `${v} y` },
  { key: 'home_ownership', label: 'Home' },
  { key: 'purpose', label: 'Purpose' },
  { key: 'verification_status', label: 'Verification' },
  { key: 'application_type', label: 'Application' },
  { key: 'revol_util', label: 'Revolving util.', f: (v) => fmt.pct(v, 0) },
  { key: 'revol_bal', label: 'Revolving balance', f: fmt.usd },
  { key: 'open_acc', label: 'Open accounts' },
  { key: 'total_acc', label: 'Total accounts' },
  { key: 'inq_last_6mths', label: 'Inquiries 6m' },
  { key: 'delinq_2yrs', label: 'Delinquencies 2y' },
  { key: 'pub_rec', label: 'Public records' },
  { key: 'mort_acc', label: 'Mortgage accounts' },
  { key: 'tot_cur_bal', label: 'Total balance', f: fmt.usdCompact },
  { key: 'bc_util', label: 'Bankcard util.', f: (v) => fmt.pct(v, 0) },
  { key: 'pct_tl_nvr_dlq', label: 'Never delinquent', f: (v) => fmt.pct(v, 0) },
  { key: 'installment', label: 'Installment (hist.)', f: fmt.usd2 },
];

/** X_i exactly as the engine saw it: pre-decision columns only (LeakageGuard). */
export function ContextCard({ report }: { report: ShadowReport }) {
  const c = report.context;
  return (
    <Card
      title={`Borrower context${report.loan_id ? ` · loan ${report.loan_id}` : ''}`}
      subtitle={`${report.issue_d ? `issued ${fmt.epoch(report.issue_d)} · ` : ''}group ${fmt.group(report.group)}`}
      actions={
        <Badge tone={report.leakage_safe ? 'good' : 'critical'} title="Every column passed LeakageGuard.assert_allowed(…, PRE_DECISION)">
          {report.leakage_safe ? `Leakage-safe · ${report.context_columns.length} pre-decision columns` : 'Leakage check failed'}
        </Badge>
      }
    >
      <dl className="num grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
        {FIELDS.map(({ key, label, f }) => {
          const v = c[key];
          const text = v === null ? '—' : typeof v === 'number' ? (f ? f(v) : fmt.num(v, Number.isInteger(v) ? 0 : 2)) : fmt.label(v.toLowerCase());
          return (
            <div key={key} className="flex min-w-0 justify-between gap-2 border-b border-line/60 pb-1">
              <dt className="truncate text-muted">{label}</dt>
              <dd className="truncate text-ink">{text}</dd>
            </div>
          );
        })}
      </dl>
      <p className="mt-3 text-2xs text-muted">Policy, disbursement, interim and terminal columns are withheld from the engine; the outcome below is shown for evaluation only.</p>
    </Card>
  );
}
