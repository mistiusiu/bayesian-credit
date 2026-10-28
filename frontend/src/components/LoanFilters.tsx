import { Chips } from '@/components/ui/Chips';
import { Select, TextInput } from '@/components/ui/Field';
import { GRADES, HOME_OWNERSHIP, PURPOSES } from '@/lib/domain';
import { fmt } from '@/lib/format';
import type { LoanFilterState } from '@/lib/loanFilters';

/** One filter row above the table it scopes (grade · purpose · home ownership · resolved · loan id). */
export function LoanFilters({ value, onChange }: { value: LoanFilterState; onChange: (v: LoanFilterState) => void }) {
  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
      <div className="flex flex-col gap-1">
        <span className="text-2xs font-medium text-muted">Grade</span>
        <Chips size="sm" label="Grade" options={GRADES.map((g) => ({ value: g, label: g }))} value={value.grade} onChange={(grade) => onChange({ ...value, grade })} />
      </div>
      <label className="flex flex-col gap-1 text-2xs font-medium text-muted">
        Purpose
        <Select
          className="h-7 w-44 text-xs"
          value={value.purpose[0] ?? ''}
          onValue={(v) => onChange({ ...value, purpose: v ? [v] : [] })}
          options={[{ value: '', label: 'Any purpose' }, ...PURPOSES.map((p) => ({ value: p, label: fmt.label(p) }))]}
        />
      </label>
      <label className="flex flex-col gap-1 text-2xs font-medium text-muted">
        Home ownership
        <Select
          className="h-7 w-36 text-xs"
          value={value.home_ownership[0] ?? ''}
          onValue={(v) => onChange({ ...value, home_ownership: v ? [v] : [] })}
          options={[{ value: '', label: 'Any' }, ...HOME_OWNERSHIP.map((h) => ({ value: h, label: fmt.label(h.toLowerCase()) }))]}
        />
      </label>
      <label className="flex flex-col gap-1 text-2xs font-medium text-muted">
        Outcome
        <Select
          className="h-7 w-32 text-xs"
          value={value.resolved}
          onValue={(resolved) => onChange({ ...value, resolved })}
          options={[
            { value: 'all', label: 'All loans' },
            { value: 'true', label: 'Resolved' },
            { value: 'false', label: 'Unresolved' },
          ]}
        />
      </label>
      <label className="flex flex-col gap-1 text-2xs font-medium text-muted">
        Loan id
        <TextInput className="h-7 w-36 text-xs" placeholder="contains…" value={value.search} inputMode="numeric" onChange={(e) => onChange({ ...value, search: e.target.value })} />
      </label>
    </div>
  );
}
