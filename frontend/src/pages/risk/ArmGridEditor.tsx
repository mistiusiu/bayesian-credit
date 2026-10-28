import { Chips } from '@/components/ui/Chips';
import { Field, TextInput } from '@/components/ui/Field';
import { Toggle } from '@/components/ui/Toggle';
import { fmt } from '@/lib/format';
import { parseAmounts, parseRates } from './armGrid';

export interface ArmGridState {
  amounts: string;
  rates: string;
  terms: number[];
  includeNull: boolean;
}

/** Arm space A_k = (L_k, r_k) × term (Table 7): three lists plus the decline arm (guardrails live under "Priors & guardrails"). */
export function ArmGridEditor({ value, onChange, error }: { value: ArmGridState; onChange: (v: ArmGridState) => void; error?: string[] }) {
  const a = parseAmounts(value.amounts);
  const r = parseRates(value.rates);
  const n = a.values.length * r.values.length * value.terms.length + (value.includeNull ? 1 : 0);
  return (
    <fieldset className="space-y-3 rounded-md border border-line p-3">
      <legend className="px-1 text-xs font-semibold text-ink">Arm space</legend>
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Amounts (USD)" hint={a.error ?? a.values.map((v) => fmt.amountShort(v)).join(' · ')} error={a.error}>
          {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={value.amounts} onChange={(e) => onChange({ ...value, amounts: e.target.value })} />}
        </Field>
        <Field label="Rates (% per year)" hint={r.error ?? r.values.map((v) => fmt.rate(v)).join(' · ')} error={r.error}>
          {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={value.rates} onChange={(e) => onChange({ ...value, rates: e.target.value })} />}
        </Field>
      </div>
      <div className="flex flex-wrap items-end gap-5">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-ink-2">Terms (months)</span>
          <Chips<string> label="Terms" options={['36', '60'].map((t) => ({ value: t, label: `${t}m` }))} value={value.terms.map(String)} onChange={(t) => onChange({ ...value, terms: t.map(Number).sort((x, y) => x - y) })} />
        </div>
        <Toggle checked={value.includeNull} onChange={(v) => onChange({ ...value, includeNull: v })} label="Include decline (arm 0)" description="lets the policy choose not to lend (§1.5)" />
      </div>
      <p className="num text-xs text-ink-2">
        {a.values.length} amounts × {r.values.length} rates × {value.terms.length} terms{value.includeNull ? ' + decline' : ''} = <strong className="text-ink">{n} arms</strong>
      </p>
      {error && <p className="text-2xs text-delta-bad">{error.join(' ')}</p>}
    </fieldset>
  );
}
