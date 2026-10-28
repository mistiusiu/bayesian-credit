import type { PolicyEffectFeature } from '@/api/types';
import { Chips } from '@/components/ui/Chips';
import { Field, NumberInput } from '@/components/ui/Field';
import { Toggle } from '@/components/ui/Toggle';
import { GUARDRAIL_INFO, POLICY_EFFECT_FEATURES } from '@/lib/domain';
import type { PriorsState } from './priors';

type Errs = Record<string, string[] | undefined>;

/** Fit form section: structured leaf prior, recovery by term, prepayment and the four lending guardrails (MATH §2.6–2.9). */
export function PriorsGuardrails({ value: v, onChange, errs }: { value: PriorsState; onChange: (v: PriorsState) => void; errs: Errs }) {
  const set = <K extends keyof PriorsState>(k: K, x: PriorsState[K]) => onChange({ ...v, [k]: x });
  const cfgErr = errs['hierarchy.policy_effect_cfg'];
  return (
    <fieldset className="rounded-md border border-line p-3">
      <legend className="px-1 text-xs font-semibold text-ink">Priors &amp; guardrails</legend>
      <div className="grid gap-5 xl:grid-cols-2">
        <div className="space-y-3">
          <Toggle
            checked={v.policyEffect}
            onChange={(x) => set('policyEffect', x)}
            label="Structured leaf prior (policy effect)"
            description="Tilts each leaf's prior mean by the within-group effect of rate, 60-month term, payment-to-income and log amount on default (offset logistic regression), so unseen arms are not priced as safe as the group."
          />
          {errs['hierarchy.policy_effect'] && <p className="text-2xs text-delta-bad">{errs['hierarchy.policy_effect'].join(' ')}</p>}
          <details className="rounded-md border border-line px-3 py-2 text-xs" open={!!cfgErr || undefined}>
            <summary className="cursor-pointer text-ink-2 select-none">Advanced: regression settings (policy_effect_cfg)</summary>
            <div className="mt-2 space-y-3">
              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-ink-2">Features</span>
                <Chips<PolicyEffectFeature> size="sm" label="Policy-effect features" options={POLICY_EFFECT_FEATURES.map((f) => ({ value: f.key, label: f.label, title: f.key }))} value={v.features} onChange={(x) => set('features', x)} />
              </div>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Field label="Ridge" hint="on standardised β">{(id, d) => <NumberInput id={id} aria-describedby={d} disabled={!v.policyEffect} min={0} step={0.001} value={v.ridge} onValue={(x) => set('ridge', x)} />}</Field>
                <Field label="Max |tilt| (logit)" hint="clip on t(g, a)">{(id, d) => <NumberInput id={id} aria-describedby={d} disabled={!v.policyEffect} min={0.1} step={0.5} value={v.maxShift} onValue={(x) => set('maxShift', x)} />}</Field>
                <Field label="Min. rows" hint="else β = 0">{(id, d) => <NumberInput id={id} aria-describedby={d} disabled={!v.policyEffect} min={1} step={500} value={v.minRows} onValue={(x) => set('minRows', x)} />}</Field>
                <Field label="Min. elapsed rows" hint="to use elapsed loans only">{(id, d) => <NumberInput id={id} aria-describedby={d} disabled={!v.policyEffect} min={0} step={1000} value={v.minElapsedRows} onValue={(x) => set('minElapsedRows', x)} />}</Field>
                <Field label="Max. iterations" hint="Newton–Raphson">{(id, d) => <NumberInput id={id} aria-describedby={d} disabled={!v.policyEffect} min={1} value={v.maxIter} onValue={(x) => set('maxIter', x)} />}</Field>
                <Field label="PTI cap" hint="clip of 12·inst/income">{(id, d) => <NumberInput id={id} aria-describedby={d} disabled={!v.policyEffect} min={0.1} step={0.1} value={v.ptiCap} onValue={(x) => set('ptiCap', x)} />}</Field>
              </div>
              {cfgErr && <p className="text-2xs text-delta-bad">{cfgErr.join(' ')}</p>}
            </div>
          </details>
          <Toggle
            checked={v.termLevel}
            onChange={(x) => set('termLevel', x)}
            label="Recovery by term"
            description="Recovery chain leaf → (group, term) → group → global: a 60-month default has amortised less principal."
          />
          <Toggle
            checked={v.prepayment}
            onChange={(x) => set('prepayment', x)}
            label="Prepayment-aware revenue (empirical ρ)"
            description="Scale expected interest and funding cost by the realised share ρ(grade, term) of fully-paid training loans; off values the full contract (ρ = 1)."
          />
        </div>
        <div className="space-y-2.5">
          <p className="text-xs text-ink-2">Lending guardrails: an arm is admissible for a borrower only if every switched-on rule holds. Decline is always admissible.</p>
          {GUARDRAIL_INFO.map((g) => {
            const err = errs[`arm_space.${g.key}`];
            return (
              <div key={g.key} className="grid grid-cols-[minmax(0,1fr)_7rem] items-start gap-3 border-t border-line pt-2.5">
                <div>
                  <Toggle checked={v.guardOn[g.key]} onChange={(x) => set('guardOn', { ...v.guardOn, [g.key]: x })} label={g.label} description={g.blurb} />
                  {err && <p className="mt-1 text-2xs text-delta-bad">{err.join(' ')}</p>}
                </div>
                <Field label={g.unit === 'pp' ? '± pp' : '%'}>
                  {(id) => (
                    <NumberInput
                      id={id}
                      aria-label={`${g.label} (${g.unit === 'pp' ? 'percentage points' : 'percent'})`}
                      disabled={!v.guardOn[g.key]}
                      min={0.5}
                      step={g.unit === 'pp' ? 0.5 : 1}
                      value={v.guard[g.key]}
                      onValue={(x) => set('guard', { ...v.guard, [g.key]: x })}
                    />
                  )}
                </Field>
              </div>
            );
          })}
        </div>
      </div>
    </fieldset>
  );
}
