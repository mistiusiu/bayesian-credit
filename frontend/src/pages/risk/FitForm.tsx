import { useState, type FormEvent } from 'react';
import { ApiError, useCreateFit } from '@/api/client';
import type { Fit, SegmentationDimension, TransitionStructure } from '@/api/types';
import { Button } from '@/components/ui/Button';
import { Chips } from '@/components/ui/Chips';
import { Disclosure } from '@/components/ui/Disclosure';
import { Field, FormError, NumberInput, TextInput } from '@/components/ui/Field';
import { Segmented } from '@/components/ui/Segmented';
import { Toggle } from '@/components/ui/Toggle';
import { DEFAULT_FIT_CONFIG, SEGMENTATION_DIMENSIONS } from '@/lib/domain';
import { ArmGridEditor, type ArmGridState } from './ArmGridEditor';
import { parseAmounts, parseRates } from './armGrid';
import { defaultPriors, priorsPayload } from './priors';
import { PriorsGuardrails } from './PriorsGuardrails';

const D = DEFAULT_FIT_CONFIG;

export function FitForm({ datasetId, onCreated }: { datasetId: number | null; onCreated: (f: Fit) => void }) {
  const [name, setName] = useState('');
  const [dims, setDims] = useState<SegmentationDimension[]>(D.segmentation.dimensions);
  const [arms, setArms] = useState<ArmGridState>({
    amounts: D.arm_space.amounts.map((a) => a / 1000).join(', '),
    rates: D.arm_space.rates.map((r) => Math.round(r * 100)).join(', '),
    terms: D.arm_space.terms,
    includeNull: true,
  });
  const [priors, setPriors] = useState(defaultPriors);
  const [kg, setKg] = useState<number | null>(20);
  const [kl, setKl] = useState<number | null>(10);
  const [auto, setAuto] = useState(false);
  const [a0, setA0] = useState<number | null>(1);
  const [b0, setB0] = useState<number | null>(1);
  const [minObs, setMinObs] = useState<number | null>(5);
  const [structure, setStructure] = useState<TransitionStructure>('proposal');
  const [prior, setPrior] = useState<number | null>(1);
  const [lag, setLag] = useState<number | null>(4);
  const [funding, setFunding] = useState<number | null>(4);
  const [lambda, setLambda] = useState<number | null>(0);
  const [lendable, setLendable] = useState<number | null>(0.6);
  const [trainEnd, setTrainEnd] = useState('2015-01');
  const create = useCreateFit();
  const errs = create.error instanceof ApiError ? create.error.fieldErrors : {};

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (datasetId === null) return;
    const pr = priorsPayload(priors);
    create.mutate(
      {
        dataset: datasetId,
        name: name.trim() || `${dims.join(' × ')} · ${structure}`,
        segmentation: { dimensions: dims },
        arm_space: {
          amounts: parseAmounts(arms.amounts).values,
          rates: parseRates(arms.rates).values,
          terms: arms.terms,
          include_null: arms.includeNull,
          ...pr.guardrails,
        },
        hierarchy: { kappa_group: auto ? 'auto' : (kg ?? 20), kappa_leaf: auto ? 'auto' : (kl ?? 10), alpha0: a0 ?? 1, beta0: b0 ?? 1, ...pr.hierarchy },
        recovery: { min_obs: minObs ?? 5, term_level: pr.term_level },
        mctm: { structure, prior_strength: prior ?? 1, chargeoff_lag_months: lag ?? 4 },
        reward: { funding_rate: (funding ?? 4) / 100, risk_aversion: lambda ?? 0, lendable_threshold: lendable ?? 0.6, prepayment_adjustment: pr.prepayment_adjustment },
        train_end: trainEnd,
      },
      { onSuccess: onCreated },
    );
  };

  return (
    <Disclosure title="New fit" subtitle="Two-Stage Beta-Beta + MCTM on the selected dataset (background job)">
      <form onSubmit={submit} className="space-y-4" noValidate>
        <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_10rem]">
          <Field label="Name" error={errs.name}>
            {(id, d) => <TextInput id={id} aria-describedby={d} value={name} placeholder="auto" onChange={(e) => setName(e.target.value)} />}
          </Field>
          <Field label="Train end (exclusive)" hint="fit on issue_d < this month" error={errs.train_end}>
            {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={trainEnd} onChange={(e) => setTrainEnd(e.target.value)} />}
          </Field>
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink-2">Segmentation dimensions (borrower group g)</span>
          <Chips<SegmentationDimension> label="Segmentation dimensions" options={SEGMENTATION_DIMENSIONS.map((d) => ({ value: d.key, label: d.label, title: `derived from ${d.source}` }))} value={dims} onChange={setDims} />
          {errs.segmentation && <p className="text-2xs text-delta-bad">{errs.segmentation.join(' ')}</p>}
        </div>
        <ArmGridEditor value={arms} onChange={setArms} error={errs.arm_space} />
        <PriorsGuardrails value={priors} onChange={setPriors} errs={errs} />
        <div className="grid gap-4 xl:grid-cols-3">
          <fieldset className="space-y-3 rounded-md border border-line p-3">
            <legend className="px-1 text-xs font-semibold text-ink">Hierarchy priors (stage 1)</legend>
            <Toggle checked={auto} onChange={setAuto} label="κ by method of moments (auto)" description="κ = m(1−m)/Var(child rates) − 1, clipped to [1, 1000]" />
            <div className="grid grid-cols-2 gap-3">
              <Field label="κ group" error={errs['hierarchy.kappa_group']}>
                {(id) => <NumberInput id={id} disabled={auto} min={0.1} value={kg} onValue={setKg} />}
              </Field>
              <Field label="κ leaf" error={errs['hierarchy.kappa_leaf']}>
                {(id) => <NumberInput id={id} disabled={auto} min={0.1} value={kl} onValue={setKl} />}
              </Field>
              <Field label="α₀ (global prior)">{(id) => <NumberInput id={id} min={0.01} step={0.5} value={a0} onValue={setA0} />}</Field>
              <Field label="β₀ (global prior)">{(id) => <NumberInput id={id} min={0.01} step={0.5} value={b0} onValue={setB0} />}</Field>
            </div>
            <Field label="Recovery min. observations" hint="below this a leaf falls back to its group, then global">
              {(id, d) => <NumberInput id={id} aria-describedby={d} min={1} value={minObs} onValue={setMinObs} />}
            </Field>
          </fieldset>
          <fieldset className="space-y-3 rounded-md border border-line p-3">
            <legend className="px-1 text-xs font-semibold text-ink">Credit-state chain (MCTM)</legend>
            <Segmented<TransitionStructure>
              label="Transition structure"
              value={structure}
              onChange={setStructure}
              options={[
                { value: 'proposal', label: 'Proposal', title: 'C leaves only via D (structural zeros C→R, C→L)' },
                { value: 'extended', label: 'Extended', title: 'also allows C→R (prepayment / payoff)' },
              ]}
            />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Dirichlet prior strength" hint="0 = MLE">{(id, d) => <NumberInput id={id} aria-describedby={d} min={0} step={0.5} value={prior} onValue={setPrior} />}</Field>
              <Field label="Charge-off lag (months)" hint="LC: 120+ DPD">{(id, d) => <NumberInput id={id} aria-describedby={d} min={1} value={lag} onValue={setLag} />}</Field>
            </div>
          </fieldset>
          <fieldset className="space-y-3 rounded-md border border-line p-3">
            <legend className="px-1 text-xs font-semibold text-ink">Reward (R = revenue − E[loss] − funding)</legend>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Funding rate (% / year)" error={errs['reward.funding_rate']}>{(id) => <NumberInput id={id} min={0} max={50} step={0.25} value={funding} onValue={setFunding} />}</Field>
              <Field label="Risk aversion λ" hint="score = mean − λ·std">{(id, d) => <NumberInput id={id} aria-describedby={d} min={0} step={0.1} value={lambda} onValue={setLambda} />}</Field>
              <Field label="Lendable threshold" hint="P(reward > 0) needed">{(id, d) => <NumberInput id={id} aria-describedby={d} min={0} max={1} step={0.05} value={lendable} onValue={setLendable} />}</Field>
            </div>
          </fieldset>
        </div>
        <FormError message={create.error && !Object.keys(errs).length ? create.error.message : null} />
        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" icon="plus" loading={create.isPending} disabled={datasetId === null || !dims.length || !arms.terms.length}>
            Fit models
          </Button>
          <span className="text-xs text-muted">{datasetId === null ? 'Select a dataset first.' : `on dataset #${datasetId}`}</span>
        </div>
      </form>
    </Disclosure>
  );
}
