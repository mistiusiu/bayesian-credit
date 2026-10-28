import { useState, type FormEvent } from 'react';
import { ApiError, useCreateSimulation } from '@/api/client';
import type { BreakerMode, FeedbackMode, HindsightSample, PolicyName, ReviewAction, Simulation } from '@/api/types';
import { Button } from '@/components/ui/Button';
import { Chips } from '@/components/ui/Chips';
import { Disclosure } from '@/components/ui/Disclosure';
import { Field, FormError, NumberInput, Select, TextInput } from '@/components/ui/Field';
import { Segmented } from '@/components/ui/Segmented';
import { BREAKER_MODES, DEFAULT_HINDSIGHT, FEEDBACK_MODES, POLICIES, policyBlurb, policyLabel, PSI_FEATURES, REVIEW_ACTIONS } from '@/lib/domain';

const HINDSIGHT: { value: HindsightSample; label: string; title: string }[] = [
  { value: 'elapsed', label: 'Term elapsed', title: 'Resolved loans whose term had elapsed by the snapshot: unbiased (default)' },
  { value: 'resolved', label: 'All resolved', title: 'Every resolved loan: biased towards early defaults' },
];

export function SimulationForm({ fitId, onCreated }: { fitId: number | null; onCreated: (s: Simulation) => void }) {
  const [name, setName] = useState('');
  const [policies, setPolicies] = useState<PolicyName[]>(['thompson', 'ucb1', 'epsilon_greedy', 'historical', 'random']);
  const [feedback, setFeedback] = useState<FeedbackMode>('simulated');
  const [start, setStart] = useState('2015-01');
  const [end, setEnd] = useState('2018-12');
  const [maxLoans, setMaxLoans] = useState<number | null>(50000);
  const [seed, setSeed] = useState<number | null>(42);
  const [interimMonths, setInterimMonths] = useState<number | null>(12);
  const [interimWeight, setInterimWeight] = useState<number | null>(0.5);
  const [features, setFeatures] = useState<string[]>(PSI_FEATURES);
  const [windowMonths, setWindowMonths] = useState<number | null>(3);
  const [watch, setWatch] = useState<number | null>(0.1);
  const [breaker, setBreaker] = useState<number | null>(0.25);
  const [mode, setMode] = useState<BreakerMode>('pause');
  const [autoAction, setAutoAction] = useState<ReviewAction>('resegment');
  const [every, setEvery] = useState<number | null>(6);
  const [hindsight, setHindsight] = useState<HindsightSample>(DEFAULT_HINDSIGHT);
  const create = useCreateSimulation();
  const errs = create.error instanceof ApiError ? create.error.fieldErrors : {};

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (fitId === null) return;
    create.mutate(
      {
        fit: fitId,
        name: name.trim() || `${policies.map((p) => policyLabel(p)).join(' vs ')} · ${feedback}`,
        policies,
        feedback_mode: feedback,
        start,
        end,
        max_loans: maxLoans,
        seed: seed ?? 42,
        interim_months: interimMonths ?? 12,
        interim_weight: interimWeight ?? 0.5,
        psi: { features, window_months: windowMonths ?? 3, thresholds: { watch: watch ?? 0.1, breaker: breaker ?? 0.25 }, auto_action: autoAction },
        breaker_mode: mode,
        checkpoint_every: every ?? 6,
        hindsight_sample: hindsight,
      },
      { onSuccess: onCreated },
    );
  };

  return (
    <Disclosure title="New simulation" subtitle="Replay 2015–2018 borrowers through competing policies (background job, checkpointed)">
      <form onSubmit={submit} className="space-y-4" noValidate>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <Field label="Name" error={errs.name}>
            {(id) => <TextInput id={id} value={name} placeholder="auto" onChange={(e) => setName(e.target.value)} />}
          </Field>
          <Field label="First epoch" hint="YYYY-MM" error={errs.start}>
            {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={start} onChange={(e) => setStart(e.target.value)} />}
          </Field>
          <Field label="Last epoch" hint="YYYY-MM, inclusive" error={errs.end}>
            {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={end} onChange={(e) => setEnd(e.target.value)} />}
          </Field>
          <Field label="Max loans" hint="sampled with the seed; empty = all" error={errs.max_loans}>
            {(id, d) => <NumberInput id={id} aria-describedby={d} min={100} step={1000} value={maxLoans} onValue={setMaxLoans} />}
          </Field>
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink-2">Policies</span>
          <Chips<PolicyName> label="Policies" options={POLICIES.map((p) => ({ value: p, label: policyLabel(p), title: policyBlurb(p) }))} value={policies} onChange={setPolicies} />
          {errs.policies && <p className="text-2xs text-delta-bad">{errs.policies.join(' ')}</p>}
        </div>
        <div className="grid gap-4 xl:grid-cols-3">
          <fieldset className="space-y-3 rounded-md border border-line p-3">
            <legend className="px-1 text-xs font-semibold text-ink">Feedback</legend>
            <Segmented<FeedbackMode> label="Feedback mode" value={feedback} onChange={setFeedback} options={FEEDBACK_MODES.map((f) => ({ value: f.key, label: f.label, title: f.blurb }))} />
            <p className="text-2xs text-muted">{FEEDBACK_MODES.find((f) => f.key === feedback)?.blurb}</p>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Interim months">{(id) => <NumberInput id={id} min={0} max={60} value={interimMonths} onValue={setInterimMonths} />}</Field>
              <Field label="Interim weight" error={errs.interim_weight}>{(id) => <NumberInput id={id} min={0} max={1} step={0.1} value={interimWeight} onValue={setInterimWeight} />}</Field>
              <Field label="Seed">{(id) => <NumberInput id={id} value={seed} onValue={setSeed} />}</Field>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-ink-2">Hindsight yardstick (oracle and regret) fitted on</span>
              <Segmented<HindsightSample> label="Hindsight sample" value={hindsight} onChange={setHindsight} options={HINDSIGHT} />
              <p className="text-2xs text-muted">{HINDSIGHT.find((h) => h.value === hindsight)?.title}</p>
              {errs.hindsight_sample && <p className="text-2xs text-delta-bad">{errs.hindsight_sample.join(' ')}</p>}
            </div>
          </fieldset>
          <fieldset className="space-y-3 rounded-md border border-line p-3">
            <legend className="px-1 text-xs font-semibold text-ink">PSI Sentinel</legend>
            <Chips size="sm" label="Monitored features" options={[...PSI_FEATURES, 'emp_length', 'verification_status'].map((f) => ({ value: f, label: f }))} value={features} onChange={setFeatures} />
            <div className="grid grid-cols-3 gap-3">
              <Field label="Window (months)" error={errs['psi.window_months']}>{(id) => <NumberInput id={id} min={1} value={windowMonths} onValue={setWindowMonths} />}</Field>
              <Field label="Watch ≥" error={errs['psi.thresholds']}>{(id) => <NumberInput id={id} min={0} step={0.01} value={watch} onValue={setWatch} />}</Field>
              <Field label="Breaker >">{(id) => <NumberInput id={id} min={0} step={0.01} value={breaker} onValue={setBreaker} />}</Field>
            </div>
          </fieldset>
          <fieldset className="space-y-3 rounded-md border border-line p-3">
            <legend className="px-1 text-xs font-semibold text-ink">Breaker & checkpoints</legend>
            <Segmented<BreakerMode> label="Breaker mode" value={mode} onChange={setMode} options={BREAKER_MODES.map((b) => ({ value: b.key, label: b.label, title: b.blurb }))} />
            <p className="text-2xs text-muted">{BREAKER_MODES.find((b) => b.key === mode)?.blurb}</p>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Auto action" hint="breaker_mode = auto-acknowledge">
                {(id, d) => <Select id={id} aria-describedby={d} disabled={mode !== 'auto_ack'} value={autoAction} onValue={setAutoAction} options={REVIEW_ACTIONS.map((a) => ({ value: a.key, label: a.label }))} />}
              </Field>
              <Field label="Checkpoint every" hint="epochs" error={errs.checkpoint_every}>
                {(id, d) => <NumberInput id={id} aria-describedby={d} min={1} value={every} onValue={setEvery} />}
              </Field>
            </div>
          </fieldset>
        </div>
        <FormError message={create.error && !Object.keys(errs).length ? create.error.message : null} />
        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" icon="play" loading={create.isPending} disabled={fitId === null || !policies.length}>
            Start simulation
          </Button>
          <span className="text-xs text-muted">{fitId === null ? 'Select a ready fit first.' : `with fit #${fitId}`}</span>
        </div>
      </form>
    </Disclosure>
  );
}
