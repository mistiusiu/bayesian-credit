import { useState, type FormEvent } from 'react';
import { ApiError, useCreateSimulation } from '@/api/client';
import type { BreakerMode, FeedbackMode, HindsightSample, PolicyName, ReviewAction, Simulation } from '@/api/types';
import { Button } from '@/components/ui/Button';
import { Chips } from '@/components/ui/Chips';
import { Disclosure } from '@/components/ui/Disclosure';
import { Field, FormError, NumberInput, Select, TextInput } from '@/components/ui/Field';
import { Segmented } from '@/components/ui/Segmented';
import { Toggle } from '@/components/ui/Toggle';
import { fmt } from '@/lib/format';
import { BREAKER_MODES, DEFAULT_HINDSIGHT, FEEDBACK_MODES, POLICIES, policyBlurb, policyLabel, PSI_FEATURES, REVIEW_ACTIONS } from '@/lib/domain';

const HINDSIGHT: { value: HindsightSample; label: string; title: string }[] = [
  { value: 'elapsed', label: 'Term elapsed', title: 'Resolved loans whose term had elapsed by the snapshot: unbiased (default)' },
  { value: 'resolved', label: 'All resolved', title: 'Every resolved loan: biased towards early defaults' },
];

const PSI_OPTIONS = [...PSI_FEATURES, 'emp_length', 'verification_status'];

type Knowledge = 'warm' | 'cold';
const KNOWLEDGE: { value: Knowledge; label: string; title: string }[] = [
  { value: 'warm', label: 'Warm · trained fit', title: 'Policies are warm-started from the fit’s risk model (an offline-trained model)' },
  { value: 'cold', label: 'Cold · no offline model', title: 'Policies start from a flat prior and learn only from the rewards their own decisions earn' },
];
const WARM_POLICIES: PolicyName[] = ['thompson', 'ucb1', 'epsilon_greedy', 'historical', 'random'];
const COLD_POLICIES: PolicyName[] = ['greedy', 'thompson', 'ucb1', 'epsilon_greedy', 'historical'];

export function SimulationForm({
  fitId,
  datasetId = null,
  excluded = [],
  datasetLoans = null,
  onCreated,
}: {
  fitId: number | null;
  datasetId?: number | null;
  excluded?: string[];
  datasetLoans?: number | null;
  onCreated: (s: Simulation) => void;
}) {
  const [name, setName] = useState('');
  const [knowledge, setKnowledgeState] = useState<Knowledge>('warm');
  const [reuseFit, setReuseFit] = useState(true);
  const [policies, setPolicies] = useState<PolicyName[]>(WARM_POLICIES);
  const cold = knowledge === 'cold';
  // a cold start needs no fit: without one (or by choice) the dataset alone is replayed
  const useFit = !cold || (reuseFit && fitId !== null);
  const ready = useFit ? fitId !== null : datasetId !== null;
  const setKnowledge = (k: Knowledge) => {
    setKnowledgeState(k);
    setPolicies(k === 'cold' ? COLD_POLICIES : WARM_POLICIES);
  };
  const [feedback, setFeedback] = useState<FeedbackMode>('simulated');
  const [start, setStart] = useState('2015-01');
  const [end, setEnd] = useState('2018-12');
  const [allLoans, setAllLoans] = useState(false);
  const [maxLoans, setMaxLoans] = useState<number | null>(50000);
  const [seed, setSeed] = useState<number | null>(42);
  const [interimMonths, setInterimMonths] = useState<number | null>(12);
  const [interimWeight, setInterimWeight] = useState<number | null>(0.5);
  const [featuresState, setFeatures] = useState<string[]>(PSI_FEATURES);
  // PSI can only monitor variables the dataset kept
  const psiOptions = PSI_OPTIONS.filter((f) => !excluded.includes(f));
  const features = featuresState.filter((f) => psiOptions.includes(f));
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
    if (!ready) return;
    create.mutate(
      {
        ...(useFit ? { fit: fitId } : { dataset: datasetId }),
        cold_start: cold,
        name: name.trim() || `${cold ? 'Cold start · ' : ''}${policies.map((p) => policyLabel(p)).join(' vs ')} · ${feedback}`,
        policies,
        feedback_mode: feedback,
        start,
        end,
        max_loans: allLoans ? null : maxLoans,
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
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink-2">Starting knowledge</span>
          <Segmented<Knowledge> label="Starting knowledge" value={knowledge} onChange={setKnowledge} options={KNOWLEDGE} />
          <p className="max-w-3xl text-2xs text-muted">
            {cold
              ? 'Cold start: every policy begins from the flat prior N(0, 1000²) with no risk-model warm start, the position of a lender entering a market it has no history in. Exploration now matters: Greedy ties at μ = 0 on every arm and never leaves Decline, while Thompson, UCB1 and ε-greedy learn to lend from their own outcomes. The replay world, guardrails, Sentinel and hindsight yardstick are the same as in a warm run, so the two are directly comparable.'
              : 'Warm start: policies begin from the fit’s risk model (μ, σ of every arm from the Two-Stage Beta-Beta posterior). Pure exploitation does well here because the model already knows the borrowers.'}
          </p>
          {errs.cold_start && <p className="text-2xs text-delta-bad">{errs.cold_start.join(' ')}</p>}
          {errs.dataset && <p className="text-2xs text-delta-bad">{errs.dataset.join(' ')}</p>}
        </div>
        {cold && (
          <Toggle
            checked={reuseFit && fitId !== null}
            onChange={setReuseFit}
            disabled={fitId === null}
            label={fitId !== null ? `Reuse fit #${fitId}’s replay world` : 'No fit needed'}
            description={
              fitId !== null
                ? 'On: the world (arm grid, groups, MCTM, rate card) comes from the selected fit, matching its warm runs exactly. Off: the run needs no fit; the world is fitted on the loans issued before the first epoch.'
                : 'This dataset has no ready fit, and a cold start does not need one: the replay world is fitted on the loans issued before the first epoch.'
            }
          />
        )}
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
          <Field label="Max loans" hint={allLoans ? 'every loan in the window' : 'pilot: sampled with the seed'} error={errs.max_loans}>
            {(id, d) => <NumberInput id={id} aria-describedby={d} min={100} step={1000} disabled={allLoans} value={allLoans ? null : maxLoans} onValue={setMaxLoans} />}
          </Field>
        </div>
        <Toggle
          checked={allLoans}
          onChange={setAllLoans}
          label="Every loan in the window"
          description={`Stream the whole dataset between the first and last epoch instead of a seeded pilot sample${datasetLoans ? ` (the dataset holds ${fmt.int(datasetLoans)} loans in total)` : ''}. Full runs take proportionally longer; they checkpoint and resume like any other.`}
        />
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
            <Chips size="sm" label="Monitored features" options={psiOptions.map((f) => ({ value: f, label: f }))} value={features} onChange={setFeatures} />
            {errs.psi && <p className="text-2xs text-delta-bad">{errs.psi.join(' ')}</p>}
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
          <Button type="submit" variant="primary" icon="play" loading={create.isPending} disabled={!ready || !policies.length || !features.length}>
            {cold ? 'Start cold-start simulation' : 'Start simulation'}
          </Button>
          <span className="text-xs text-muted">
            {!ready
              ? cold
                ? 'Select a ready dataset first.'
                : 'Select a ready fit first, or switch to a cold start (no fit needed).'
              : useFit
                ? `${cold ? 'cold start in ' : 'with '}fit #${fitId}${cold ? '’s world' : ''}`
                : `cold start on dataset #${datasetId}, no fit`}
          </span>
        </div>
      </form>
    </Disclosure>
  );
}
