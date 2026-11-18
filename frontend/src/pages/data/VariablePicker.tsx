import { useVariableCatalogue } from '@/api/client';
import type { Stage, VariableOption } from '@/api/types';
import { StageBadge } from '@/components/ui/Badge';
import { Icon } from '@/components/ui/Icon';
import { QueryState } from '@/components/ui/States';
import { cn } from '@/lib/cn';
import { STAGE_BLURB } from '@/lib/domain';
import { matchingPreset, VARIABLE_PRESETS } from '@/lib/variables';

const SELECTABLE_STAGES: Stage[] = ['pre_decision', 'interim'];

function VariableRowCheck({ v, kept, onToggle }: { v: VariableOption; kept: boolean; onToggle: () => void }) {
  return (
    <label className={cn('flex cursor-pointer items-start gap-2 rounded px-1.5 py-1 hover:bg-surface-2', !kept && 'opacity-60')}>
      <input type="checkbox" checked={kept} onChange={onToggle} className="mt-0.5 size-3.5 shrink-0 accent-[var(--accent)]" />
      <span className="min-w-0">
        <code className={cn('text-xs', kept ? 'text-ink' : 'text-muted line-through')}>{v.name}</code>
        <span className="block truncate text-2xs text-muted" title={v.uses.join(' · ') || undefined}>
          {v.uses.length ? v.uses.join(' · ') : v.stage === 'interim' ? 'not read by any model component' : 'context only'}
        </span>
      </span>
    </label>
  );
}

function Picker({ catalogue, excluded, onChange }: { catalogue: VariableOption[]; excluded: string[]; onChange: (v: string[]) => void }) {
  const selectable = catalogue.filter((v) => v.selectable);
  const names = selectable.map((v) => v.name);
  const required = catalogue.filter((v) => !v.selectable);
  // keep the list in Table 3 order (the order the API echoes back)
  const set = (next: Set<string>) => onChange(names.filter((n) => next.has(n)));
  const toggle = (name: string) => {
    const next = new Set(excluded);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    set(next);
  };
  const setStage = (stage: Stage, keep: boolean) => {
    const next = new Set(excluded);
    for (const v of selectable) {
      if (v.stage !== stage) continue;
      if (keep) next.delete(v.name);
      else next.add(v.name);
    }
    set(next);
  };
  const preset = matchingPreset(excluded, names);
  const feeding = selectable.filter((v) => excluded.includes(v.name) && v.uses.length);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Variable presets">
        <span className="mr-1 text-xs font-medium text-ink-2">Preset</span>
        {VARIABLE_PRESETS.map((p) => {
          const on = preset?.key === p.key;
          return (
            <button
              key={p.key}
              type="button"
              aria-pressed={on}
              title={p.blurb}
              onClick={() => set(new Set(p.exclude))}
              className={cn(
                'h-7 rounded-full border px-2.5 text-xs font-medium whitespace-nowrap transition-colors',
                on ? 'border-accent-strong bg-accent-wash text-ink' : 'border-line-strong text-muted hover:text-ink',
              )}
            >
              {p.label}
            </button>
          );
        })}
        {!preset && <span className="text-2xs text-muted">custom selection</span>}
      </div>
      <p className="text-2xs text-muted">{preset?.blurb ?? 'Unchecked variables are dropped from the stored dataset, so no fit, simulation or shadow evaluation built on it can see them.'}</p>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        {SELECTABLE_STAGES.map((stage) => {
          const rows = selectable.filter((v) => v.stage === stage);
          const kept = rows.filter((v) => !excluded.includes(v.name)).length;
          return (
            <fieldset key={stage} className="rounded-md border border-line p-2.5">
              <legend className="flex items-center gap-2 px-1">
                <StageBadge stage={stage} />
                <span className="num text-xs text-ink-2">
                  {kept} of {rows.length} kept
                </span>
              </legend>
              <div className="mb-1.5 flex items-center justify-between gap-2 px-1.5">
                <p className="text-2xs text-muted">{STAGE_BLURB[stage]}</p>
                <span className="flex shrink-0 gap-2 text-2xs">
                  <button type="button" className="text-ink-2 underline-offset-2 hover:underline" onClick={() => setStage(stage, true)}>
                    all
                  </button>
                  <button type="button" className="text-ink-2 underline-offset-2 hover:underline" onClick={() => setStage(stage, false)}>
                    none
                  </button>
                </span>
              </div>
              <div className={cn('grid gap-x-2', stage === 'pre_decision' ? 'sm:grid-cols-2 2xl:grid-cols-3' : '')}>
                {rows.map((v) => (
                  <VariableRowCheck key={v.name} v={v} kept={!excluded.includes(v.name)} onToggle={() => toggle(v.name)} />
                ))}
              </div>
            </fieldset>
          );
        })}
      </div>

      {feeding.length > 0 && (
        <div className="flex items-start gap-2 rounded-md border border-line bg-surface-2/60 px-3 py-2 text-xs text-ink-2">
          <Icon name="info" size={14} className="mt-0.5 shrink-0 text-muted" />
          <div>
            <p>Dropped variables the default model reads; fits and simulations on this dataset adapt:</p>
            <ul className="mt-1 space-y-0.5 text-2xs">
              {feeding.map((v) => (
                <li key={v.name}>
                  <code>{v.name}</code> — {v.uses.join(' · ')}
                </li>
              ))}
            </ul>
            <p className="mt-1 text-2xs text-muted">
              Segmentation dimensions and PSI features built on them are dropped from the defaults; without annual_inc the affordability guardrails and the PTI policy effect switch off.
            </p>
          </div>
        </div>
      )}

      <p className="flex items-start gap-2 text-2xs text-muted">
        <Icon name="lock" size={12} className="mt-0.5 shrink-0" />
        <span>
          Always kept ({required.length}):{' '}
          {required.map((v, i) => (
            <span key={v.name} title={v.required_reason ?? undefined}>
              <code>{v.name}</code>
              {i < required.length - 1 ? ', ' : ''}
            </span>
          ))}{' '}
          — the arm, the time index and the outcome labels the rewards are computed from.
        </span>
      </p>
    </div>
  );
}

/** Choose the Table 3 variables a new dataset keeps; ``value`` is the exclusion list. */
export function VariablePicker({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const q = useVariableCatalogue();
  return <QueryState query={q} loadingLabel="Loading Table 3 catalogue">{(catalogue) => <Picker catalogue={catalogue} excluded={value} onChange={onChange} />}</QueryState>;
}
