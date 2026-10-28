import type { Guardrails, ReferenceRates } from '@/api/types';
import { GUARDRAIL_INFO, guardrailValue } from '@/lib/domain';
import { fmt } from '@/lib/format';

/** The fit's lending guardrails and the grade reference rates behind the rate corridor (docs/MATH.md §2.7). */
export function GuardrailStrip({ guardrails, reference, source }: { guardrails: Guardrails | null; reference: ReferenceRates | null | undefined; source: 'metrics' | 'config' }) {
  return (
    <section aria-label="Lending guardrails and reference rates" className="rounded-lg border border-line bg-surface px-4 py-2.5 text-xs">
      <div className="flex flex-wrap items-start gap-x-8 gap-y-2">
        <div className="min-w-0">
          <h2 className="text-2xs font-semibold tracking-wide text-muted uppercase">Guardrails{source === 'config' ? ' (from the fit configuration)' : ''}</h2>
          <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
            {GUARDRAIL_INFO.map((g) => (
              <li key={g.key} title={g.blurb}>
                <span className="text-muted">{g.label}</span> <strong className="num font-semibold text-ink">{guardrailValue(guardrails?.[g.key], g.unit)}</strong>
              </li>
            ))}
          </ul>
        </div>
        <div className="min-w-0">
          <h2 className="text-2xs font-semibold tracking-wide text-muted uppercase">Reference rates · training mean by grade</h2>
          {reference ? (
            <ul className="num mt-1 flex flex-wrap gap-x-3 gap-y-1">
              {Object.entries(reference.by_grade)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([g, r]) => (
                  <li key={g}>
                    <span className="text-muted">{g}</span> <span className="text-ink">{fmt.pct(r, 1)}</span>
                  </li>
                ))}
              <li>
                <span className="text-muted">all</span> <span className="text-ink">{fmt.pct(reference.global, 1)}</span>
              </li>
            </ul>
          ) : (
            <p className="mt-1 text-muted">Not reported: the rate corridor is inactive for this fit.</p>
          )}
        </div>
      </div>
      <details className="mt-1.5 text-2xs text-muted">
        <summary className="cursor-pointer select-none hover:text-ink">What the guardrails mean</summary>
        <ul className="mt-1 space-y-0.5 pl-3">
          {GUARDRAIL_INFO.map((g) => (
            <li key={g.key}>
              <span className="text-ink-2">{g.label}:</span> {g.blurb}
            </li>
          ))}
          <li>An arm is admissible when every guardrail whose input is known holds; decline is always admissible, and a fallback arm (closest to the reference rate, smallest instalment) is kept when no lending arm passes.</li>
        </ul>
      </details>
    </section>
  );
}
