import { Icon } from '@/components/ui/Icon';

/** Reading guide for counterfactual uplift (docs/MATH.md §2.6–2.8). */
export function HowToReadUplift({ className }: { className?: string }) {
  return (
    <aside aria-labelledby="how-to-read-uplift" className={`rounded-lg border border-line bg-surface-2 px-4 py-3 text-xs text-ink-2 ${className ?? ''}`}>
      <h3 id="how-to-read-uplift" className="mb-1.5 flex items-center gap-1.5 text-[13px] font-semibold text-ink">
        <Icon name="info" size={14} className="text-muted" />
        How to read the uplift
      </h3>
      <ul className="list-disc space-y-1 pl-4 leading-relaxed">
        <li>
          <strong className="font-medium text-ink">A model quantity, not an observed result.</strong> Uplift is the fitted risk model&apos;s expected profit of the arms the shadow engine would offer minus that of the arms Lending Club issued,
          for the same borrowers. Borrowers are assumed to accept any admissible offer: there is no take-up or adverse-selection term beyond the policy-effect prior (higher rates, longer terms and heavier payments raise the
          predicted default rate).
        </li>
        <li>
          <strong className="font-medium text-ink">Where it comes from.</strong> The decomposition re-optimises with the historical amount and term fixed (repricing only), then with the term fixed (repricing and resizing), then
          with nothing fixed (term choice), all within the guardrails.
        </li>
        <li>
          <strong className="font-medium text-ink">Risk appetite.</strong> The λ sweep scores arms by mean − λ·std: a larger λ trades expected profit for lower variance, shrinking amounts, shortening terms and declining more.
        </li>
      </ul>
    </aside>
  );
}
