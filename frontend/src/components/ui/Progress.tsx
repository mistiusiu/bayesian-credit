import { cn } from '@/lib/cn';
import { fmt } from '@/lib/format';

/** Determinate progress (fill + track from the same blue ramp). */
export function ProgressBar({ value, label, className, showValue = true }: { value: number; label: string; className?: string; showValue?: boolean }) {
  const v = Math.min(1, Math.max(0, value));
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)} className="h-1.5 min-w-12 flex-1 overflow-hidden rounded-full bg-accent-wash">
        <div className="h-full rounded-full bg-accent-strong transition-[width] duration-500" style={{ width: `${v * 100}%` }} />
      </div>
      {showValue && <span className="num w-9 text-right text-2xs text-muted">{fmt.pct(v, 0)}</span>}
    </div>
  );
}

/** Meter for a ratio (e.g. shrinkage weight n/(n+κ)): same-ramp track, value label beside it. */
export function Meter({ value, label, width = 64 }: { value: number; label: string; width?: number }) {
  const v = Math.min(1, Math.max(0, value));
  return (
    <span className="inline-flex items-center gap-1.5" title={label}>
      <span role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={1} aria-valuenow={Number(v.toFixed(3))} className="inline-block h-1.5 overflow-hidden rounded-full bg-accent-wash" style={{ width }}>
        <span className="block h-full rounded-full bg-accent-strong" style={{ width: `${v * 100}%` }} />
      </span>
      <span className="num text-2xs text-ink-2">{fmt.pct(v, 0)}</span>
    </span>
  );
}
