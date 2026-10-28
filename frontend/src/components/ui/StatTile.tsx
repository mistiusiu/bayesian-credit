import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';
import type { Tone } from '@/lib/palette';
import { StatusGlyph } from './Badge';

export interface Delta {
  text: string;
  /** true = good direction, false = bad, null = neutral */
  good: boolean | null;
}

/** Stat tile contract: label · value (proportional figures) · delta or hint · optional status glyph. */
export function StatTile({
  label,
  value,
  hint,
  delta,
  status,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  delta?: Delta | null;
  status?: { tone: Tone; label: string } | null;
  className?: string;
}) {
  return (
    <div className={cn('min-w-0 rounded-lg border border-line bg-surface px-3.5 py-3', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs text-muted">{label}</span>
        {status && (
          <span className="inline-flex items-center gap-1 text-2xs font-medium text-ink-2">
            <StatusGlyph tone={status.tone} />
            {status.label}
          </span>
        )}
      </div>
      <div className="mt-1 truncate text-[22px] leading-7 font-semibold text-ink">{value}</div>
      {(delta || hint) && (
        <div className="mt-0.5 flex flex-wrap items-baseline gap-x-2 text-xs">
          {delta && (
            <span className={cn('num font-medium', delta.good === true ? 'text-delta-good' : delta.good === false ? 'text-delta-bad' : 'text-ink-2')}>
              {delta.text}
            </span>
          )}
          {hint && <span className="text-muted">{hint}</span>}
        </div>
      )}
    </div>
  );
}
