import type { BreakerState, SentinelLevel } from '@/api/types';
import { StatusGlyph } from '@/components/ui/Badge';
import { cn } from '@/lib/cn';
import { fmt } from '@/lib/format';
import { BREAKER_LABEL, BREAKER_TONE, LEVEL_TONE, TONE_COLOR } from '@/lib/palette';

/**
 * Breaker state per epoch as a strip of status-coloured cells (closed / half-open / open),
 * with the PSI level as a thin second row. Status colours always come with the legend glyphs.
 */
export function BreakerStrip({
  epochs,
  states,
  levels,
  selected,
  onSelect,
}: {
  epochs: string[];
  states: BreakerState[];
  levels: SentinelLevel[];
  selected?: string | null;
  onSelect?: (epoch: string) => void;
}) {
  return (
    <div>
      <div className="flex gap-[2px]" role="list" aria-label="Breaker state by epoch">
        {epochs.map((e, i) => {
          const s = states[i];
          const lvl = levels[i];
          return (
            <button
              key={e}
              type="button"
              role="listitem"
              onClick={onSelect ? () => onSelect(e) : undefined}
              aria-label={`${fmt.epoch(e)}: breaker ${BREAKER_LABEL[s]}, PSI ${lvl}`}
              title={`${fmt.epoch(e)} · breaker ${BREAKER_LABEL[s].toLowerCase()} · PSI ${lvl}`}
              className={cn('flex min-w-0 flex-1 flex-col gap-[2px] rounded-[2px] outline-offset-1', selected === e && 'ring-2 ring-[var(--focus)]')}
            >
              <span className="block h-5 w-full rounded-t-[3px]" style={{ background: s === 'closed' ? 'var(--surface-3)' : TONE_COLOR[BREAKER_TONE[s]] }} />
              <span className="block h-1 w-full rounded-b-[2px]" style={{ background: lvl === 'normal' ? 'var(--surface-2)' : TONE_COLOR[LEVEL_TONE[lvl]] }} />
            </button>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-2xs text-axis">
        <span className="num">{fmt.epochShort(epochs[0])}</span>
        <span className="num">{fmt.epochShort(epochs[epochs.length - 1])}</span>
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <li className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-[2px] bg-surface-3" /> Closed
        </li>
        <li className="inline-flex items-center gap-1.5">
          <StatusGlyph tone="warning" /> Half-open (awaiting normal readings)
        </li>
        <li className="inline-flex items-center gap-1.5">
          <StatusGlyph tone="critical" /> Open (policies frozen)
        </li>
        <li className="inline-flex items-center gap-1.5 text-muted">thin row: PSI level (watch amber, breaker red)</li>
      </ul>
    </div>
  );
}
