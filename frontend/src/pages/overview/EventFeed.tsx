import type { UseQueryResult } from '@tanstack/react-query';
import type { EventLevel, EventPage } from '@/api/types';
import { StatusGlyph } from '@/components/ui/Badge';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import type { Tone } from '@/lib/palette';

const LEVEL: Record<EventLevel, Tone> = { info: 'neutral', warning: 'warning', critical: 'critical' };

export function EventFeed({ query, max = 14 }: { query: UseQueryResult<EventPage>; max?: number }) {
  return (
    <QueryState query={query} isEmpty={(d) => d.results.length === 0} height={120}>
      {(d) => (
        <ol className="max-h-[26rem] divide-y divide-line overflow-y-auto" aria-live="polite">
          {d.results.slice(0, max).map((e) => (
            <li key={e.seq} className="flex gap-2.5 px-4 py-2">
              <span className="mt-0.5">
                <StatusGlyph tone={LEVEL[e.level]} />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-2xs font-semibold text-ink-2">{e.type}</span>
                  <span className="num shrink-0 text-2xs text-muted">{e.epoch ? fmt.epochShort(e.epoch) : fmt.dateTime(e.ts)}</span>
                </div>
                <p className="text-xs leading-snug text-ink">{e.message}</p>
              </div>
            </li>
          ))}
        </ol>
      )}
    </QueryState>
  );
}
