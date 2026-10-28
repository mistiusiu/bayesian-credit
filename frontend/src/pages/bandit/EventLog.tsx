import { useState } from 'react';
import { useEvents } from '@/api/client';
import type { EventLevel, Simulation } from '@/api/types';
import { StatusGlyph } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Chips } from '@/components/ui/Chips';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import type { Tone } from '@/lib/palette';

const GROUPS = {
  lifecycle: ['RunStarted', 'RunPaused', 'RunResumed', 'WorkerKilled', 'RunCompleted', 'RunCancelled', 'RunFailed', 'CheckpointWritten'],
  sentinel: ['SentinelEvaluated', 'BreakerTripped', 'ReviewAcknowledged'],
  learning: ['FeedbackReleased', 'PosteriorUpdated', 'EpochCompleted'],
  borrowers: ['BorrowerArrived', 'PolicySelected'],
} as const;
type Group = keyof typeof GROUPS;
const LEVEL: Record<EventLevel, Tone> = { info: 'neutral', warning: 'warning', critical: 'critical' };

/** Tail of the append-only event store (polled while the run is live). */
export function EventLog({ sim }: { sim: Simulation }) {
  const [groups, setGroups] = useState<Group[]>(['lifecycle', 'sentinel', 'learning']);
  const types = groups.flatMap((g) => [...GROUPS[g]]);
  const q = useEvents(sim.id, sim, { type: types, order: 'desc', limit: 80 });
  return (
    <Card
      title="Event log"
      subtitle="Newest first; per-borrower events are sampled"
      actions={<Chips<Group> size="sm" label="Event types" options={(Object.keys(GROUPS) as Group[]).map((g) => ({ value: g, label: fmt.label(g) }))} value={groups} onChange={setGroups} />}
      bodyClassName="p-0"
    >
      <QueryState query={q} isEmpty={(d) => d.results.length === 0}>
        {(d) => (
          <ol className="max-h-80 overflow-y-auto font-mono text-[11.5px]" aria-live="polite">
            {d.results.map((e) => (
              <li key={e.seq} className="grid grid-cols-[3.5rem_4.5rem_1rem_10.5rem_1fr] items-baseline gap-2 border-b border-line px-4 py-1 last:border-b-0">
                <span className="num text-muted">#{e.seq}</span>
                <span className="num text-muted">{e.epoch ?? ''}</span>
                <span className="self-center">
                  <StatusGlyph tone={LEVEL[e.level]} />
                </span>
                <span className="truncate text-ink-2">{e.type}</span>
                <span className="truncate font-sans text-ink" title={e.message}>
                  {e.message}
                </span>
              </li>
            ))}
          </ol>
        )}
      </QueryState>
    </Card>
  );
}
