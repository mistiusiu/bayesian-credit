import { useState } from 'react';
import { Link, useSearchParams } from 'react-router';
import { useSimulationControl, type ControlAction } from '@/api/client';
import type { Simulation } from '@/api/types';
import { Badge, StatusBadge, StatusGlyph } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { FormError } from '@/components/ui/Field';
import { ProgressBar } from '@/components/ui/Progress';
import { fmt } from '@/lib/format';
import { BREAKER_LABEL, BREAKER_TONE } from '@/lib/palette';
import { selectionSearch } from '@/lib/selection';

/** Run status, lifecycle controls and the kill → resume-from-checkpoint record (G.6). */
export function RunHeader({ sim }: { sim: Simulation }) {
  const control = useSimulationControl(sim.id);
  const [confirmChaos, setConfirmChaos] = useState(false);
  const [params] = useSearchParams();
  const act = (a: ControlAction) => control.mutate(a, { onSettled: () => setConfirmChaos(false) });
  const live = sim.status === 'running' || sim.status === 'queued';
  const rec = sim.last_recovery;
  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-semibold text-ink">
              #{sim.id} {sim.name}
            </h2>
            <StatusBadge status={sim.status} />
            <Badge tone={BREAKER_TONE[sim.breaker_state]}>breaker {BREAKER_LABEL[sim.breaker_state].toLowerCase()}</Badge>
          </div>
          <p className="num mt-1 text-xs text-muted">
            fit #{sim.fit} · {sim.config.policies.length} policies · {sim.config.feedback_mode} feedback · {sim.config.start} → {sim.config.end} · {sim.config.max_loans ? `${fmt.compact(sim.config.max_loans)} loans` : 'all loans'} · seed {sim.config.seed}{sim.config.hindsight_sample ? ` · hindsight on ${sim.config.hindsight_sample === 'elapsed' ? 'term-elapsed' : 'all resolved'} loans` : ''}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {sim.status === 'running' && (
            <Button size="sm" icon="pause" onClick={() => act('pause')} loading={control.isPending && control.variables === 'pause'}>
              Pause
            </Button>
          )}
          {(sim.status === 'paused' || sim.status === 'interrupted') && (
            <Button size="sm" variant="primary" icon="play" onClick={() => act('resume')} loading={control.isPending && control.variables === 'resume'}>
              {sim.status === 'interrupted' ? 'Resume from checkpoint' : 'Resume'}
            </Button>
          )}
          {['running', 'paused', 'awaiting_review', 'interrupted', 'queued'].includes(sim.status) && (
            <Button size="sm" variant="ghost" icon="stop" onClick={() => act('cancel')}>
              Cancel
            </Button>
          )}
          {sim.status === 'running' &&
            (confirmChaos ? (
              <span className="inline-flex items-center gap-1.5 rounded-md border border-critical/50 px-2 py-1 text-xs">
                SIGKILL worker pid {sim.worker.pid ?? '?'}?
                <Button size="sm" variant="danger" icon="bolt" onClick={() => act('chaos')}>
                  Kill
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setConfirmChaos(false)}>
                  Keep
                </Button>
              </span>
            ) : (
              <Button size="sm" variant="danger" icon="bolt" onClick={() => setConfirmChaos(true)} title="Kill the worker process to test resume-from-checkpoint (Table 10, state recovery)">
                Chaos
              </Button>
            ))}
        </div>
      </div>
      <div className="mt-3 grid items-center gap-x-6 gap-y-2 md:grid-cols-[minmax(0,1fr)_auto]">
        <ProgressBar value={sim.progress} label="Simulation progress" />
        <p className="num text-xs text-ink-2">
          epoch {sim.current_epoch ? fmt.epoch(sim.current_epoch) : '—'} · {sim.epochs_done}/{sim.epochs_total} · last checkpoint {sim.last_checkpoint_epoch ? fmt.epochShort(sim.last_checkpoint_epoch) : 'none'} (every {sim.config.checkpoint_every}) ·{' '}
          {sim.worker.alive ? `worker pid ${sim.worker.pid}` : 'no worker'} · restarts {sim.restarts}
        </p>
      </div>
      {sim.status === 'awaiting_review' && (
        <p className="mt-3 flex flex-wrap items-center gap-2 rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-ink">
          <StatusGlyph tone="warning" /> PSI breaker tripped at {fmt.epoch(sim.current_epoch)}; the run is paused until a human acknowledges the review ({sim.pending_reviews} pending).
          <Link to={{ pathname: '/sentinel', search: selectionSearch(params) }} className="font-medium text-accent-strong underline">
            Open the review queue
          </Link>
        </p>
      )}
      {sim.status === 'interrupted' && (
        <p className="mt-3 flex items-center gap-2 rounded-md border border-critical/50 bg-critical/10 px-3 py-2 text-xs text-ink">
          <StatusGlyph tone="critical" /> Worker killed at {fmt.epoch(rec?.killed_at_epoch)}. Resuming starts a new worker from the last checkpoint ({fmt.epoch(sim.last_checkpoint_epoch)}) and replays the epochs after it from the event store and saved RNG state.
        </p>
      )}
      {rec && sim.status !== 'interrupted' && (
        <div className="num mt-3 grid gap-2 rounded-md bg-surface-2 px-3 py-2 text-xs text-ink-2 md:grid-cols-5">
          <span>
            <span className="block text-2xs text-muted">Killed</span>
            {fmt.epoch(rec.killed_at_epoch)} · {fmt.dateTime(rec.killed_at)}
          </span>
          <span>
            <span className="block text-2xs text-muted">Resumed from checkpoint</span>
            {rec.resumed_from_epoch ? fmt.epoch(rec.resumed_from_epoch) : 'start'}
          </span>
          <span>
            <span className="block text-2xs text-muted">Replayed epochs</span>
            {rec.replayed_epochs}
          </span>
          <span className="inline-flex flex-col">
            <span className="block text-2xs text-muted">Deterministic replay</span>
            <span className="inline-flex items-center gap-1.5">
              <StatusGlyph tone={rec.deterministic_match === null ? 'info' : rec.deterministic_match ? 'good' : 'critical'} pulse={rec.deterministic_match === null && live} />
              {rec.deterministic_match === null ? 'replaying…' : rec.deterministic_match ? 'identical metrics' : 'mismatch'}
            </span>
          </span>
          <span>
            <span className="block text-2xs text-muted">Recovery time</span>
            {fmt.duration(rec.recovery_seconds)}
          </span>
        </div>
      )}
      <FormError message={control.error?.message} />
    </div>
  );
}
