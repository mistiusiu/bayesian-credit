import { useState } from 'react';
import { useAcknowledgeReview, useReviews } from '@/api/client';
import type { Review, ReviewAction } from '@/api/types';
import { Badge, StatusBadge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { FormError } from '@/components/ui/Field';
import { Empty, QueryState } from '@/components/ui/States';
import { REVIEW_ACTIONS } from '@/lib/domain';
import { fmt } from '@/lib/format';

function AckForm({ review }: { review: Review }) {
  const [action, setAction] = useState<ReviewAction>('resegment');
  const [note, setNote] = useState('');
  const ack = useAcknowledgeReview();
  const name = `review-${review.id}`;
  return (
    <form
      className="mt-3 space-y-2"
      onSubmit={(e) => {
        e.preventDefault();
        ack.mutate({ id: review.id, body: { action, note } });
      }}
    >
      <fieldset className="space-y-1.5">
        <legend className="mb-1 text-2xs font-medium text-muted">Decision</legend>
        {REVIEW_ACTIONS.map((a) => (
          <label key={a.key} className="flex cursor-pointer items-start gap-2 rounded-md border border-line p-2 text-xs has-[:checked]:border-accent-strong has-[:checked]:bg-accent-wash">
            <input type="radio" name={name} value={a.key} checked={action === a.key} onChange={() => setAction(a.key)} className="mt-0.5 accent-[var(--accent-strong)]" />
            <span>
              <span className="font-medium text-ink">{a.label}</span>
              <span className="block text-2xs text-muted">{a.blurb}</span>
            </span>
          </label>
        ))}
      </fieldset>
      <label className="block text-2xs font-medium text-muted">
        Note for the audit trail
        <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} className="field-input mt-1 h-auto py-1.5 text-xs" placeholder="e.g. FICO and DTI shifted after 2016-06; refine by fico_band" />
      </label>
      <FormError message={ack.error?.message} />
      <Button type="submit" variant="primary" size="sm" icon="check" loading={ack.isPending}>
        Acknowledge & {action}
      </Button>
    </form>
  );
}

/** Human-in-the-loop queue: the breaker never authorises changes on its own (G.5). */
export function ReviewQueue({ simId }: { simId: number | null }) {
  const pending = useReviews({ status: 'pending' });
  const history = useReviews({ status: 'acknowledged' });
  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Card title="Review queue" subtitle="Pending breaker reviews across all simulations" bodyClassName="p-0">
        <QueryState query={pending} isEmpty={(p) => p.results.length === 0} empty={<Empty title="No pending reviews" height={90}>Breaker trips in pause mode wait here for a decision.</Empty>}>
          {(p) => (
            <ul className="divide-y divide-line">
              {p.results.map((r) => (
                <li key={r.id} className="px-4 py-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-xs font-semibold text-ink">
                        Review #{r.id} · {fmt.epoch(r.epoch)}
                      </div>
                      <div className="truncate text-2xs text-muted">
                        sim #{r.simulation} {r.simulation_name}
                        {r.simulation === simId ? ' (selected)' : ''}
                      </div>
                    </div>
                    <Badge tone="critical">PSI {fmt.num(r.psi_aggregate, 3)}</Badge>
                  </div>
                  <p className="mt-1.5 text-xs text-ink-2">{r.reason}</p>
                  <p className="num mt-1 text-2xs text-muted">top features: {r.top_features.map((f) => `${f.feature} ${fmt.num(f.psi, 3)}`).join(' · ')}</p>
                  <p className="mt-1 text-2xs text-muted">segmentation: {r.segmentation_before.join(' × ')}</p>
                  <AckForm review={r} />
                </li>
              ))}
            </ul>
          )}
        </QueryState>
      </Card>
      <Card title="Review history" bodyClassName="p-0">
        <QueryState query={history} isEmpty={(p) => p.results.length === 0} empty={<Empty title="No acknowledged reviews" height={70} />}>
          {(p) => (
            <ul className="max-h-96 divide-y divide-line overflow-y-auto">
              {p.results.map((r) => (
                <li key={r.id} className="px-4 py-2.5 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-ink">
                      #{r.id} · {fmt.epoch(r.epoch)} · {r.action}
                    </span>
                    <StatusBadge status={r.status} label={r.auto ? 'auto' : 'human'} />
                  </div>
                  <div className="num mt-0.5 text-2xs text-muted">
                    sim #{r.simulation} · PSI {fmt.num(r.psi_aggregate, 3)} · {r.acknowledged_by} · {fmt.dateTime(r.acknowledged_at)}
                  </div>
                  {r.segmentation_after && <div className="mt-0.5 text-2xs text-ink-2">segmentation → {r.segmentation_after.join(' × ')}</div>}
                  {r.note && <p className="mt-0.5 text-2xs text-ink-2">“{r.note}”</p>}
                </li>
              ))}
            </ul>
          )}
        </QueryState>
      </Card>
    </div>
  );
}
