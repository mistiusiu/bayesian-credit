/** Build the SimView used by all simulation payloads from a store record. */
import type { SimView } from './simViews';
import { db, fitModel, simSeriesFor, type SimRec } from './store';

export function simView(s: SimRec): SimView {
  const fit = db.fits.get(s.fitId)!;
  const model = fitModel(fit);
  return {
    id: s.id,
    config: s.config,
    series: simSeriesFor(s),
    model,
    done: s.done,
    epochTs: s.epochTs,
    startedMs: s.startedMs ?? s.createdMs,
    lifecycle: s.lifecycle,
    reviewId: s.reviewId,
    decision: s.decision,
    completed: s.status === 'completed',
    nReference: model.train.length,
  };
}
