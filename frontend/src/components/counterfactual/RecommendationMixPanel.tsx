import type { RecommendationMix, SelectionMix } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { GroupedBars } from '@/charts/GroupedBars';
import { Legend } from '@/charts/Legend';
import { AMOUNT_BUCKETS, RATE_BUCKETS, TERM_BUCKETS, bucketLabel } from '@/lib/counterfactual';
import { fmt } from '@/lib/format';
import { policyColor } from '@/lib/palette';

const SERIES = [
  { key: 'historical', label: 'Historical (issued)', color: policyColor('historical') },
  { key: 'shadow', label: 'Shadow (recommended)', color: 'var(--series-1)' },
];

const FACETS: { key: 'by_term' | 'by_rate' | 'by_amount'; title: string; buckets: string[] }[] = [
  { key: 'by_term', title: 'Term', buckets: TERM_BUCKETS },
  { key: 'by_rate', title: 'Interest rate', buckets: RATE_BUCKETS },
  { key: 'by_amount', title: 'Amount', buckets: AMOUNT_BUCKETS },
];

const ppSigned = (v: number | null | undefined) => (typeof v === 'number' ? `${v >= 0 ? '+' : '−'}${fmt.num(Math.abs(v) * 100, 1)} pp` : '—');

function StatRow({ label, h, s, f }: { label: string; h: number | null | undefined; s: number | null | undefined; f: (v: number | null | undefined) => string }) {
  return (
    <tr className="border-t border-line">
      <th scope="row" className="px-3 py-1 text-left font-normal text-ink-2">
        {label}
      </th>
      <td className="px-3 py-1 text-right text-ink-2">{f(h)}</td>
      <td className="px-3 py-1 text-right font-semibold text-ink">{f(s)}</td>
    </tr>
  );
}

/** Where recommendations land vs what was issued: term / rate / amount buckets (shares of lending decisions). */
export function RecommendationMixPanel({ mix, className }: { mix: RecommendationMix; className?: string }) {
  const h: SelectionMix = mix.historical;
  const s: SelectionMix = mix.recommended;
  const top = Math.max(...FACETS.flatMap((f) => f.buckets.flatMap((b) => [h[f.key][b] ?? 0, s[f.key][b] ?? 0])), 0.1);
  const domain: [number, number] = [0, Math.min(1, Math.ceil(top * 10) / 10)];
  const ref = mix.reference;
  return (
    <ChartFrame
      className={className}
      title="Recommendation mix"
      caption={`Shares of lending decisions by bucket, recommended vs issued, for the same ${fmt.int(s.n)} borrowers (declines excluded: shadow declines ${fmt.pct(s.decline_share, 1)}).`}
      legend={<Legend items={SERIES.map((x) => ({ ...x, kind: 'rect' as const }))} />}
      table={{
        columns: [{ key: 'd', label: 'Dimension' }, { key: 'b', label: 'Bucket' }, { key: 'h', label: 'Historical', align: 'right' }, { key: 's', label: 'Shadow', align: 'right' }],
        rows: FACETS.flatMap((f) => f.buckets.map((b) => ({ d: f.title, b: bucketLabel(b), h: fmt.pct(h[f.key][b] ?? 0, 1), s: fmt.pct(s[f.key][b] ?? 0, 1) }))),
      }}
    >
      <div className="grid gap-3 px-1 lg:grid-cols-[2fr_5fr_4fr]">
        {FACETS.map((f) => (
          <div key={f.key} className="min-w-0">
            <div className="px-2 text-2xs font-medium text-ink-2">{f.title}</div>
            <div className="px-2 pb-0.5 text-2xs text-muted">Share of lending decisions</div>
            <GroupedBars
              data={f.buckets.map((b) => ({ x: bucketLabel(b), historical: h[f.key][b] ?? 0, shadow: s[f.key][b] ?? 0 }))}
              series={SERIES}
              format={(v) => fmt.pct(v, 1)}
              yTick={(v) => fmt.pct(v, 0)}
              domain={domain}
              yWidth={40}
              height={180}
            />
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 px-2 pb-1 lg:grid-cols-2">
        <table className="num w-full text-xs">
          <caption className="sr-only">Recommended vs historical summary</caption>
          <thead className="text-2xs text-muted">
            <tr>
              <th scope="col" className="px-3 py-1 text-left font-medium" />
              <th scope="col" className="px-3 py-1 text-right font-medium">Historical</th>
              <th scope="col" className="px-3 py-1 text-right font-medium">Shadow</th>
            </tr>
          </thead>
          <tbody>
            <StatRow label="Mean rate" h={h.mean_rate} s={s.mean_rate} f={(v) => fmt.pct(v, 2)} />
            <StatRow label="Mean amount" h={h.mean_amount} s={s.mean_amount} f={fmt.usd} />
            <StatRow label="36-month share" h={h.share_36m} s={s.share_36m} f={(v) => fmt.pct(v, 1)} />
            <StatRow label="At the top grid rate" h={h.share_at_max_rate} s={s.share_at_max_rate} f={(v) => fmt.pct(v, 1)} />
            <StatRow label="At the largest amount" h={h.share_at_max_amount} s={s.share_at_max_amount} f={(v) => fmt.pct(v, 1)} />
            <StatRow label="Decline share" h={h.decline_share} s={s.decline_share} f={(v) => fmt.pct(v, 1)} />
          </tbody>
        </table>
        <div className="rounded-md bg-surface-2 px-3 py-2 text-xs text-ink-2">
          <div className="mb-1 text-2xs font-medium text-muted">Against the sub-grade reference rate</div>
          {ref ? (
            <ul className="num space-y-0.5">
              <li>
                Shadow rates sit <strong className="text-ink">{ppSigned(ref.mean_rate_minus_reference)}</strong> from the reference on average (issued: {ppSigned(ref.historical_mean_rate_minus_reference)}); mean
                reference {fmt.pct(ref.mean_reference_rate, 1)}.
              </li>
              <li>
                {fmt.pct(ref.share_above_reference, 0)} above the reference · {fmt.pct(ref.share_within_corridor, 0)} inside the rate corridor · n {fmt.int(ref.n)}
              </li>
            </ul>
          ) : (
            <p className="text-muted">No reference rates for this fit (rate corridor off).</p>
          )}
        </div>
      </div>
    </ChartFrame>
  );
}
