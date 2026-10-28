import type { SelectionMix } from '@/api/types';
import { DataTable } from '@/components/ui/DataTable';
import { AMOUNT_BUCKETS, RATE_BUCKETS, bucketLabel } from '@/lib/counterfactual';
import { fmt } from '@/lib/format';
import { seqColor } from '@/lib/palette';
import { type Theme, useTheme } from '@/lib/theme';

export interface SelectionMixRow {
  key: string;
  label: string;
  color: string;
  mix: SelectionMix | null;
  /** Share of the top-3 arms in the window. */
  top3?: number | null;
  topLabels?: string[];
  distinct?: number | null;
}

/** Ordered buckets on the sequential ramp (low bucket light, high bucket dark). */
const RATE_T = [0.25, 0.44, 0.62, 0.81, 1];
const AMOUNT_T = [0.28, 0.52, 0.76, 1];

function MiniStack({ shares, buckets, ts, theme, label }: { shares: number[]; buckets: string[]; ts: number[]; theme: Theme; label: string }) {
  return (
    <span role="img" aria-label={`${label}: ${buckets.map((b, i) => `${bucketLabel(b)} ${fmt.pct(shares[i], 0)}`).join(', ')}`} className="flex h-2.5 w-36 gap-[2px]">
      {buckets.map((b, i) =>
        shares[i] > 0.0005 ? (
          <span key={b} title={`${bucketLabel(b)}: ${fmt.pct(shares[i], 1)}`} className="h-full min-w-[2px] first:rounded-l-[3px] last:rounded-r-[3px]" style={{ flexGrow: shares[i], flexBasis: 0, background: seqColor(ts[i], theme) }} />
        ) : null,
      )}
    </span>
  );
}

function BucketKey({ title, buckets, ts, theme }: { title: string; buckets: string[]; ts: number[]; theme: Theme }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-x-2 gap-y-0.5">
      <span className="text-muted">{title}</span>
      {buckets.map((b, i) => (
        <span key={b} className="inline-flex items-center gap-1">
          <span aria-hidden="true" className="inline-block size-2.5 rounded-[2px]" style={{ background: seqColor(ts[i], theme) }} />
          {bucketLabel(b)}
        </span>
      ))}
    </span>
  );
}

/** Decisions per policy by term / rate / amount (one row per policy), with the top-3 concentration. */
export function SelectionMixTable({ rows, caption }: { rows: SelectionMixRow[]; caption: string }) {
  const { theme } = useTheme();
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-x-6 gap-y-1 px-1 text-2xs text-ink-2" aria-hidden="true">
        <BucketKey title="Rate" buckets={RATE_BUCKETS} ts={RATE_T} theme={theme} />
        <BucketKey title="Amount" buckets={AMOUNT_BUCKETS} ts={AMOUNT_T} theme={theme} />
      </div>
      <DataTable<SelectionMixRow>
        caption={caption}
        rows={rows}
        rowKey={(r) => r.key}
        columns={[
          {
            key: 'p',
            header: 'Policy',
            render: (r) => (
              <span className="inline-flex items-center gap-1.5 font-medium">
                <span aria-hidden="true" className="inline-block h-0.5 w-3 rounded-full" style={{ background: r.color }} />
                {r.label}
              </span>
            ),
          },
          { key: 'n', header: 'Decisions', align: 'right', render: (r) => fmt.int(r.mix?.n), sortValue: (r) => r.mix?.n ?? null },
          { key: 'd', header: 'Declined', align: 'right', render: (r) => fmt.pct(r.mix?.decline_share, 1), sortValue: (r) => r.mix?.decline_share ?? null },
          { key: 't', header: '36-month', align: 'right', render: (r) => fmt.pct(r.mix?.share_36m, 0), sortValue: (r) => r.mix?.share_36m ?? null },
          { key: 'r', header: 'Mean rate', align: 'right', render: (r) => fmt.pct(r.mix?.mean_rate, 2), sortValue: (r) => r.mix?.mean_rate ?? null },
          { key: 'a', header: 'Mean amount', align: 'right', render: (r) => fmt.usd(r.mix?.mean_amount), sortValue: (r) => r.mix?.mean_amount ?? null },
          { key: 'rm', header: 'Rate mix', render: (r) => (r.mix ? <MiniStack label="Rate buckets" shares={RATE_BUCKETS.map((b) => r.mix?.by_rate[b] ?? 0)} buckets={RATE_BUCKETS} ts={RATE_T} theme={theme} /> : '—') },
          { key: 'am', header: 'Amount mix', render: (r) => (r.mix ? <MiniStack label="Amount buckets" shares={AMOUNT_BUCKETS.map((b) => r.mix?.by_amount[b] ?? 0)} buckets={AMOUNT_BUCKETS} ts={AMOUNT_T} theme={theme} /> : '—') },
          { key: 'c', header: 'Top-3 share', align: 'right', render: (r) => fmt.pct(r.top3, 0), sortValue: (r) => r.top3 ?? null },
          {
            key: 'top',
            header: 'Most chosen arms',
            render: (r) =>
              r.topLabels?.length ? (
                <ol className="text-2xs leading-4 text-ink-2">
                  {r.topLabels.map((l, i) => (
                    <li key={l} className="whitespace-nowrap">
                      <span className="text-muted">{i + 1}.</span> {l}
                    </li>
                  ))}
                </ol>
              ) : (
                '—'
              ),
          },
        ]}
      />
    </div>
  );
}
