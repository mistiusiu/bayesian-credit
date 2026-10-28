import { useState } from 'react';
import type { RiskAversionPoint } from '@/api/types';
import { Card } from '@/components/ui/Card';
import { Segmented } from '@/components/ui/Segmented';
import { StatTile } from '@/components/ui/StatTile';
import { cn } from '@/lib/cn';
import { lambdaKeys, upliftFraction } from '@/lib/counterfactual';
import { fmt } from '@/lib/format';

const COLUMNS: { key: string; label: string; cell: (p: RiskAversionPoint) => string }[] = [
  { key: 'profit', label: 'Shadow profit', cell: (p) => fmt.usdCompact(p.expected_profit_shadow) },
  { key: 'uplift', label: 'Uplift', cell: (p) => fmt.usdCompactSigned(p.uplift) },
  { key: 'pct', label: 'Uplift %', cell: (p) => fmt.pctSigned(upliftFraction(p.uplift, p.expected_profit_historical), 0) },
  { key: 't36', label: '36-month', cell: (p) => fmt.pct(p.share_36m_recommended, 0) },
  { key: 'amt', label: 'Mean amount', cell: (p) => fmt.usd(p.mean_amount_recommended) },
  { key: 'rate', label: 'Mean rate', cell: (p) => fmt.pct(p.mean_rate_recommended, 2) },
  { key: 'dec', label: 'Declined', cell: (p) => fmt.pct(p.decline_share, 1) },
  { key: 'maxa', label: 'At max amount', cell: (p) => fmt.pct(p.share_at_max_amount_recommended, 0) },
];

/** The shadow policy re-scored with mean − λ·std at λ = 0, 0.5, 1 on the same tables and eligibility. */
export function RiskAversionSweep({ points, configured, className }: { points: Record<string, RiskAversionPoint>; configured?: number | null; className?: string }) {
  const keys = lambdaKeys(points);
  const closest = configured == null ? keys[0] : [...keys].sort((a, b) => Math.abs(Number(a) - configured) - Math.abs(Number(b) - configured))[0];
  const [picked, setPicked] = useState<string | null>(null);
  const key = picked !== null && keys.includes(picked) ? picked : closest;
  const p = key ? points[key] : undefined;
  if (!p) return null;
  const pct = upliftFraction(p.uplift, p.expected_profit_historical);
  return (
    <Card
      className={className}
      title="Risk appetite (λ sweep)"
      subtitle="Arms scored by mean − λ·std; a larger λ gives up expected profit for lower variance."
      actions={<Segmented<string> size="sm" label="Risk aversion λ" value={key} onChange={setPicked} options={keys.map((k) => ({ value: k, label: `λ = ${Number(k)}`, title: configured != null && Number(k) === configured ? "the fit's λ" : undefined }))} />}
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
        <StatTile label="Uplift" value={fmt.usdCompactSigned(p.uplift)} delta={{ text: `${fmt.pctSigned(pct, 0)} vs historical`, good: p.uplift >= 0 }} />
        <StatTile label="Shadow expected profit" value={fmt.usdCompact(p.expected_profit_shadow)} hint={`historical ${fmt.usdCompact(p.expected_profit_historical)}`} />
        <StatTile label="36-month share" value={fmt.pct(p.share_36m_recommended, 0)} hint="of lending offers" />
        <StatTile label="Mean amount" value={fmt.usd(p.mean_amount_recommended)} hint={`${fmt.pct(p.share_at_max_amount_recommended, 0)} at the largest`} />
        <StatTile label="Mean rate" value={fmt.pct(p.mean_rate_recommended, 2)} hint={`${fmt.pct(p.share_at_max_rate_recommended, 0)} at the top rate`} />
        <StatTile label="Declined" value={fmt.pct(p.decline_share, 1)} hint="null arm chosen" />
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="num w-full min-w-[40rem] text-xs">
          <caption className="sr-only">Shadow policy by risk aversion λ</caption>
          <thead className="bg-surface-2 text-2xs text-muted">
            <tr>
              <th scope="col" className="px-3 py-1.5 text-left font-medium">λ</th>
              {COLUMNS.map((c) => (
                <th key={c.key} scope="col" className="px-3 py-1.5 text-right font-medium">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k} aria-current={k === key ? 'true' : undefined} className={cn('border-t border-line', k === key ? 'bg-accent-wash text-ink' : 'text-ink-2')}>
                <th scope="row" className="px-3 py-1.5 text-left font-medium">
                  {Number(k)}
                  {configured != null && Number(k) === configured && <span className="ml-1.5 text-2xs font-normal text-muted">fit</span>}
                </th>
                {COLUMNS.map((c) => (
                  <td key={c.key} className="px-3 py-1.5 text-right">
                    {c.cell(points[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
