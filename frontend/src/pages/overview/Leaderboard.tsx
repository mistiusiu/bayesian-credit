import type { PolicyHeadline, Simulation } from '@/api/types';
import { DataTable } from '@/components/ui/DataTable';
import { policyLabel } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { policyColor } from '@/lib/palette';

export function Leaderboard({ sim }: { sim: Simulation }) {
  const hist = sim.headline.find((h) => h.policy === 'historical');
  return (
    <DataTable<PolicyHeadline>
      caption="Policy headline metrics"
      rows={sim.headline}
      rowKey={(r) => r.policy}
      initialSort={{ key: 'cum', dir: 'desc' }}
      empty="No completed epochs yet"
      columns={[
        {
          key: 'policy',
          header: 'Policy',
          render: (r) => (
            <span className="inline-flex items-center gap-2">
              <span aria-hidden="true" className="inline-block h-0.5 w-3.5 rounded-full" style={{ background: policyColor(r.policy) }} />
              {policyLabel(r.policy)}
            </span>
          ),
        },
        { key: 'cum', header: 'Cumulative reward', align: 'right', render: (r) => fmt.usdCompact(r.cum_reward), sortValue: (r) => r.cum_reward },
        {
          key: 'uplift',
          header: 'vs historical',
          align: 'right',
          render: (r) => (hist && r.policy !== 'historical' ? fmt.usdCompactSigned(r.cum_reward - hist.cum_reward) : '—'),
          sortValue: (r) => r.cum_reward - (hist?.cum_reward ?? 0),
        },
        { key: 'regret', header: 'Cumulative regret', align: 'right', render: (r) => fmt.usdCompact(r.cum_regret), sortValue: (r) => r.cum_regret },
        { key: 'mean', header: 'Reward / decision', align: 'right', render: (r) => fmt.usd(r.mean_reward), sortValue: (r) => r.mean_reward },
        { key: 'realized', header: 'Realized (to date)', align: 'right', render: (r) => fmt.usdCompact(r.cum_realized), sortValue: (r) => r.cum_realized },
        { key: 'decline', header: 'Decline rate', align: 'right', render: (r) => fmt.pct(r.decline_rate), sortValue: (r) => r.decline_rate },
        { key: 'amount', header: 'Mean amount', align: 'right', render: (r) => fmt.usd(r.mean_amount), sortValue: (r) => r.mean_amount },
        { key: 'rate', header: 'Mean rate', align: 'right', render: (r) => fmt.pct(r.mean_rate), sortValue: (r) => r.mean_rate },
      ]}
    />
  );
}
