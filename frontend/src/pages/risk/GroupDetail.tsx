import { useGroups, useRisk } from '@/api/client';
import type { Fit, HierarchyRow } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { DensityChart } from '@/charts/Density';
import { IntervalBar } from '@/charts/IntervalBar';
import { Legend } from '@/charts/Legend';
import { Card } from '@/components/ui/Card';
import { DataTable } from '@/components/ui/DataTable';
import { Meter } from '@/components/ui/Progress';
import { QueryState } from '@/components/ui/States';
import { StatTile } from '@/components/ui/StatTile';
import { recoverySourceLabel } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { useParam } from '@/lib/params';

/** Leaves of one group (arms with data) and the risk estimate of the selected leaf. */
export function GroupDetail({ fit, group, max }: { fit: Fit; group: string; max: number }) {
  const leaves = useGroups(fit.id, true, { level: 'leaf', group });
  const [armParam, setArm] = useParam('arm', null);
  const arm = armParam === null ? null : Number(armParam);
  const risk = useRisk(fit.id, true, group, arm);
  return (
    <div className="grid gap-4 2xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <Card
        title={`Leaves of ${fmt.group(group)}`}
        subtitle="Observed (group, arm) leaves. Prior mean = the group mean tilted on the logit scale by t = β·(x − x̄) for the arm's rate, term, payment-to-income and amount (0 = untilted). Select one for its posterior."
      >
        <QueryState query={leaves} isEmpty={(r) => r.length === 0} empty={<p className="text-xs text-muted">No observed leaves: every arm uses the group posterior.</p>}>
          {(rows) => (
            <DataTable<HierarchyRow>
              caption="Leaf posteriors"
              rows={rows}
              rowKey={(r) => r.arm ?? r.key}
              selectedKey={arm}
              onRowClick={(r) => setArm(r.arm === arm ? null : String(r.arm))}
              maxHeight={380}
              initialSort={{ key: 'n', dir: 'desc' }}
              columns={[
                { key: 'arm', header: 'Arm', render: (r) => <span className="font-medium">{r.dims.arm}</span>, sortValue: (r) => r.arm },
                { key: 'n', header: 'n', align: 'right', render: (r) => fmt.int(r.n_obs), sortValue: (r) => r.n_obs },
                { key: 'mean', header: 'P(default)', align: 'right', render: (r) => fmt.pct(r.mean), sortValue: (r) => r.mean },
                { key: 'hdi', header: '95% HDI', render: (r) => <IntervalBar mean={r.mean} low={r.hdi_low} high={r.hdi_high} prior={r.prior_mean} max={max} width={150} label={r.dims.arm ?? ''} /> },
                {
                  key: 'tilt',
                  header: 'Prior tilt',
                  title: 'prior_tilt_logit: logit shift of the leaf prior mean against the group mean',
                  align: 'right',
                  render: (r) => (typeof r.prior_tilt_logit === 'number' ? <span title={`prior mean ${fmt.pct(r.prior_mean)}`}>{fmt.numSigned(r.prior_tilt_logit, 2)}</span> : '—'),
                  sortValue: (r) => r.prior_tilt_logit ?? null,
                },
                { key: 'w', header: 'Data weight', render: (r) => <Meter value={r.shrinkage_weight} label="data weight" width={48} />, sortValue: (r) => r.shrinkage_weight },
                {
                  key: 'rec',
                  header: 'E[R]',
                  align: 'right',
                  render: (r) => (
                    <span className="whitespace-nowrap">
                      {fmt.pct(r.recovery?.mean, 0)}
                      {(r.recovery?.core_source ?? r.recovery?.source) !== 'leaf' && <span className="text-2xs text-muted"> ({recoverySourceLabel(r.recovery)})</span>}
                    </span>
                  ),
                },
              ]}
            />
          )}
        </QueryState>
      </Card>
      <QueryState query={risk} loadingLabel="Loading posterior">
        {(r) => (
          <div className="space-y-3">
            <ChartFrame
              title={`P(default) posterior · ${r.arm === null ? 'group' : r.label}`}
              caption={`Beta(${fmt.num(r.p_default.alpha, 1)}, ${fmt.num(r.p_default.beta, 1)}) at the ${r.p_default.level} level vs its prior; n ${fmt.int(r.p_default.n_obs)}, data weight ${fmt.pct(r.p_default.shrinkage_weight, 0)}.`}
              legend={<Legend items={[{ key: 'post', label: 'Posterior', color: 'var(--series-1)' }, { key: 'prior', label: 'Prior (shrinkage target)', color: 'var(--text-muted)' }]} />}
              yLabel="Density"
              xLabel="Probability of default"
              table={{ columns: [{ key: 'x', label: 'p' }, { key: 'post', label: 'Posterior', align: 'right' }, { key: 'prior', label: 'Prior', align: 'right' }], rows: r.pdf.p_default.filter((_, i) => i % 5 === 0).map((p, i) => ({ x: fmt.pct(p.x, 0), post: fmt.num(p.density, 2), prior: fmt.num(r.pdf.p_default_prior[i * 5]?.density, 2) })) }}
            >
              <DensityChart posterior={r.pdf.p_default} prior={r.pdf.p_default_prior} xMax={Math.min(1, Math.max(0.3, r.p_default.hdi_high * 1.6))} height={190} />
            </ChartFrame>
            <div className="grid grid-cols-2 gap-3">
              <StatTile label="E[p] (1 − E[R])" value={fmt.pct(r.expected_loss_fraction, 2)} hint="expected loss fraction" />
              <StatTile label="Recovery E[R | D]" value={fmt.pct(r.recovery.mean)} hint={`${recoverySourceLabel({ source: r.recovery.level, core_source: r.recovery.core_source })} · n ${fmt.int(r.recovery.n_obs)}`} />
              {r.reward && (
                <>
                  <StatTile label="Expected reward" value={fmt.usd(r.reward.mean)} hint={`σ ${fmt.usd(r.reward.std)}`} />
                  <StatTile label="Revenue − loss − funding" value={`${fmt.usdCompact(r.reward.revenue)} − ${fmt.usdCompact(r.reward.expected_loss)} − ${fmt.usdCompact(r.reward.funding_cost)}`} />
                  <StatTile
                    label="Realised interest ρ"
                    value={fmt.pct(r.reward.realization, 1)}
                    hint={typeof r.reward.gross_interest === 'number' ? `of ${fmt.usd(r.reward.gross_interest)} scheduled interest (prepayment)` : 'prepayment adjustment'}
                  />
                </>
              )}
            </div>
          </div>
        )}
      </QueryState>
    </div>
  );
}
