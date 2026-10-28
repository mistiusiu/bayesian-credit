import { useFit, useMctm } from '@/api/client';
import type { Mctm, TransitionStructure } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Legend } from '@/charts/Legend';
import { OccupancyChart } from '@/charts/Occupancy';
import { SeriesLines } from '@/charts/SeriesLines';
import { PageHeader } from '@/components/layout/PageHeader';
import { Empty, Loading, QueryState } from '@/components/ui/States';
import { Segmented } from '@/components/ui/Segmented';
import { StatTile } from '@/components/ui/StatTile';
import { CREDIT_STATES, STATE_LABEL, STATE_STACK } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { stateColor } from '@/lib/palette';
import { useParam } from '@/lib/params';
import { useSelection } from '@/lib/selection';
import { StructureNote } from './StructureNote';
import { TransitionMatrix } from './TransitionMatrix';

function MctmView({ m }: { m: Mctm }) {
  const ts = m.trajectory.map((r) => r.t);
  const fromC = m.absorption[0];
  const fromD = m.absorption[1];
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <TransitionMatrix m={m} />
        <StructureNote m={m} />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <ChartFrame
          title={`State occupancy from ${STATE_LABEL[m.start]}`}
          caption={`π_t = π₀Pᵗ over ${m.horizon} months. The top of the delinquent band is P(Default, t) = π_D(t) + π_L(t); at maturity the remaining mass collapses (C → R, D splits p_DR : p_DL).`}
          legend={<Legend items={[...STATE_STACK].reverse().map((s) => ({ key: s, label: STATE_LABEL[s], color: stateColor(s), kind: 'rect' as const }))} />}
          yLabel="Share of loans in state"
          xLabel="Months since origination"
          table={{
            columns: [{ key: 't', label: 'Month', align: 'right' }, ...CREDIT_STATES.map((s) => ({ key: s, label: STATE_LABEL[s], align: 'right' as const }))],
            rows: m.trajectory.filter((r) => r.t % 6 === 0 || r.t === m.horizon).map((r) => ({ t: r.t, current: fmt.pct(r.current), delinquent: fmt.pct(r.delinquent), recovered: fmt.pct(r.recovered), loss: fmt.pct(r.loss) })),
          }}
        >
          <OccupancyChart rows={m.trajectory} height={270} maturity={m.horizon} />
        </ChartFrame>
        <ChartFrame
          title="P(Default | t)"
          caption={`Probability of being delinquent or absorbed in loss by month t; the dot is P(loss) after the maturity collapse at t = ${m.horizon}: ${fmt.pct(m.maturity.p_default)}.`}
          legend={<Legend items={[{ key: 'pd', label: 'P(Default, t) = π_D + π_L', color: 'var(--series-1)' }, { key: 'pl', label: 'P(absorbed loss, t) = π_L', color: stateColor('loss') }]} />}
          yLabel="Probability"
          xLabel="Months since origination"
          table={{
            columns: [{ key: 't', label: 'Month', align: 'right' }, { key: 'pd', label: 'P(Default)', align: 'right' }, { key: 'pl', label: 'P(loss)', align: 'right' }],
            rows: m.p_default_curve.filter((r) => r.t % 6 === 0 || r.t === m.horizon).map((r) => ({ t: r.t, pd: fmt.pct(r.p_default, 2), pl: fmt.pct(r.p_loss, 2) })),
          }}
        >
          <SeriesLines
            xs={ts}
            xTick={(t) => String(t)}
            xLabelFor={(t) => `Month ${t}`}
            yTick={(v) => fmt.pct(v, 0)}
            format={(v) => fmt.pct(v, 2)}
            height={270}
            lines={[
              { key: 'pd', label: 'P(Default, t)', color: 'var(--series-1)', values: m.p_default_curve.map((r) => r.p_default) },
              { key: 'pl', label: 'P(absorbed loss, t)', color: stateColor('loss'), values: m.p_default_curve.map((r) => r.p_loss) },
            ]}
            dots={[{ x: m.horizon, y: m.maturity.p_default, label: 'at maturity' }]}
          />
        </ChartFrame>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 2xl:grid-cols-8">
        <StatTile label="P(Recovered | start C)" value={fmt.pct(fromC[0])} hint="eventual absorption" />
        <StatTile label="P(Loss | start C)" value={fmt.pct(fromC[1])} />
        <StatTile label="P(Recovered | start D)" value={fmt.pct(fromD[0])} />
        <StatTile label="P(Loss | start D)" value={fmt.pct(fromD[1])} />
        <StatTile label="E[T] from Current" value={`${fmt.num(m.expected_steps[0], 1)} mo`} hint="to absorption" />
        <StatTile label="E[T] from Delinquent" value={`${fmt.num(m.expected_steps[1], 1)} mo`} />
        <StatTile label={`P(loss) at t = ${m.horizon}`} value={fmt.pct(m.maturity.p_default)} hint="after maturity collapse" />
        <StatTile label="Holdout log-lik" value={fmt.num(m.log_likelihood / m.n_transitions, 4)} hint={`nats / loan-month · n ${fmt.compact(m.n_transitions)}`} />
      </div>
    </div>
  );
}

export function MctmPage() {
  const sel = useSelection();
  const fit = useFit(sel.fitId).data ?? sel.fit;
  const ready = fit?.status === 'ready';
  const [structure, setStructure] = useParam('structure', fit?.config.mctm.structure ?? 'proposal');
  const [start, setStart] = useParam('start', 'current');
  const [horizon, setHorizon] = useParam('horizon', '60');
  const q = useMctm(fit?.id, ready, { structure, horizon: Number(horizon), start });
  return (
    <>
      <PageHeader
        title="MCTM"
        description="Markov chain over credit states S = {Current, Delinquent, Recovered, Absorbed Loss}; R and L are absorbing. Interim behaviour enters the reward before terminal outcomes are known."
      />
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <Segmented<TransitionStructure>
          label="Transition structure"
          value={structure as TransitionStructure}
          onChange={setStructure}
          options={[
            { value: 'proposal', label: 'Proposal' },
            { value: 'extended', label: 'Extended' },
          ]}
        />
        <Segmented<string>
          label="Start state"
          value={start}
          onChange={setStart}
          options={[
            { value: 'current', label: 'Start Current' },
            { value: 'delinquent', label: 'Start Delinquent' },
          ]}
        />
        <Segmented<string>
          label="Horizon"
          value={horizon}
          onChange={setHorizon}
          options={[
            { value: '36', label: '36 months' },
            { value: '60', label: '60 months' },
          ]}
        />
        {fit && <span className="text-xs text-muted">fit #{fit.id} configured as {fit.config.mctm.structure}, Dirichlet prior {fit.config.mctm.prior_strength}</span>}
      </div>
      {!fit && sel.loading ? (
        <Loading label="Resolving selection" height={240} />
      ) : !fit ? (
        <Empty title="No fit selected">The transition matrix is estimated as part of a fit (Risk models page).</Empty>
      ) : !ready ? (
        <Empty title="Fit is still running" />
      ) : (
        <QueryState query={q} loadingLabel="Propagating the chain">
          {(m) => <MctmView m={m} />}
        </QueryState>
      )}
    </>
  );
}
