import type { Mctm } from '@/api/types';
import { Card } from '@/components/ui/Card';
import { fmt } from '@/lib/format';

export function StructureNote({ m }: { m: Mctm }) {
  const proposal = m.structure === 'proposal';
  return (
    <Card title={proposal ? 'Proposal structure (§3.3)' : 'Extended structure'} subtitle={proposal ? 'P[C,R] = P[C,L] = 0' : 'C → R allowed (prepayment / payoff)'}>
      <div className="space-y-3 text-xs leading-relaxed text-ink-2">
        {proposal ? (
          <>
            <p>
              Exactly the matrix written in the proposal: a loan can leave <strong className="text-ink">Current</strong> only through <strong className="text-ink">Delinquent</strong>. &ldquo;Recovered&rdquo; means repayment that follows a delinquency.
            </p>
            <p>
              A never-delinquent payoff is treated as right-censoring when the matrix is estimated, and as the <em>maturity collapse</em> when predicting: at the term, Current mass moves to Recovered and Delinquent mass splits in the ratio p<sub>DR</sub> : p<sub>DL</sub>.
            </p>
            <p>
              Consequence of the zeros: absorption probabilities from C and D are identical (every path to absorption passes through D): P(L) = {fmt.pct(m.absorption[0][1])} from both, and the chain alone implies a long expected life of {fmt.num(m.expected_steps[0], 0)} months from Current.
            </p>
          </>
        ) : (
          <>
            <p>
              Adds C → R so the chain is a full loan-outcome model: performing loans can be absorbed directly through prepayment or scheduled payoff (P[C,R] = {fmt.num(m.matrix[0][2], 4)} per month).
            </p>
            <p>
              Absorption from Current is then dominated by payoff: P(R | C) = {fmt.pct(m.absorption[0][0])} vs P(R | D) = {fmt.pct(m.absorption[1][0])}, and the expected time to absorption from Current drops to {fmt.num(m.expected_steps[0], 0)} months.
            </p>
            <p>C → L stays structurally zero: a loss is always preceded by delinquency.</p>
          </>
        )}
        <p className="num rounded-md bg-surface-2 px-3 py-2 text-2xs text-muted">
          Holdout row error (mean |ΔP|): C {fmt.num(m.row_prediction_error[0], 5)} · D {fmt.num(m.row_prediction_error[1], 5)} · log-likelihood {fmt.num(m.log_likelihood, 0)} nats over {fmt.int(m.n_transitions)} loan-months
        </p>
      </div>
    </Card>
  );
}
