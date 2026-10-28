import { useCalibration } from '@/api/client';
import { ReliabilityChart, RocChart } from '@/charts/Calibration';
import { ChartFrame } from '@/charts/ChartFrame';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';

export function CalibrationPanel({ fitId }: { fitId: number }) {
  const q = useCalibration(fitId, true);
  return (
    <QueryState query={q} loadingLabel="Scoring the holdout">
      {(c) => (
        <div className="grid gap-4 xl:grid-cols-2">
          <ChartFrame
            title="Reliability diagram"
            caption={`Observed default rate against mean predicted P(default) in 10 prediction deciles; on the diagonal = calibrated. ECE ${fmt.pct(c.ece, 2)}, holdout n ${fmt.int(c.n_test)}, base rate ${fmt.pct(c.base_rate)}.`}
            yLabel="Observed default rate"
            xLabel="Mean predicted P(default)"
            table={{
              columns: [
                { key: 'b', label: 'Decile' },
                { key: 'r', label: 'Predicted range' },
                { key: 'n', label: 'n', align: 'right' },
                { key: 'p', label: 'Mean predicted', align: 'right' },
                { key: 'o', label: 'Observed', align: 'right' },
              ],
              rows: c.bins.map((b) => ({ b: b.bin + 1, r: `${fmt.pct(b.lo)}–${fmt.pct(b.hi)}`, n: fmt.int(b.n), p: fmt.pct(b.mean_predicted), o: fmt.pct(b.observed_rate) })),
            }}
          >
            <ReliabilityChart bins={c.bins} height={260} />
          </ChartFrame>
          <ChartFrame
            title={`ROC curve · AUC ${fmt.num(c.auc, 3)}`}
            caption={`True vs false positive rate as the P(default) threshold falls; the diagonal is a random ranking. Brier ${fmt.num(c.brier, 4)}, log loss ${fmt.num(c.log_loss, 4)}.`}
            yLabel="True positive rate"
            xLabel="False positive rate"
            table={{
              columns: [{ key: 't', label: 'Threshold', align: 'right' }, { key: 'f', label: 'FPR', align: 'right' }, { key: 'p', label: 'TPR', align: 'right' }],
              rows: c.roc.filter((_, i) => i % 4 === 0).map((r) => ({ t: fmt.prob(r.threshold), f: fmt.pct(r.fpr), p: fmt.pct(r.tpr) })),
            }}
          >
            <RocChart roc={c.roc} auc={c.auc} height={260} />
          </ChartFrame>
        </div>
      )}
    </QueryState>
  );
}
