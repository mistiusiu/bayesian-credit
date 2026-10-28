import type { PolicyEffectDiagnostics } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { DotWhisker, type WhiskerRow } from '@/charts/DotWhisker';
import { Badge } from '@/components/ui/Badge';
import { POLICY_EFFECT_FEATURES } from '@/lib/domain';
import { fmt } from '@/lib/format';
import type { Tone } from '@/lib/palette';

const Z = 1.959964;

const STATUS_TONE = (s: string): Tone => (s === 'fitted' ? 'good' : s === 'disabled' ? 'neutral' : 'warning');

const TEXT = 'Within-group effect of rate / 60-month term / payment-to-income / log amount on default log-odds';

/** Structured leaf prior (docs/MATH.md §2.6): β of the offset logistic regression, per meaningful step. */
export function PolicyEffectCard({ pe, className }: { pe: PolicyEffectDiagnostics | null | undefined; className?: string }) {
  const rows: WhiskerRow[] = POLICY_EFFECT_FEATURES.flatMap((f) => {
    const b = pe?.beta?.[f.key];
    if (typeof b !== 'number') return [];
    const se = pe?.std_errors?.[f.key];
    const eff = b * f.step;
    const half = typeof se === 'number' ? Z * se * Math.abs(f.step) : 0;
    const [lo, hi] = [eff - half, eff + half];
    return [
      {
        key: f.key,
        label: f.label,
        sub: f.stepLabel,
        value: eff,
        low: lo,
        high: hi,
        ariaLabel: `${f.label}, ${f.stepLabel}: ${fmt.numSigned(eff, 3)} on the log-odds, 95% interval ${fmt.numSigned(lo, 3)} to ${fmt.numSigned(hi, 3)}, odds ×${fmt.num(Math.exp(eff), 2)}`,
        tooltip: (
          <div className="num space-y-0.5">
            <div className="font-medium text-ink">
              {f.label} · {f.stepLabel}
            </div>
            <div>
              {fmt.numSigned(eff, 3)} logit · odds ×{fmt.num(Math.exp(eff), 2)}
            </div>
            <div className="text-muted">
              95% {fmt.numSigned(lo, 3)} to {fmt.numSigned(hi, 3)} · β {fmt.num(b, 3)} ± {fmt.num(se, 3)} per unit
            </div>
          </div>
        ),
      },
    ];
  });
  const status = pe?.status ?? 'not reported';
  const meta = pe
    ? [
        pe.n_fit !== undefined && `n ${fmt.int(pe.n_fit)}${pe.sample ? ` (${pe.sample} sample)` : ''}`,
        pe.converged !== undefined && (pe.converged ? `converged in ${pe.n_iter ?? '?'} Newton steps` : 'did not converge'),
        pe.max_abs_logit_shift !== undefined && `tilt clipped to ±${pe.max_abs_logit_shift} logit`,
        pe.installed === false && 'not installed (β = 0)',
      ].filter(Boolean)
    : [];
  return (
    <ChartFrame
      className={className}
      title="Structured leaf prior · policy effect"
      actions={<Badge tone={STATUS_TONE(status)}>{fmt.label(status)}</Badge>}
      caption={`${TEXT} (dot: β × step; whisker: 95% interval, ±1.96 SE). Fitted by offset logistic regression with features centred within each borrower group; a leaf's prior mean is tilted by t = clip(β·(x − x̄_g), ±${pe?.max_abs_logit_shift ?? 3}).`}
      xLabel={rows.length ? 'Change in default log-odds per step (0 = no effect)' : undefined}
      table={
        rows.length
          ? {
              columns: [
                { key: 'f', label: 'Feature' },
                { key: 'st', label: 'Step' },
                { key: 'b', label: 'β per unit', align: 'right' },
                { key: 'se', label: 'SE', align: 'right' },
                { key: 'e', label: 'Effect per step', align: 'right' },
                { key: 'ci', label: '95% interval', align: 'right' },
                { key: 'or', label: 'Odds ratio', align: 'right' },
              ],
              rows: POLICY_EFFECT_FEATURES.filter((f) => typeof pe?.beta?.[f.key] === 'number').map((f) => {
                const r = rows.find((x) => x.key === f.key)!;
                return { f: f.label, st: f.stepLabel, b: fmt.num(pe?.beta?.[f.key], 4), se: fmt.num(pe?.std_errors?.[f.key], 4), e: fmt.numSigned(r.value, 3), ci: `${fmt.numSigned(r.low, 3)} to ${fmt.numSigned(r.high, 3)}`, or: `×${fmt.num(Math.exp(r.value), 3)}` };
              }),
            }
          : undefined
      }
    >
      {rows.length ? (
        <div className="px-1">
          <DotWhisker rows={rows} tick={(v) => fmt.numSigned(v, 2)} label={TEXT} />
          <p className="num px-1 pt-1 text-2xs text-muted">{meta.join(' · ')}</p>
        </div>
      ) : (
        <p className="px-2 py-6 text-center text-xs text-muted">
          {pe ? `No tilt (β = 0)${pe.reason ? `: ${pe.reason}` : ''}.` : 'This fit does not report a policy effect (fitted before the structured prior, or served without it).'}
        </p>
      )}
    </ChartFrame>
  );
}
