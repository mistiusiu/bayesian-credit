import type { InterestRealization } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Heatmap } from '@/charts/Heatmap';
import { ScaleLegend } from '@/charts/ScaleLegend';
import { realizationFor } from '@/lib/counterfactual';
import { GRADES } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { seqColor, seqStops } from '@/lib/palette';
import { useTheme } from '@/lib/theme';

const SOURCE_NOTE = { cell: 'own cell', term: 'fallback: term ρ', global: 'fallback: global ρ', contract: 'adjustment off' } as const;

/** ρ(grade, term): realised share of the scheduled interest of fully-paid training loans (docs/MATH.md §2.8). */
export function InterestRealizationCard({ ir, className }: { ir: InterestRealization | null | undefined; className?: string }) {
  const { theme } = useTheme();
  if (!ir) {
    return (
      <ChartFrame className={className} title="Prepayment · realised interest ρ" caption="Not reported by this fit (fitted before the prepayment adjustment).">
        <p className="px-2 py-6 text-center text-xs text-muted">Rewards value the full contractual schedule (ρ = 1).</p>
      </ChartFrame>
    );
  }
  const grades = [...new Set([...GRADES, ...Object.keys(ir.by_grade_term)])].sort();
  const terms = [...new Set(['36', '60', ...Object.keys(ir.by_term)])].sort((a, b) => Number(a) - Number(b));
  const cells = terms.map((t) => grades.map((g) => realizationFor(ir, g, t)));
  const values = cells.flat().map((c) => c.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const t = (v: number) => 0.2 + 0.8 * ((v - lo) / (hi - lo || 1));
  const off = ir.prepayment_adjustment === null;
  return (
    <ChartFrame
      className={className}
      title="Prepayment · realised interest ρ"
      caption={
        off
          ? 'Adjustment off for this fit (prepayment_adjustment = null): rewards value the full contract, ρ = 1.'
          : `Share of the scheduled interest that fully-paid training loans actually paid (${ir.sample ?? 'sample'}, n ${fmt.int(ir.n_fit)}); expected revenue and funding cost are multiplied by ρ. Cells with fewer than ${fmt.int(ir.min_count)} loans fall back to the term, then the global ρ (gray); clipped to [${ir.clip[0]}, ${ir.clip[1]}].`
      }
      legend={<ScaleLegend title="ρ, realised / scheduled interest" stops={seqStops(theme)} labels={[fmt.pct(lo, 0), fmt.pct(hi, 0)]} />}
      table={{
        columns: [{ key: 'g', label: 'Grade' }, ...terms.map((tm) => ({ key: tm, label: `${tm}m`, align: 'right' as const })), { key: 's', label: 'Source' }],
        rows: grades.map((g, gi) => ({
          g,
          ...Object.fromEntries(terms.map((tm, ti) => [tm, fmt.pct(cells[ti][gi].value, 1)])),
          s: terms.map((tm, ti) => `${tm}m ${SOURCE_NOTE[cells[ti][gi].source]}`).join(' · '),
        })),
      }}
    >
      <div className="px-2 pb-1">
        <Heatmap
          label="Realised interest share by term and grade"
          rowTitle="Term"
          colTitle="Grade"
          cellHeight={30}
          rowHeaderWidth="2.25rem"
          rows={terms.map((tm) => ({ key: tm, label: `${tm}m` }))}
          cols={grades.map((g) => ({ key: g, label: g }))}
          color={(v) => seqColor(t(v), theme)}
          cell={(ri, ci) => {
            const c = cells[ri][ci];
            const own = c.source === 'cell';
            return {
              value: own ? c.value : null,
              na: !own,
              text: fmt.pct(c.value, 0),
              ariaLabel: `Grade ${grades[ci]}, ${terms[ri]} months: ρ ${fmt.pct(c.value, 1)} (${SOURCE_NOTE[c.source]})`,
              tooltip: (
                <span className="num">
                  Grade {grades[ci]} · {terms[ri]}m: ρ {fmt.pct(c.value, 1)} <span className="text-muted">({SOURCE_NOTE[c.source]})</span>
                </span>
              ),
            };
          }}
        />
        <p className="num mt-2 text-2xs text-muted">
          Global ρ {fmt.pct(ir.global, 1)} · {terms.map((tm) => `${tm}m ${fmt.pct(ir.by_term[tm], 1)}`).join(' · ')}
        </p>
      </div>
    </ChartFrame>
  );
}
