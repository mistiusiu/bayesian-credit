import { useState, type ReactNode } from 'react';
import { useElementWidth } from '@/lib/useElementWidth';
import { niceScale } from './kit';

export interface WhiskerRow {
  key: string;
  label: string;
  /** Second, smaller line under the label (e.g. the unit step). */
  sub?: string;
  value: number;
  low: number;
  high: number;
  ariaLabel: string;
  tooltip?: ReactNode;
}

/**
 * Dot-and-whisker on one shared axis: dot = estimate, whisker = interval, solid zero
 * reference. Rows are focusable; hover and focus show the same tooltip.
 */
export function DotWhisker({ rows, tick, label, color = 'var(--series-1)' }: { rows: WhiskerRow[]; tick: (v: number) => string; label: string; color?: string }) {
  const [box, width] = useElementWidth<HTMLDivElement>(480);
  const [hover, setHover] = useState<number | null>(null);
  const { domain, ticks } = niceScale(rows.flatMap((r) => [r.low, r.high, r.value]), { includeZero: true, ticks: 4 });
  const left = Math.min(190, Math.max(118, width * 0.34));
  const right = 18;
  const rowH = 40;
  const top = 4;
  const plotW = Math.max(40, width - left - right);
  const x = (v: number) => left + ((v - domain[0]) / (domain[1] - domain[0] || 1)) * plotW;
  const plotBottom = top + rows.length * rowH;
  const h = plotBottom + 22;
  const tip = hover === null ? null : rows[hover];
  return (
    <div ref={box} className="relative w-full" onPointerLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${width} ${h}`} width={width} height={h} role="list" aria-label={label} className="block">
        {ticks.map((t) => (
          <g key={t} aria-hidden="true">
            <line x1={x(t)} x2={x(t)} y1={top} y2={plotBottom} stroke="var(--grid)" />
            <text x={x(t)} y={h - 6} textAnchor="middle" fontSize={11} fill="var(--text-axis)" className="num">
              {tick(t)}
            </text>
          </g>
        ))}
        <line x1={x(0)} x2={x(0)} y1={top} y2={plotBottom} stroke="var(--axis)" strokeWidth={1.5} aria-hidden="true" />
        {rows.map((r, i) => {
          const y = top + i * rowH + rowH / 2;
          return (
            <g
              key={r.key}
              role="listitem"
              tabIndex={0}
              aria-label={r.ariaLabel}
              onPointerEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              onBlur={() => setHover(null)}
              className="outline-none focus-visible:[&>rect]:stroke-[var(--focus)]"
            >
              <rect x={0} y={y - rowH / 2} width={width} height={rowH} rx={4} fill={hover === i ? 'var(--surface-2)' : 'transparent'} stroke="transparent" strokeWidth={2} />
              <text x={left - 12} y={r.sub ? y - 6 : y} dy="0.32em" textAnchor="end" fontSize={11.5} fill="var(--text-secondary)">
                {r.label}
              </text>
              {r.sub && (
                <text x={left - 12} y={y + 8} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--text-muted)">
                  {r.sub}
                </text>
              )}
              <line x1={x(r.low)} x2={x(r.high)} y1={y} y2={y} stroke={color} strokeWidth={2} strokeLinecap="round" />
              <line x1={x(r.low)} x2={x(r.low)} y1={y - 4} y2={y + 4} stroke={color} strokeWidth={2} strokeLinecap="round" />
              <line x1={x(r.high)} x2={x(r.high)} y1={y - 4} y2={y + 4} stroke={color} strokeWidth={2} strokeLinecap="round" />
              <circle cx={x(r.value)} cy={y} r={5} fill={color} stroke="var(--surface-1)" strokeWidth={2} />
            </g>
          );
        })}
      </svg>
      {tip?.tooltip && hover !== null && (
        <div
          className="pointer-events-none absolute z-20 w-max max-w-72 -translate-x-1/2 rounded-md border border-line bg-[var(--tooltip-bg)] px-2.5 py-2 text-xs shadow-pop"
          style={{ left: Math.min(Math.max(x(tip.value), 120), width - 120), top: top + hover * rowH + rowH + 2 }}
        >
          {tip.tooltip}
        </div>
      )}
    </div>
  );
}
