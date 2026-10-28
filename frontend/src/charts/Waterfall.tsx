import { useState, type ReactNode } from 'react';
import { useElementWidth } from '@/lib/useElementWidth';
import { niceScale } from './kit';

export interface WaterfallBar {
  key: string;
  label: string;
  /** `total` columns grow from 0; `delta` columns float from the previous level. */
  kind: 'total' | 'delta';
  from: number;
  to: number;
  color: string;
  valueLabel: string;
  ariaLabel: string;
  tooltip?: ReactNode;
}

/** Column with a 4px rounded data end (the end the value moves to) and a square start. */
function colPath(x: number, w: number, y0: number, y1: number, r: number): string {
  const rr = Math.min(r, Math.abs(y1 - y0), w / 2);
  const s = y1 < y0 ? 1 : -1;
  return `M${x},${y0} V${y1 + s * rr} Q${x},${y1} ${x + rr},${y1} H${x + w - rr} Q${x + w},${y1} ${x + w},${y1 + s * rr} V${y0} Z`;
}

/**
 * Waterfall: a starting total, floating increments and an ending total on one axis,
 * joined by hairline connectors; every column carries its value at the data end.
 */
export function Waterfall({ bars, tick, label, height = 250 }: { bars: WaterfallBar[]; tick: (v: number) => string; label: string; height?: number }) {
  const [box, width] = useElementWidth<HTMLDivElement>(560);
  const [hover, setHover] = useState<number | null>(null);
  const { domain, ticks } = niceScale(bars.flatMap((b) => [b.from, b.to]), { includeZero: true, ticks: 4 });
  const left = 58;
  const right = 10;
  const top = 20;
  const bottom = 26;
  const plotW = Math.max(60, width - left - right);
  const plotH = height - top - bottom;
  const y = (v: number) => top + ((domain[1] - v) / (domain[1] - domain[0] || 1)) * plotH;
  const band = plotW / Math.max(bars.length, 1);
  const barW = Math.min(24, band * 0.6);
  const cx = (i: number) => left + band * i + band / 2;
  const tip = hover === null ? null : bars[hover];
  return (
    <div ref={box} className="relative w-full" onPointerLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="list" aria-label={label} className="block">
        {ticks.map((t) => (
          <g key={t} aria-hidden="true">
            <line x1={left} x2={width - right} y1={y(t)} y2={y(t)} stroke={t === 0 ? 'var(--axis)' : 'var(--grid)'} />
            <text x={left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-axis)" className="num">
              {tick(t)}
            </text>
          </g>
        ))}
        {bars.slice(0, -1).map((b, i) => (
          <line key={`c-${b.key}`} x1={cx(i) + barW / 2} x2={cx(i + 1) - barW / 2} y1={y(b.to)} y2={y(b.to)} stroke="var(--text-axis)" strokeWidth={1} aria-hidden="true" />
        ))}
        {bars.map((b, i) => {
          const up = b.to >= b.from;
          const y0 = y(b.from);
          const y1 = y(b.to);
          return (
            <g key={b.key} role="listitem" tabIndex={0} aria-label={b.ariaLabel} onPointerEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)} className="outline-none focus-visible:[&>rect]:stroke-[var(--focus)]">
              <rect x={cx(i) - band / 2 + 2} y={top - 16} width={band - 4} height={plotH + 16} rx={4} fill={hover === i ? 'var(--accent-wash)' : 'transparent'} stroke="transparent" strokeWidth={2} />
              {Math.abs(y1 - y0) >= 0.5 ? (
                <path d={colPath(cx(i) - barW / 2, barW, y0, y1, 4)} style={{ fill: b.color }} />
              ) : (
                <line x1={cx(i) - barW / 2} x2={cx(i) + barW / 2} y1={y1} y2={y1} style={{ stroke: b.color }} strokeWidth={2} />
              )}
              <text x={cx(i)} y={up ? y1 - 6 : y1 + 13} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--text-primary)" className="num">
                {b.valueLabel}
              </text>
              <text x={cx(i)} y={height - 8} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
                {b.label}
              </text>
            </g>
          );
        })}
      </svg>
      {tip?.tooltip && hover !== null && (
        <div
          className="pointer-events-none absolute z-20 w-max max-w-64 -translate-x-1/2 rounded-md border border-line bg-[var(--tooltip-bg)] px-2.5 py-2 text-xs shadow-pop"
          style={{ left: Math.min(Math.max(cx(hover), 110), width - 110), top: Math.min(y(Math.min(tip.from, tip.to)), height - 40) + 8 }}
        >
          {tip.tooltip}
        </div>
      )}
    </div>
  );
}
