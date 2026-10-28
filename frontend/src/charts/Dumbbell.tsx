import { useState } from 'react';
import { useElementWidth } from '@/lib/useElementWidth';
import { fmt } from '@/lib/format';
import { niceDomain } from './kit';

export interface DumbbellRow {
  key: string;
  label: string;
  /** Reference value (gray dot). */
  a: number;
  /** Value of interest (accent dot). */
  b: number;
  note?: string;
}

/**
 * Dumbbell: per row, the reference value (gray) and the value of interest (blue) joined by
 * a line; the gap is labelled at the right. Shared x domain across rows.
 */
export function Dumbbell({ rows, aLabel, bLabel }: { rows: DumbbellRow[]; aLabel: string; bLabel: string }) {
  const [hover, setHover] = useState<string | null>(null);
  const [box, width] = useElementWidth<HTMLDivElement>(640);
  const domain = niceDomain(rows.flatMap((r) => [r.a, r.b]));
  const left = Math.min(250, Math.max(150, width * 0.3));
  const right = 76;
  const rowH = 22;
  const plotW = width - left - right;
  const x = (v: number) => left + ((v - domain[0]) / (domain[1] - domain[0] || 1)) * plotW;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => domain[0] + t * (domain[1] - domain[0]));
  const h = rows.length * rowH + 26;
  return (
    <div ref={box} className="w-full">
    <svg viewBox={`0 0 ${width} ${h}`} width={width} height={h} role="img" aria-label={`${bLabel} versus ${aLabel} per group`} className="block">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={4} y2={h - 20} stroke="var(--grid)" />
          <text x={x(t)} y={h - 6} textAnchor="middle" fontSize={11} fill="var(--text-axis)" className="num">
            {fmt.usdCompact(t)}
          </text>
        </g>
      ))}
      {domain[0] < 0 && <line x1={x(0)} x2={x(0)} y1={4} y2={h - 20} stroke="var(--axis)" />}
      {rows.map((r, i) => {
        const y = 12 + i * rowH;
        const on = hover === r.key;
        return (
          <g key={r.key} onPointerEnter={() => setHover(r.key)} onPointerLeave={() => setHover(null)}>
            <rect x={0} y={y - rowH / 2} width={width} height={rowH} fill={on ? 'var(--surface-2)' : 'transparent'} />
            <text x={left - 10} y={y} dy="0.32em" textAnchor="end" fontSize={11.5} fill="var(--text-secondary)">
              {r.label}
            </text>
            <line x1={x(r.a)} x2={x(r.b)} y1={y} y2={y} stroke="var(--text-axis)" strokeWidth={1.5} />
            <circle cx={x(r.a)} cy={y} r={4.5} fill="var(--text-axis)" stroke="var(--surface-1)" strokeWidth={2}>
              <title>{`${aLabel}: ${fmt.usd(r.a)}`}</title>
            </circle>
            <circle cx={x(r.b)} cy={y} r={4.5} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2}>
              <title>{`${bLabel}: ${fmt.usd(r.b)}${r.note ? ` · ${r.note}` : ''}`}</title>
            </circle>
            <text x={width - right + 8} y={y} dy="0.32em" fontSize={11} fill="var(--text-primary)" className="num">
              {r.b - r.a > 0.5 ? fmt.usdSigned(r.b - r.a) : 'same arm'}
            </text>
          </g>
        );
      })}
    </svg>
    </div>
  );
}
