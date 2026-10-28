import { useXAxisScale, useYAxisScale } from 'recharts';

/**
 * Direct end-of-line labels rendered inside a Recharts chart. Items are placed in priority
 * order; a label that would sit within `minGap` px of one already placed is dropped (the
 * legend + tooltip carry it) instead of being nudged away from its line.
 */
export function EndLabels({ items, minGap = 13 }: { items: { key: string; label: string; x: number; y: number }[]; minGap?: number }) {
  const xs = useXAxisScale();
  const ys = useYAxisScale();
  if (!xs || !ys) return null;
  const placed: number[] = [];
  const out: { key: string; label: string; x: number; y: number }[] = [];
  for (const it of items) {
    const px = xs(it.x);
    const py = ys(it.y);
    if (px === undefined || py === undefined || !Number.isFinite(py)) continue;
    if (placed.some((p) => Math.abs(p - py) < minGap)) continue;
    placed.push(py);
    out.push({ key: it.key, label: it.label, x: px, y: py });
  }
  return (
    <g aria-hidden="true">
      {out.map((o) => (
        <text key={o.key} x={o.x + 7} y={o.y} dy="0.32em" fontSize={11} fill="var(--text-secondary)">
          {o.label}
        </text>
      ))}
    </g>
  );
}
