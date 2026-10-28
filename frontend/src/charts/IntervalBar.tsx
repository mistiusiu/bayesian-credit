import { fmt } from '@/lib/format';

/**
 * Posterior mean (dot) with its 95% HDI (bar) on a shared [0, max] domain, plus the prior
 * mean as a tick: the visual unit of the hierarchy explorer.
 */
export function IntervalBar({
  mean,
  low,
  high,
  prior,
  max,
  width = 180,
  label,
}: {
  mean: number;
  low: number;
  high: number;
  prior?: number | null;
  max: number;
  width?: number;
  label: string;
}) {
  const h = 16;
  const pad = 5;
  const x = (v: number) => pad + (Math.min(Math.max(v, 0), max) / max) * (width - 2 * pad);
  const clipped = high > max;
  return (
    <svg width={width} height={h} role="img" aria-label={`${label}: mean ${fmt.pct(mean)}, 95% HDI ${fmt.pct(low)} to ${fmt.pct(high)}${prior != null ? `, prior ${fmt.pct(prior)}` : ''}`}>
      <line x1={pad} x2={width - pad} y1={h / 2} y2={h / 2} stroke="var(--grid)" strokeWidth={1} />
      <line x1={x(low)} x2={x(high)} y1={h / 2} y2={h / 2} stroke="var(--series-1)" strokeWidth={2} strokeLinecap="round" opacity={0.75} />
      {clipped && <path d={`M${width - pad - 4} ${h / 2 - 3} L${width - pad} ${h / 2} L${width - pad - 4} ${h / 2 + 3}`} fill="none" stroke="var(--series-1)" strokeWidth={1.5} />}
      {prior != null && <line x1={x(prior)} x2={x(prior)} y1={2} y2={h - 2} stroke="var(--text-muted)" strokeWidth={1.5} />}
      <circle cx={x(mean)} cy={h / 2} r={4} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} />
    </svg>
  );
}
