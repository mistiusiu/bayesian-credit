/** Shared chart chrome: recessive hairline axes/grid, tick helpers, domains, breaker runs. */
import type { BreakerState } from '@/api/types';
import { fmt } from '@/lib/format';

export const AXIS = {
  axisLine: { stroke: 'var(--axis)' },
  tickLine: false,
  tick: { fill: 'var(--text-axis)', fontSize: 11 },
  tickMargin: 6,
} as const;

/** Solid hairline grid, horizontal only. */
export const GRID = { stroke: 'var(--grid)', strokeDasharray: undefined, vertical: false } as const;

export const MARGIN = { top: 8, right: 14, bottom: 2, left: 2 };

export const CURSOR_LINE = { stroke: 'var(--text-axis)', strokeWidth: 1 };
export const CURSOR_BAND = { fill: 'var(--accent-wash)' };
export const ACTIVE_DOT = { r: 4, stroke: 'var(--surface-1)', strokeWidth: 2 };

/** Evenly spaced integer tick positions (≤ max) over [0, n−1], always including the ends. */
export function indexTicks(n: number, max = 8): number[] {
  if (n <= 0) return [];
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = Math.ceil((n - 1) / (max - 1));
  const ticks: number[] = [];
  for (let i = 0; i < n; i += step) ticks.push(i);
  if (ticks[ticks.length - 1] !== n - 1) {
    if (n - 1 - ticks[ticks.length - 1] < step / 2) ticks[ticks.length - 1] = n - 1;
    else ticks.push(n - 1);
  }
  return ticks;
}

/** January ticks for a monthly epoch series (falls back to evenly spaced). */
export function yearTicks(epochs: string[]): number[] {
  const t = epochs.flatMap((e, i) => (e.endsWith('-01') ? [i] : []));
  return t.length >= 2 ? t : indexTicks(epochs.length, 6);
}

export const epochTick = (epochs: string[]) => (i: number) => {
  const e = epochs[Math.round(i)];
  return e ? (e.endsWith('-01') ? e.slice(0, 4) : fmt.epochShort(e)) : '';
};

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

/** Nice domain plus its tick positions (multiples of a 1 / 2 / 2.5 / 5 step; zero included when asked). */
export function niceScale(values: number[], opts: { includeZero?: boolean; ticks?: number } = {}): { domain: [number, number]; ticks: number[] } {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1] };
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (opts.includeZero ?? true) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (lo === hi) {
    hi = lo + (Math.abs(lo) || 1);
  }
  const step = niceStep((hi - lo) / (opts.ticks ?? 5));
  const d0 = Math.floor(lo / step) * step;
  const d1 = Math.ceil(hi / step) * step;
  const n = Math.round((d1 - d0) / step);
  const ticks = Array.from({ length: n + 1 }, (_, i) => Math.round((d0 + i * step) / step) * step);
  return { domain: [d0, d1], ticks };
}

/** Nice [min, max] that includes zero when the data straddle or touch it. */
export function niceDomain(values: number[], opts: { includeZero?: boolean; ticks?: number } = {}): [number, number] {
  return niceScale(values, opts).domain;
}

export interface Run {
  from: number;
  to: number;
  state: BreakerState;
}

/** Contiguous runs of non-closed breaker states (for open / half-open shading). */
export function breakerRuns(states: BreakerState[]): Run[] {
  const runs: Run[] = [];
  states.forEach((s, i) => {
    if (s === 'closed') return;
    const last = runs[runs.length - 1];
    if (last && last.state === s && last.to === i - 1) last.to = i;
    else runs.push({ from: i, to: i, state: s });
  });
  return runs;
}

export const RUN_FILL: Record<BreakerState, string> = {
  closed: 'transparent',
  open: 'var(--band-open)',
  half_open: 'var(--band-half)',
};

/**
 * Pick which series get a direct end label: sorted by end value, a label is kept only if it
 * sits ≥ minGap px from every label already placed (priority order decides who wins).
 */
export function endLabels(ends: { key: string; value: number }[], domain: [number, number], plotHeight: number, priority: string[], minGap = 13): Set<string> {
  const px = (v: number) => ((domain[1] - v) / (domain[1] - domain[0] || 1)) * plotHeight;
  const ordered = [...ends].sort((a, b) => priority.indexOf(a.key) - priority.indexOf(b.key));
  const placed: number[] = [];
  const keep = new Set<string>();
  for (const e of ordered) {
    const y = px(e.value);
    if (placed.every((p) => Math.abs(p - y) >= minGap)) {
      placed.push(y);
      keep.add(e.key);
    }
  }
  return keep;
}
