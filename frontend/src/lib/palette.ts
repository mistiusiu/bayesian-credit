/**
 * Colour roles. Categorical identity uses the validated palette slots in fixed order and
 * colour follows the entity (a policy keeps its hue whatever is filtered). Sequential =
 * the documented blue ramp (anchor flips in dark mode). Diverging = red ↔ neutral ↔ blue.
 * Status colours are reserved for state and always ship with an icon + label.
 */
import type { BreakerState, CreditState, KnownPolicy, PolicyName, SentinelLevel } from '@/api/types';
import type { Theme } from './theme';

/** Validated with the dataviz validator (adjacent pairs, light and dark). */
export const POLICY_SLOT: Record<KnownPolicy, number> = {
  thompson: 1,
  ucb1: 2,
  epsilon_greedy: 3,
  historical: 4,
  random: 5,
  greedy: 6,
  thompson_se: 7,
};
/** Colour follows the policy's base name (parametrised specs share it); unknown specs use slot 8. */
export const policyColor = (p: PolicyName) => {
  const base = p.split('(')[0].trim();
  const slot = (POLICY_SLOT as Record<string, number | undefined>)[base] ?? 8;
  return `var(--series-${slot})`;
};

/** Stack order loss → delinquent → current → recovered validated adjacent in both modes. */
export const STATE_SLOT: Record<CreditState, number> = { current: 1, delinquent: 4, recovered: 3, loss: 7 };
export const stateColor = (s: CreditState) => `var(--series-${STATE_SLOT[s]})`;

export const seriesColor = (slot: number) => `var(--series-${slot})`;

export type Tone = 'good' | 'warning' | 'serious' | 'critical' | 'neutral' | 'info';

export const TONE_COLOR: Record<Tone, string> = {
  good: 'var(--status-good)',
  warning: 'var(--status-warning)',
  serious: 'var(--status-serious)',
  critical: 'var(--status-critical)',
  neutral: 'var(--text-axis)',
  info: 'var(--accent-strong)',
};

export const LEVEL_TONE: Record<SentinelLevel, Tone> = { normal: 'good', watch: 'warning', breaker: 'critical' };
export const BREAKER_TONE: Record<BreakerState, Tone> = { closed: 'good', half_open: 'warning', open: 'critical' };
export const BREAKER_LABEL: Record<BreakerState, string> = { closed: 'Closed', half_open: 'Half-open', open: 'Open' };

/** Tone for any lifecycle status string (jobs, datasets, fits, simulations, reviews). */
export function statusTone(status: string): Tone {
  switch (status) {
    case 'ready':
    case 'succeeded':
    case 'completed':
    case 'acknowledged':
      return 'good';
    case 'queued':
    case 'running':
      return 'info';
    case 'awaiting_review':
    case 'pending':
      return 'warning';
    case 'failed':
    case 'interrupted':
      return 'critical';
    default:
      return 'neutral';
  }
}

// ─────────────────────────────────────────────── sequential & diverging scales
/** Documented blue ramp, steps 100 → 700. */
export const SEQ_BLUE = [
  '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
  '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
];

const DIV = {
  light: { neg: '#e34948', mid: '#f0efec', pos: '#2a78d6' },
  dark: { neg: '#e66767', mid: '#383835', pos: '#3987e5' },
};

type Lab = [number, number, number];

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as [number, number, number];
}
const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toGamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function rgbToOklab([r, g, b]: [number, number, number]): Lab {
  const [lr, lg, lb] = [toLinear(r), toLinear(g), toLinear(b)];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToHex([L, a, b]: Lab): string {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return `#${rgb
    .map((c) => Math.round(Math.min(1, Math.max(0, toGamma(c))) * 255).toString(16).padStart(2, '0'))
    .join('')}`;
}

export function mixOklab(a: string, b: string, t: number): string {
  const A = rgbToOklab(hexToRgb(a));
  const B = rgbToOklab(hexToRgb(b));
  return oklabToHex([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t]);
}

function ramp(stops: string[], t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return mixOklab(stops[i], stops[i + 1], x - i);
}

/** Dark mode: anchor flipped (700 → 100) with a near-surface step so ~0 recedes into the card. */
const SEQ_DARK = ['#1d2430', ...[...SEQ_BLUE].reverse().slice(1)];

/** Magnitude in [0, 1] → documented blue ramp; near-zero recedes toward the surface in both modes. */
export function seqColor(t: number, theme: Theme): string {
  return ramp(theme === 'dark' ? SEQ_DARK : SEQ_BLUE, t);
}

/** Stops for a sequential scale legend in the current theme. */
export function seqStops(theme: Theme): string[] {
  return (theme === 'dark' ? SEQ_DARK : SEQ_BLUE).filter((_, i) => i % 3 === 0);
}

/** Polarity in [−1, 1] → red (negative) · neutral gray (0) · blue (positive). */
export function divColor(v: number, theme: Theme): string {
  const p = DIV[theme];
  const t = Math.min(1, Math.abs(v)) ** 0.8;
  return mixOklab(p.mid, v < 0 ? p.neg : p.pos, t);
}

export function divStops(theme: Theme): string[] {
  return [-1, -0.5, 0, 0.5, 1].map((v) => divColor(v, theme));
}

function luminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex).map(toLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Ink colour (near-black or white) with the higher contrast on a given fill. */
export function inkOn(hex: string): string {
  const L = luminance(hex);
  const dark = (L + 0.05) / (0.0033 + 0.05);
  const light = 1.05 / (L + 0.05);
  return dark >= light ? '#0b0b0b' : '#ffffff';
}
