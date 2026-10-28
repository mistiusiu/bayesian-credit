/**
 * Shared number / date formatting. Every figure in the console goes through here so
 * units are consistent: money in USD, rates and probabilities as percentages of a
 * fraction, epochs as "Jun 2016". Missing values render as an em dash.
 */
const LOCALE = 'en-US';
const MINUS = '−';
export const DASH = '—';

type Maybe = number | null | undefined;

const isNum = (v: Maybe): v is number => typeof v === 'number' && Number.isFinite(v);

const nf = (opts: Intl.NumberFormatOptions) => new Intl.NumberFormat(LOCALE, opts);

const F = {
  int: nf({ maximumFractionDigits: 0 }),
  compact: nf({ notation: 'compact', maximumFractionDigits: 1 }),
  usd0: nf({ style: 'currency', currency: 'USD', maximumFractionDigits: 0 }),
  usd2: nf({ style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  usdCompact: nf({ style: 'currency', currency: 'USD', notation: 'compact', maximumFractionDigits: 1 }),
};

const pctCache = new Map<number, Intl.NumberFormat>();
const numCache = new Map<number, Intl.NumberFormat>();

function pctFormatter(digits: number): Intl.NumberFormat {
  let f = pctCache.get(digits);
  if (!f) {
    f = nf({ style: 'percent', minimumFractionDigits: digits, maximumFractionDigits: digits });
    pctCache.set(digits, f);
  }
  return f;
}

function numFormatter(digits: number): Intl.NumberFormat {
  let f = numCache.get(digits);
  if (!f) {
    f = nf({ minimumFractionDigits: digits, maximumFractionDigits: digits });
    numCache.set(digits, f);
  }
  return f;
}

/** Replace the ASCII hyphen-minus with a typographic minus sign. */
const typo = (s: string) => s.replace('-', MINUS);

function signed(v: number, body: (abs: number) => string): string {
  if (v === 0) return body(0);
  return (v > 0 ? '+' : MINUS) + body(Math.abs(v));
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export const fmt = {
  /** 1,284 */
  int: (v: Maybe) => (isNum(v) ? typo(F.int.format(v)) : DASH),
  /** 12.9K · 1.2M */
  compact: (v: Maybe) => (isNum(v) ? typo(F.compact.format(v)) : DASH),
  /** Fixed decimals: num(0.14532, 3) → 0.145 */
  num: (v: Maybe, digits = 2) => (isNum(v) ? typo(numFormatter(digits).format(v)) : DASH),
  /** $1,284 */
  usd: (v: Maybe) => (isNum(v) ? typo(F.usd0.format(v)) : DASH),
  /** $1,284.50 */
  usd2: (v: Maybe) => (isNum(v) ? typo(F.usd2.format(v)) : DASH),
  /** $4.2M */
  usdCompact: (v: Maybe) => (isNum(v) ? typo(F.usdCompact.format(v)) : DASH),
  /** +$1,284 / −$312 */
  usdSigned: (v: Maybe) => (isNum(v) ? signed(v, (a) => F.usd0.format(a)) : DASH),
  /** +$4.2M / −$1.1K */
  usdCompactSigned: (v: Maybe) => (isNum(v) ? signed(v, (a) => F.usdCompact.format(a)) : DASH),
  /** Fraction → percent: pct(0.1299) → 13.0% */
  pct: (v: Maybe, digits = 1) => (isNum(v) ? typo(pctFormatter(digits).format(v)) : DASH),
  /** Fraction → signed percent: +2.4% */
  pctSigned: (v: Maybe, digits = 1) => (isNum(v) ? signed(v, (a) => pctFormatter(digits).format(a)) : DASH),
  /** Signed plain number: +0.012 */
  numSigned: (v: Maybe, digits = 3) => (isNum(v) ? signed(v, (a) => numFormatter(digits).format(a)) : DASH),
  /** Probability to 3 decimals: 0.145 */
  prob: (v: Maybe) => (isNum(v) ? numFormatter(3).format(v) : DASH),
  /** Interest rate fraction → "13%" (whole-percent grid values) or "13.25%". */
  rate: (v: Maybe) => {
    if (!isNum(v)) return DASH;
    const p = Math.round(v * 10000) / 100;
    return Number.isInteger(p) ? `${p}%` : `${p.toFixed(2)}%`;
  },
  /** Amount → "$15k" (grid labels). */
  amountShort: (v: Maybe) => {
    if (!isNum(v)) return DASH;
    if (v >= 1000) return `$${(v / 1000).toLocaleString(LOCALE, { maximumFractionDigits: 1 })}k`;
    return `$${v.toFixed(0)}`;
  },
  /** "2016-06" → "Jun 2016" */
  epoch: (e: string | null | undefined) => {
    if (!e) return DASH;
    const [y, m] = e.split('-');
    const mi = Number(m) - 1;
    return MONTHS[mi] ? `${MONTHS[mi]} ${y}` : e;
  },
  /** "2016-06" → "Jun ’16" (axis ticks) */
  epochShort: (e: string | null | undefined) => {
    if (!e) return DASH;
    const [y, m] = e.split('-');
    const mi = Number(m) - 1;
    return MONTHS[mi] && y ? `${MONTHS[mi]} ’${y.slice(2)}` : e;
  },
  /** ISO timestamp → "Oct 16, 09:41:07" (local time). */
  dateTime: (iso: string | null | undefined) => {
    if (!iso) return DASH;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return DASH;
    return d.toLocaleString(LOCALE, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  },
  /** ISO timestamp → "Oct 16, 2026" */
  date: (iso: string | null | undefined) => {
    if (!iso) return DASH;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return DASH;
    return d.toLocaleDateString(LOCALE, { month: 'short', day: 'numeric', year: 'numeric' });
  },
  /** Seconds → "2m 14s" / "1h 03m" / "4.2s" */
  duration: (s: Maybe) => {
    if (!isNum(s)) return DASH;
    if (s < 10) return `${s.toFixed(1)}s`;
    if (s < 60) return `${Math.round(s)}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${String(Math.round(s % 60)).padStart(2, '0')}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${String(m % 60).padStart(2, '0')}m`;
  },
  /** Megabytes → "1.8 GB" / "640 MB" */
  megabytes: (mb: Maybe) => {
    if (!isNum(mb)) return DASH;
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
  },
  /** Arm label from its policy: "$15k · 13% · 36m" (decline for the null arm). */
  arm: (a: { amount: number; rate: number; term: number; is_null?: boolean } | null | undefined) => {
    if (!a) return DASH;
    if (a.is_null || a.amount <= 0 || a.term <= 0) return 'Decline';
    return `${fmt.amountShort(a.amount)} · ${fmt.rate(a.rate)} · ${a.term}m`;
  },
  /** Group key "B|debt_consolidation|RENT" → "B · debt consolidation · rent" */
  group: (key: string | null | undefined) => {
    if (!key) return DASH;
    return key
      .split('|')
      .map((p) => (p.length <= 2 ? p : p.replace(/_/g, ' ').toLowerCase()))
      .join(' · ');
  },
  /** snake_case → "Sentence case" */
  label: (s: string | null | undefined) => {
    if (!s) return DASH;
    const t = s.replace(/_/g, ' ');
    return t.charAt(0).toUpperCase() + t.slice(1);
  },
};

/** Format a value according to a backend unit string (report metrics, feature summaries). */
export function formatByUnit(v: Maybe, unit: string): string {
  switch (unit) {
    case 'usd':
      return Math.abs(v ?? 0) >= 100_000 ? fmt.usdCompact(v) : fmt.usd(v);
    case 'usd_per_decision':
      return `${fmt.usd(v)}/decision`;
    case 'fraction':
      return fmt.pct(v, 1);
    case 'pct':
      return isNum(v) ? `${fmt.num(v, 1)}%` : DASH;
    case 'count':
      return fmt.int(v);
    case 'seconds':
      return fmt.duration(v);
    case 'megabytes':
      return fmt.megabytes(v);
    case 'nats':
      return fmt.num(v, 3);
    case 'score': {
      // Brier scores (< 1) and FICO-like scores (hundreds) share this unit: pick decimals by magnitude.
      if (!isNum(v)) return DASH;
      const a = Math.abs(v);
      return fmt.num(v, a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.1 ? 3 : 4);
    }
    case 'years':
      return isNum(v) ? `${fmt.num(v, 1)} y` : DASH;
    case 'ratio':
      return fmt.num(v, isNum(v) && Math.abs(v) < 0.1 ? 4 : 3);
    default:
      return fmt.num(v, 3);
  }
}

/** Compact axis tick for a backend unit. */
export function tickByUnit(v: number, unit: string): string {
  switch (unit) {
    case 'usd':
      return fmt.usdCompact(v);
    case 'fraction':
      return fmt.pct(v, 0);
    case 'pct':
      return `${fmt.num(v, 0)}%`;
    default:
      return Math.abs(v) >= 1000 ? fmt.compact(v) : fmt.num(v, Number.isInteger(v) ? 0 : 1);
  }
}
