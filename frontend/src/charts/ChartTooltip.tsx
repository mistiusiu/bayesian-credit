import type { ReactNode } from 'react';

interface Entry {
  dataKey?: unknown;
  name?: unknown;
  value?: unknown;
  color?: string;
  stroke?: string;
  fill?: string;
  payload?: unknown;
}

export interface TooltipFormat {
  label?: (label: string | number) => ReactNode;
  value: (v: number, key: string) => string;
  name?: (key: string) => string;
  color?: (key: string) => string;
  /** Extra rows rendered under the series rows (from the hovered datum). */
  extra?: (datum: Record<string, unknown>) => ReactNode;
  /** Hide series whose key returns false. */
  include?: (key: string) => boolean;
}

const keyOf = (p: Entry): string => {
  const k = p.dataKey ?? p.name;
  return typeof k === 'string' || typeof k === 'number' ? String(k) : '';
};

/** One tooltip, every series; the value leads, the name follows; series keyed with a short line. */
export function ChartTooltip({ active, payload, label, format }: { active?: boolean; payload?: readonly Entry[]; label?: string | number; format: TooltipFormat }) {
  if (!active || !payload?.length) return null;
  const datum = (payload[0]?.payload ?? {}) as Record<string, unknown>;
  const rows = payload.filter((p) => {
    const key = keyOf(p);
    return typeof p.value === 'number' && Number.isFinite(p.value) && (!format.include || format.include(key));
  });
  return (
    <div className="pointer-events-none min-w-36 rounded-md border border-line bg-[var(--tooltip-bg)] px-2.5 py-2 text-xs shadow-pop">
      {label !== undefined && <div className="mb-1 font-medium text-ink-2">{format.label ? format.label(label) : String(label)}</div>}
      <ul className="space-y-0.5">
        {rows.map((p) => {
          const key = keyOf(p);
          const color = format.color?.(key) ?? p.stroke ?? p.color ?? p.fill ?? 'var(--text-axis)';
          return (
            <li key={key} className="flex items-center gap-2">
              <span aria-hidden="true" className="inline-block h-0.5 w-3 shrink-0 rounded-full" style={{ background: color }} />
              <span className="num font-semibold text-ink">{format.value(p.value as number, key)}</span>
              <span className="truncate text-muted">{format.name ? format.name(key) : key}</span>
            </li>
          );
        })}
      </ul>
      {format.extra && <div className="mt-1 border-t border-line pt-1 text-muted">{format.extra(datum)}</div>}
    </div>
  );
}
