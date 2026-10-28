import { useState, type ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { inkOn } from '@/lib/palette';

export interface HeatCell {
  value: number | null;
  /** Main label inside the cell (omit to leave it blank). */
  text?: string;
  /** Second, smaller line. */
  sub?: string;
  /** Annotation ring (e.g. recommended arm). */
  outline?: boolean;
  /** Corner marker (e.g. historical arm). */
  marker?: string | null;
  /** Not applicable (structural zero / ineligible): no fill, muted text. */
  na?: boolean;
  /** With `na`: hatched background (e.g. an arm the guardrails exclude). */
  hatched?: boolean;
  ariaLabel: string;
  tooltip?: ReactNode;
}

/** Outline around a band of columns spanning every row (e.g. the rate corridor). */
export interface ColumnBand {
  from: number;
  to: number;
  label: string;
}

interface Hover {
  r: number;
  c: number;
  left: number;
  top: number;
}

/**
 * Grid heatmap built from buttons: every cell is focusable, hover and focus show the same
 * tooltip, text colour is picked for contrast against the fill.
 */
export function Heatmap({
  rows,
  cols,
  cell,
  color,
  rowTitle,
  colTitle,
  cellHeight = 36,
  onSelect,
  selected,
  label,
  rowHeaderWidth = '2.5rem',
  band = null,
}: {
  rows: { key: string; label: ReactNode }[];
  cols: { key: string; label: ReactNode }[];
  cell: (r: number, c: number) => HeatCell;
  color: (v: number) => string;
  rowTitle: string;
  colTitle: string;
  cellHeight?: number;
  onSelect?: (r: number, c: number) => void;
  selected?: [number, number] | null;
  label: string;
  rowHeaderWidth?: string;
  band?: ColumnBand | null;
}) {
  const [hover, setHover] = useState<Hover | null>(null);
  const show = (r: number, c: number, el: HTMLElement) => setHover({ r, c, left: el.offsetLeft + el.offsetWidth / 2, top: el.offsetTop + el.offsetHeight });
  const tip = hover ? cell(hover.r, hover.c).tooltip : null;
  return (
    <div className="relative" onPointerLeave={() => setHover(null)}>
      <div className={cn('text-center text-2xs text-muted', band ? 'mb-[18px]' : 'mb-1')} style={{ paddingLeft: `calc(${rowHeaderWidth} + 1rem)` }}>
        {colTitle}
      </div>
      <div className="flex">
        <div className="flex w-4 items-center justify-center">
          <span className="-rotate-90 text-2xs whitespace-nowrap text-muted">{rowTitle}</span>
        </div>
        <div
          role="grid"
          aria-label={label}
          className="grid flex-1 gap-[2px]"
          style={{ gridTemplateColumns: `${rowHeaderWidth} repeat(${cols.length}, minmax(2.6rem, 1fr))` }}
        >
          <div role="row" className="contents">
            <span role="columnheader" style={{ gridRow: 1, gridColumn: 1 }} />
            {cols.map((c, ci) => (
              <span key={c.key} role="columnheader" className="num pb-0.5 text-center text-2xs text-axis" style={{ gridRow: 1, gridColumn: ci + 2 }}>
                {c.label}
              </span>
            ))}
          </div>
          {band && (
            <div
              aria-hidden="true"
              className="pointer-events-none relative z-10 -m-[3px] rounded-[5px] border-[1.5px] border-[var(--text-secondary)]"
              style={{ gridRow: `1 / ${rows.length + 2}`, gridColumn: `${band.from + 2} / ${band.to + 3}` }}
            >
              <span className="absolute -top-[16px] left-1/2 -translate-x-1/2 rounded-[3px] bg-[var(--surface-1)] px-1 text-[10px] leading-[14px] whitespace-nowrap text-ink-2">{band.label}</span>
            </div>
          )}
          {rows.map((row, r) => (
            <div role="row" key={row.key} className="contents">
              <span role="rowheader" className="num flex items-center justify-end pr-1.5 text-2xs text-axis" style={{ gridRow: r + 2, gridColumn: 1 }}>
                {row.label}
              </span>
              {cols.map((col, c) => {
                const d = cell(r, c);
                const fill = d.na || d.value === null ? undefined : color(d.value);
                const isSel = selected?.[0] === r && selected?.[1] === c;
                return (
                  <button
                    key={col.key}
                    type="button"
                    role="gridcell"
                    aria-label={d.ariaLabel}
                    aria-selected={onSelect ? isSel : undefined}
                    onPointerEnter={(e) => show(r, c, e.currentTarget)}
                    onFocus={(e) => show(r, c, e.currentTarget)}
                    onBlur={() => setHover(null)}
                    onClick={onSelect ? () => onSelect(r, c) : undefined}
                    className={cn(
                      'relative flex flex-col items-center justify-center rounded-[3px] leading-tight transition-[filter] hover:brightness-110',
                      d.na || d.value === null ? (d.hatched ? 'hatch-na text-muted' : 'bg-surface-2 text-muted') : '',
                      d.outline && 'ring-2 ring-[var(--text-primary)] ring-offset-1 ring-offset-[var(--surface-1)]',
                      isSel && !d.outline && 'ring-2 ring-[var(--focus)]',
                      !onSelect && 'cursor-default',
                    )}
                    style={{ gridRow: r + 2, gridColumn: c + 2, height: cellHeight, background: fill, color: fill ? inkOn(fill) : undefined }}
                  >
                    {d.text !== undefined && <span className="num text-[11px] font-medium">{d.text}</span>}
                    {d.sub && <span className="num text-[9.5px] opacity-80">{d.sub}</span>}
                    {d.marker && (
                      <span
                        aria-hidden="true"
                        className="absolute top-0.5 right-0.5 rounded-[2px] bg-[var(--surface-1)] px-[3px] text-[8.5px] leading-[11px] font-bold text-ink ring-1 ring-line"
                      >
                        {d.marker}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      {hover && tip && (
        <div
          className="pointer-events-none absolute z-20 w-max max-w-64 -translate-x-1/2 translate-y-1.5 rounded-md border border-line bg-[var(--tooltip-bg)] px-2.5 py-2 text-xs shadow-pop"
          style={{ left: hover.left + 18, top: hover.top + 14 }}
        >
          {tip}
        </div>
      )}
    </div>
  );
}
