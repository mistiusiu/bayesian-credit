import { useState, type KeyboardEvent, type ReactNode } from 'react';
import { cn } from '@/lib/cn';

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  align?: 'left' | 'right' | 'center';
  className?: string;
  /** Enables click-to-sort on this column. */
  sortValue?: (row: T) => number | string | null;
  title?: string;
}

/**
 * Dense data table: tabular numerals, sticky header, optional sort, keyboard-selectable
 * rows (Enter / Space) when `onRowClick` is given.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  selectedKey,
  empty = 'No rows',
  maxHeight,
  caption,
  dense = true,
  initialSort,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string | number;
  onRowClick?: (row: T) => void;
  selectedKey?: string | number | null;
  empty?: ReactNode;
  maxHeight?: number | string;
  caption?: string;
  dense?: boolean;
  initialSort?: { key: string; dir: 'asc' | 'desc' };
}) {
  const [sort, setSort] = useState(initialSort ?? null);
  const sortCol = sort ? columns.find((c) => c.key === sort.key) : undefined;
  const sorted =
    sortCol?.sortValue && sort
      ? [...rows].sort((a, b) => {
          const va = sortCol.sortValue!(a);
          const vb = sortCol.sortValue!(b);
          if (va === vb) return 0;
          if (va === null) return 1;
          if (vb === null) return -1;
          const c = va < vb ? -1 : 1;
          return sort.dir === 'asc' ? c : -c;
        })
      : rows;
  const onKey = (e: KeyboardEvent, row: T) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onRowClick?.(row);
    }
  };
  const align = (a?: string) => (a === 'right' ? 'text-right' : a === 'center' ? 'text-center' : 'text-left');
  return (
    <div className="cell-scroll overflow-auto rounded-md border border-line" style={{ maxHeight }}>
      <table className="num w-full text-[12.5px]">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead className="sticky top-0 z-10 bg-surface-2">
          <tr>
            {columns.map((c) => {
              const active = sort?.key === c.key;
              return (
                <th
                  key={c.key}
                  scope="col"
                  title={c.title}
                  aria-sort={active ? (sort?.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                  className={cn('border-b border-line px-2.5 py-1.5 font-medium whitespace-nowrap text-muted', align(c.align), c.className)}
                >
                  {c.sortValue ? (
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-ink"
                      onClick={() => setSort({ key: c.key, dir: active && sort?.dir === 'desc' ? 'asc' : 'desc' })}
                    >
                      {c.header}
                      <span aria-hidden="true" className="text-axis">
                        {active ? (sort?.dir === 'asc' ? '▲' : '▼') : '↕'}
                      </span>
                    </button>
                  ) : (
                    c.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-3 py-6 text-center text-muted">
                {empty}
              </td>
            </tr>
          )}
          {sorted.map((row) => {
            const key = rowKey(row);
            const selected = selectedKey !== undefined && selectedKey === key;
            return (
              <tr
                key={key}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                onKeyDown={onRowClick ? (e) => onKey(e, row) : undefined}
                tabIndex={onRowClick ? 0 : undefined}
                aria-selected={onRowClick ? selected : undefined}
                className={cn(
                  'border-b border-line last:border-b-0',
                  onRowClick && 'cursor-pointer hover:bg-surface-2 focus-visible:bg-surface-2',
                  selected && 'bg-accent-wash hover:bg-accent-wash',
                )}
              >
                {columns.map((c) => (
                  <td key={c.key} className={cn('px-2.5 whitespace-nowrap text-ink', dense ? 'py-1' : 'py-2', align(c.align), c.className)}>
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
