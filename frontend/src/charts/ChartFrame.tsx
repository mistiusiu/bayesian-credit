import { useId, useState, type ReactNode } from 'react';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { Icon } from '@/components/ui/Icon';
import { cn } from '@/lib/cn';

export interface TableSpec {
  columns: { key: string; label: string; align?: 'left' | 'right' }[];
  rows: Record<string, ReactNode>[];
}

/**
 * Chart container: title, legend row, y-unit label above the plot, x label below, a
 * one-line caption, and a chart ⇄ table toggle (the accessible twin of every chart).
 */
export function ChartFrame({
  title,
  caption,
  legend,
  actions,
  table,
  yLabel,
  xLabel,
  children,
  className,
  dimmed = false,
}: {
  title: ReactNode;
  caption: ReactNode;
  legend?: ReactNode;
  actions?: ReactNode;
  table?: TableSpec;
  yLabel?: ReactNode;
  xLabel?: ReactNode;
  children: ReactNode;
  className?: string;
  dimmed?: boolean;
}) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const id = useId();
  const columns: Column<Record<string, ReactNode> & { __i: number }>[] = (table?.columns ?? []).map((c) => ({
    key: c.key,
    header: c.label,
    align: c.align,
    render: (r) => r[c.key],
  }));
  return (
    <figure aria-labelledby={`${id}-t`} className={cn('m-0 flex min-w-0 flex-col rounded-lg border border-line bg-surface', className)}>
      <div className="flex items-start justify-between gap-3 px-4 pt-3">
        <h3 id={`${id}-t`} className="text-[13px] font-semibold text-ink">
          {title}
        </h3>
        <div className="flex shrink-0 items-center gap-2">
          {actions}
          {table && (
            <div role="group" aria-label="View" className="inline-flex rounded-md border border-line p-0.5">
              {(['chart', 'table'] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  aria-pressed={view === v}
                  aria-label={v === 'chart' ? 'Show chart' : 'Show data table'}
                  title={v === 'chart' ? 'Chart' : 'Data table'}
                  onClick={() => setView(v)}
                  className={cn('rounded-[4px] p-1', view === v ? 'bg-surface-3 text-ink' : 'text-muted hover:text-ink')}
                >
                  <Icon name={v} size={13} />
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      {legend && <div className="px-4 pt-1.5">{legend}</div>}
      <div className={cn('flex-1 px-2 pt-2 transition-opacity', dimmed && 'opacity-60')}>
        {view === 'chart' || !table ? (
          <>
            {yLabel && <div className="px-2 pb-0.5 text-2xs text-muted">{yLabel}</div>}
            {children}
            {xLabel && <div className="pt-0.5 text-center text-2xs text-muted">{xLabel}</div>}
          </>
        ) : (
          <div className="px-2">
            <DataTable columns={columns} rows={table.rows.map((r, i) => ({ ...r, __i: i }))} rowKey={(r) => r.__i} maxHeight={320} />
          </div>
        )}
      </div>
      <figcaption className="px-4 pt-1.5 pb-3 text-xs text-muted">{caption}</figcaption>
    </figure>
  );
}
