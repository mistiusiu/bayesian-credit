import { cn } from '@/lib/cn';

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  kind?: 'line' | 'rect' | 'dot' | 'band';
}

function Swatch({ color, kind = 'line' }: { color: string; kind?: LegendItem['kind'] }) {
  if (kind === 'rect') return <span aria-hidden="true" className="inline-block size-2.5 rounded-[2px]" style={{ background: color }} />;
  if (kind === 'dot') return <span aria-hidden="true" className="inline-block size-2.5 rounded-full ring-2 ring-[var(--surface-1)]" style={{ background: color }} />;
  if (kind === 'band') return <span aria-hidden="true" className="inline-block h-2.5 w-3.5 rounded-[2px] border border-line" style={{ background: color }} />;
  return <span aria-hidden="true" className="inline-block h-0.5 w-3.5 rounded-full" style={{ background: color }} />;
}

/** Legend mirrors the mark (line / rect / dot). With onToggle, items toggle series visibility. */
export function Legend({ items, hidden, onToggle, className }: { items: LegendItem[]; hidden?: Set<string>; onToggle?: (key: string) => void; className?: string }) {
  return (
    <ul className={cn('flex flex-wrap items-center gap-x-3.5 gap-y-1 text-xs text-ink-2', className)} aria-label="Legend">
      {items.map((it) => {
        const off = hidden?.has(it.key);
        const body = (
          <>
            <Swatch color={it.color} kind={it.kind} />
            <span className={cn(off && 'text-muted line-through')}>{it.label}</span>
          </>
        );
        return (
          <li key={it.key}>
            {onToggle ? (
              <button type="button" aria-pressed={!off} onClick={() => onToggle(it.key)} className="inline-flex items-center gap-1.5 rounded px-1 hover:bg-surface-2">
                {body}
              </button>
            ) : (
              <span className="inline-flex items-center gap-1.5">{body}</span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
