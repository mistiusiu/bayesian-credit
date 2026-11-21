import { useRef, type KeyboardEvent, type ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { panelId, tabId } from './tabIds';

export interface TabDef<T extends string> {
  id: T;
  label: ReactNode;
  badge?: ReactNode;
}


/** WAI-ARIA tablist (arrow keys / Home / End move focus and selection). */
export function Tabs<T extends string>({
  scope,
  tabs,
  value,
  onChange,
  label,
  className,
}: {
  scope: string;
  tabs: TabDef<T>[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, i: number) => {
    let next = -1;
    if (e.key === 'ArrowRight') next = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = tabs.length - 1;
    if (next >= 0) {
      e.preventDefault();
      onChange(tabs[next].id);
      refs.current[next]?.focus();
    }
  };
  return (
    <div role="tablist" aria-label={label} className={cn('flex gap-0.5 overflow-x-auto overflow-y-hidden shadow-[inset_0_-1px_0_var(--border)]', className)}>
      {tabs.map((t, i) => {
        const active = t.id === value;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            id={tabId(scope, t.id)}
            role="tab"
            type="button"
            aria-selected={active}
            aria-controls={panelId(scope, t.id)}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => onKey(e, i)}
            className={cn(
              'inline-flex h-10 shrink-0 items-center gap-1.5 border-b-[3px] px-3.5 text-[13px] whitespace-nowrap transition-colors',
              active ? 'border-brand font-semibold text-ink' : 'border-transparent font-medium text-muted hover:border-line-strong hover:text-ink',
            )}
          >
            {t.label}
            {t.badge}
          </button>
        );
      })}
    </div>
  );
}

export function TabPanel({ scope, id, children, className }: { scope: string; id: string; children: ReactNode; className?: string }) {
  return (
    <div role="tabpanel" id={panelId(scope, id)} aria-labelledby={tabId(scope, id)} tabIndex={0} className={cn('focus-visible:outline-offset-4', className)}>
      {children}
    </div>
  );
}
