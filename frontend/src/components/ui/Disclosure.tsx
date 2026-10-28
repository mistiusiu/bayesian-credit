import { useId, useState, type ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { Icon } from './Icon';

export function Disclosure({ title, subtitle, defaultOpen = false, children, className }: { title: ReactNode; subtitle?: ReactNode; defaultOpen?: boolean; children: ReactNode; className?: string }) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <section className={cn('rounded-lg border border-line bg-surface', className)}>
      <h2>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center gap-2 px-4 py-2.5 text-left hover:bg-surface-2"
        >
          <Icon name={open ? 'chevronDown' : 'chevronRight'} size={14} className="text-muted" />
          <span className="text-[13px] font-semibold text-ink">{title}</span>
          {subtitle && <span className="ml-2 truncate text-xs text-muted">{subtitle}</span>}
        </button>
      </h2>
      {open && (
        <div id={id} className="border-t border-line p-4">
          {children}
        </div>
      )}
    </section>
  );
}
