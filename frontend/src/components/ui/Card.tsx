import { useId, type ReactNode } from 'react';
import { cn } from '@/lib/cn';

export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  const id = useId();
  return (
    <section aria-labelledby={title ? id : undefined} className={cn('min-w-0 rounded-lg border border-line bg-surface', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 border-b border-line px-4 py-2.5">
          <div className="min-w-0">
            {title && (
              <h2 id={id} className="text-[13px] font-semibold text-ink">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn('p-4', bodyClassName)}>{children}</div>
    </section>
  );
}

/** Section heading inside a page (between cards). */
export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mt-2 flex items-end justify-between gap-4">
      <h2 className="text-xs font-semibold tracking-wide text-muted uppercase">{children}</h2>
      {aside}
    </div>
  );
}
