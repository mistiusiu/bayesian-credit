import type { ReactNode } from 'react';

/** Page header: eyebrow kicker (with a short brand rule) · bold display title · one-paragraph description. */
export function PageHeader({ title, description, actions, eyebrow }: { title: string; description: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        {eyebrow && (
          <div className="eyebrow mb-1.5 flex items-center gap-2">
            <span aria-hidden="true" className="h-0.5 w-5 bg-brand" />
            {eyebrow}
          </div>
        )}
        <h1 className="font-display text-[28px] leading-9 font-bold tracking-tight text-ink">{title}</h1>
        <p className="mt-1.5 max-w-3xl text-[13px] leading-5 text-ink-2">{description}</p>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
