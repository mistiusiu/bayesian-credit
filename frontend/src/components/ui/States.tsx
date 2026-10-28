import type { ReactNode } from 'react';
import type { UseQueryResult } from '@tanstack/react-query';
import { ApiError, MOCK_MODE } from '@/api/client';
import { cn } from '@/lib/cn';
import { Button } from './Button';
import { Icon } from './Icon';

export function Loading({ label = 'Loading', className, height = 120 }: { label?: string; className?: string; height?: number }) {
  return (
    <div role="status" aria-live="polite" className={cn('flex items-center justify-center gap-2 text-xs text-muted', className)} style={{ minHeight: height }}>
      <span className="size-3.5 animate-spin rounded-full border-2 border-current border-r-transparent" aria-hidden="true" />
      {label}…
    </div>
  );
}

export function Empty({ title, children, className, height = 120 }: { title: string; children?: ReactNode; className?: string; height?: number }) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-1 px-4 text-center', className)} style={{ minHeight: height }}>
      <p className="text-[13px] font-medium text-ink-2">{title}</p>
      {children && <div className="max-w-md text-xs text-muted">{children}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const status = error instanceof ApiError ? error.status : null;
  const msg = error instanceof Error ? error.message : 'Unexpected error';
  return (
    <div role="alert" className={cn('flex flex-col items-center justify-center gap-2 px-4 py-6 text-center', className)}>
      <p className="inline-flex items-center gap-1.5 text-[13px] font-medium text-ink">
        <Icon name="alert" size={14} style={{ color: 'var(--status-critical)' }} />
        {status === 409 ? 'Not ready yet' : status === 0 ? 'API unreachable' : 'Request failed'}
      </p>
      <p className="max-w-md text-xs text-muted">
        {msg}
        {status === 0 && !MOCK_MODE && ' — start the Django API on :8010 or run `pnpm dev` (mock mode).'}
      </p>
      {onRetry && (
        <Button size="sm" icon="refresh" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

/**
 * Loading / error / empty / data switch for a query. While refetching with data on
 * screen the previous frame stays and dims (no skeleton flash).
 */
export function QueryState<T>({
  query,
  children,
  isEmpty,
  empty,
  loadingLabel,
  height = 160,
}: {
  query: UseQueryResult<T>;
  children: (data: T) => ReactNode;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  loadingLabel?: string;
  height?: number;
}) {
  if (query.isPending) return <Loading label={loadingLabel} height={height} />;
  if (query.isError && !query.data) return <ErrorState error={query.error} onRetry={() => void query.refetch()} />;
  const data = query.data as T;
  if (isEmpty?.(data)) return <>{empty ?? <Empty title="Nothing to show" height={height} />}</>;
  return <div className={cn('transition-opacity', query.isPlaceholderData && 'opacity-60')}>{children(data)}</div>;
}
