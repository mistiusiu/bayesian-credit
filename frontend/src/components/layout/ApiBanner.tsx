import { API_BASE, MOCK_MODE, useDatasets } from '@/api/client';
import { Button } from '@/components/ui/Button';
import { Icon } from '@/components/ui/Icon';

/** App-level notice when the API cannot be reached (list views would otherwise look merely empty). */
export function ApiBanner() {
  const q = useDatasets();
  if (!q.isError) return null;
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 border-b border-critical/40 bg-critical/10 px-5 py-2 text-xs text-ink">
      <Icon name="alert" size={14} style={{ color: 'var(--status-critical)' }} />
      <span className="min-w-0 flex-1">
        Cannot load data from <code>{API_BASE}</code>: {q.error.message}.{' '}
        {MOCK_MODE ? 'The mock API failed; reload the page.' : 'Start the Django API on :8010 (or set VITE_API_PROXY_TARGET), or run `pnpm dev` for the mock API.'}
      </span>
      <Button size="sm" icon="refresh" onClick={() => void q.refetch()} loading={q.isFetching}>
        Retry
      </Button>
    </div>
  );
}
