import { NavLink, useSearchParams } from 'react-router';
import { MOCK_MODE, useActiveJobs, useHealth, useReviews } from '@/api/client';
import type { JobKind } from '@/api/types';
import { StatusGlyph } from '@/components/ui/Badge';
import { Icon, type IconName } from '@/components/ui/Icon';
import { cn } from '@/lib/cn';
import { selectionSearch } from '@/lib/selection';

const NAV: { to: string; label: string; icon: IconName; jobs?: JobKind[]; reviews?: boolean }[] = [
  { to: '/', label: 'Overview', icon: 'overview' },
  { to: '/data', label: 'Data', icon: 'data', jobs: ['ingest'] },
  { to: '/risk', label: 'Risk models', icon: 'risk', jobs: ['fit'] },
  { to: '/mctm', label: 'MCTM', icon: 'mctm' },
  { to: '/bandit', label: 'Bandit lab', icon: 'bandit', jobs: ['simulation'] },
  { to: '/shadow', label: 'Shadow engine', icon: 'shadow', jobs: ['shadow_portfolio'] },
  { to: '/sentinel', label: 'Sentinel', icon: 'sentinel', reviews: true },
  { to: '/evaluation', label: 'Evaluation', icon: 'evaluation' },
];

export function SideNav() {
  const [params] = useSearchParams();
  const search = selectionSearch(params);
  const jobs = useActiveJobs().data?.results ?? [];
  const pending = useReviews({ status: 'pending' }).data?.count ?? 0;
  const health = useHealth();
  return (
    <nav aria-label="Primary" className="sticky top-0 flex h-screen w-14 shrink-0 flex-col border-r border-line bg-surface xl:w-52">
      <div className="flex h-12 items-center gap-2 border-b border-line px-3.5">
        <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true" className="shrink-0">
          <circle cx="16" cy="16" r="9" fill="none" stroke="var(--series-1)" strokeWidth="2.5" />
          <circle cx="16" cy="7" r="3" fill="var(--text-primary)" />
          <circle cx="23.8" cy="20.5" r="3" fill="var(--text-primary)" />
          <circle cx="8.2" cy="20.5" r="3" fill="var(--text-primary)" />
        </svg>
        <div className="hidden min-w-0 xl:block">
          <div className="truncate text-[13px] font-semibold text-ink">Bayesian Credit</div>
          <div className="truncate text-2xs text-muted">shadow lending console</div>
        </div>
      </div>
      <ul className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {NAV.map((item) => {
          const running = item.jobs ? jobs.filter((j) => item.jobs?.includes(j.kind)).length : 0;
          const reviews = item.reviews ? pending : 0;
          const status = reviews ? `${reviews} pending review${reviews > 1 ? 's' : ''}` : running ? `${running} running job${running > 1 ? 's' : ''}` : '';
          return (
            <li key={item.to}>
              <NavLink
                to={{ pathname: item.to, search }}
                end={item.to === '/'}
                title={item.label}
                aria-label={status ? `${item.label} (${status})` : undefined}
                className={({ isActive }) =>
                  cn(
                    'relative flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] transition-colors',
                    isActive ? 'bg-surface-3 font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                  )
                }
              >
                <Icon name={item.icon} size={16} className="shrink-0" />
                <span className="hidden flex-1 truncate xl:inline">{item.label}</span>
                {(running > 0 || reviews > 0) && (
                  <span className="absolute top-1.5 right-1.5 inline-flex items-center gap-1 xl:static" title={status}>
                    <StatusGlyph tone={reviews ? 'warning' : 'info'} pulse={!reviews} />
                    {reviews > 0 && <span className="num hidden text-2xs text-ink-2 xl:inline">{reviews}</span>}
                  </span>
                )}
              </NavLink>
            </li>
          );
        })}
      </ul>
      <div className="border-t border-line p-3 text-2xs text-muted">
        <div className="flex items-center gap-1.5" title={MOCK_MODE ? 'Serving seeded mock data in the browser' : 'Talking to the Django API via /api'}>
          <StatusGlyph tone={health.isError ? 'critical' : health.data ? 'good' : 'neutral'} />
          <span className="hidden xl:inline">{MOCK_MODE ? 'Mock API' : 'Live API'}</span>
          <span className="num hidden xl:inline">{health.data ? `v${health.data.version}` : health.isError ? 'unreachable' : ''}</span>
        </div>
      </div>
    </nav>
  );
}
