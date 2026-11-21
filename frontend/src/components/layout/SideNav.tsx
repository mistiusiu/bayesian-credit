import { Fragment } from 'react';
import { NavLink, useSearchParams } from 'react-router';
import { MOCK_MODE, useActiveJobs, useHealth, useReviews } from '@/api/client';
import type { JobKind } from '@/api/types';
import { StatusGlyph } from '@/components/ui/Badge';
import { Icon, type IconName } from '@/components/ui/Icon';
import { cn } from '@/lib/cn';
import { selectionSearch } from '@/lib/selection';

const NAV: { to: string; label: string; icon: IconName; section?: string; jobs?: JobKind[]; reviews?: boolean }[] = [
  { to: '/', label: 'Overview', icon: 'overview' },
  { to: '/data', label: 'Data', icon: 'data', section: 'Pipeline', jobs: ['ingest'] },
  { to: '/risk', label: 'Risk models', icon: 'risk', jobs: ['fit'] },
  { to: '/mctm', label: 'MCTM', icon: 'mctm' },
  { to: '/bandit', label: 'Bandit lab', icon: 'bandit', jobs: ['simulation'] },
  { to: '/shadow', label: 'Shadow Lending Engine', icon: 'shadow', section: 'Decisions', jobs: ['shadow_portfolio'] },
  { to: '/sentinel', label: 'Sentinel', icon: 'sentinel', section: 'Governance', reviews: true },
  { to: '/evaluation', label: 'Evaluation', icon: 'evaluation' },
];

export function SideNav() {
  const [params] = useSearchParams();
  const search = selectionSearch(params);
  const jobs = useActiveJobs().data?.results ?? [];
  const pending = useReviews({ status: 'pending' }).data?.count ?? 0;
  const health = useHealth();
  const temporal = health.data?.jobs?.temporal ?? null;
  return (
    // data-theme="dark": the nav is near-black chrome in both themes (tokens re-resolve in this subtree).
    <nav aria-label="Primary" data-theme="dark" className="sticky top-0 flex h-screen w-14 shrink-0 flex-col border-r border-line bg-chrome xl:w-60">
      <div className="flex h-14 items-center gap-2.5 border-b border-line px-4">
        <svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true" className="shrink-0">
          <circle cx="16" cy="16" r="9" fill="none" stroke="var(--brand)" strokeWidth="2.75" />
          <circle cx="16" cy="7" r="3" fill="var(--text-primary)" />
          <circle cx="23.8" cy="20.5" r="3" fill="var(--text-primary)" />
          <circle cx="8.2" cy="20.5" r="3" fill="var(--text-primary)" />
        </svg>
        <div className="hidden min-w-0 xl:block">
          <div className="truncate font-display text-[15px] leading-5 font-bold tracking-tight text-ink">Bayesian Credit</div>
          <div className="truncate text-2xs font-medium text-muted">Shadow Lending Console</div>
        </div>
      </div>
      <ul className="flex-1 overflow-y-auto py-2">
        {NAV.map((item) => {
          const running = item.jobs ? jobs.filter((j) => item.jobs?.includes(j.kind)).length : 0;
          const reviews = item.reviews ? pending : 0;
          const status = reviews ? `${reviews} pending review${reviews > 1 ? 's' : ''}` : running ? `${running} running job${running > 1 ? 's' : ''}` : '';
          return (
            <Fragment key={item.to}>
              {item.section && (
                <li aria-hidden="true" className="mx-3 mt-3 mb-1 border-t border-line pt-3 xl:mx-4 xl:border-0 xl:pt-1">
                  <span className="eyebrow hidden xl:block">{item.section}</span>
                </li>
              )}
              <li>
                <NavLink
                  to={{ pathname: item.to, search }}
                  end={item.to === '/'}
                  title={item.label}
                  aria-label={status ? `${item.label} (${status})` : undefined}
                  className={({ isActive }) =>
                    cn(
                      'relative flex h-9 items-center gap-2.5 px-5 text-[13px] transition-colors xl:px-4',
                      'before:absolute before:inset-y-0 before:left-0 before:w-[3px]',
                      isActive ? 'bg-surface-2 font-semibold text-ink before:bg-brand' : 'text-ink-2 hover:bg-surface-1 hover:text-ink hover:before:bg-line-strong',
                    )
                  }
                >
                  <Icon name={item.icon} size={16} className="shrink-0" />
                  <span className="hidden flex-1 truncate xl:inline">{item.label}</span>
                  {(running > 0 || reviews > 0) && (
                    <span className="absolute top-1.5 right-2 inline-flex items-center gap-1 xl:static" title={status}>
                      <StatusGlyph tone={reviews ? 'warning' : 'info'} pulse={!reviews} />
                      {reviews > 0 && <span className="num hidden text-2xs text-ink-2 xl:inline">{reviews}</span>}
                    </span>
                  )}
                </NavLink>
              </li>
            </Fragment>
          );
        })}
      </ul>
      <div className="border-t border-line px-4 py-3 text-2xs text-muted">
        <div className="flex items-center gap-1.5" title={MOCK_MODE ? 'Serving seeded mock data in the browser' : 'Talking to the Django API via /api'}>
          <StatusGlyph tone={health.isError ? 'critical' : health.data ? 'good' : 'neutral'} />
          <span className="hidden xl:inline">{MOCK_MODE ? 'Mock API' : 'Live API'}</span>
          <span className="num hidden xl:inline">{health.data ? `v${health.data.version}` : health.isError ? 'unreachable' : ''}</span>
        </div>
        {temporal && (
          <a
            href={temporal.ui_url}
            target="_blank"
            rel="noreferrer"
            className="mt-1.5 flex items-center gap-1.5 hover:text-ink"
            title={
              temporal.connected
                ? `Temporal ${temporal.address} · queue ${temporal.task_queue}\n${temporal.workers.length ? temporal.workers.join('\n') : 'no worker polling: jobs stay queued'}`
                : `Temporal ${temporal.address} unreachable: ${temporal.error ?? ''}`
            }
          >
            <StatusGlyph tone={!temporal.connected ? 'critical' : temporal.workers.length ? 'good' : 'warning'} />
            <span className="hidden xl:inline">Temporal</span>
            <span className="num hidden xl:inline">
              {temporal.connected ? `${temporal.workers.length} worker${temporal.workers.length === 1 ? '' : 's'}` : 'down'}
            </span>
          </a>
        )}
      </div>
    </nav>
  );
}
