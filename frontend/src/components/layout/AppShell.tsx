import { Suspense } from 'react';
import { Outlet } from 'react-router';
import { Loading } from '@/components/ui/States';
import { ApiBanner } from './ApiBanner';
import { SideNav } from './SideNav';
import { TopBar } from './TopBar';

export function AppShell() {
  return (
    <div className="flex min-h-screen bg-plane">
      <a href="#main" className="sr-only rounded bg-surface px-3 py-2 focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50">
        Skip to content
      </a>
      <SideNav />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <ApiBanner />
        <main id="main" tabIndex={-1} className="min-w-0 flex-1 px-5 pt-4 pb-10 focus:outline-none">
          <Suspense fallback={<Loading label="Loading page" height={320} />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  );
}
