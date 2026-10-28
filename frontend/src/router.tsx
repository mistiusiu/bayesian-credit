import { createBrowserRouter } from 'react-router';
import { AppShell } from '@/components/layout/AppShell';
import { RouteError } from '@/components/layout/RouteError';

/** Pages are code-split; the shell (nav, selectors) loads first. */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    errorElement: <RouteError />,
    hydrateFallbackElement: <div className="min-h-screen bg-plane" />,
    children: [
      { index: true, lazy: async () => ({ Component: (await import('@/pages/overview/OverviewPage')).OverviewPage }) },
      { path: 'data', lazy: async () => ({ Component: (await import('@/pages/data/DataPage')).DataPage }) },
      { path: 'risk', lazy: async () => ({ Component: (await import('@/pages/risk/RiskPage')).RiskPage }) },
      { path: 'mctm', lazy: async () => ({ Component: (await import('@/pages/mctm/MctmPage')).MctmPage }) },
      { path: 'bandit', lazy: async () => ({ Component: (await import('@/pages/bandit/BanditPage')).BanditPage }) },
      { path: 'shadow', lazy: async () => ({ Component: (await import('@/pages/shadow/ShadowPage')).ShadowPage }) },
      { path: 'sentinel', lazy: async () => ({ Component: (await import('@/pages/sentinel/SentinelPage')).SentinelPage }) },
      { path: 'evaluation', lazy: async () => ({ Component: (await import('@/pages/evaluation/EvaluationPage')).EvaluationPage }) },
      { path: '*', lazy: async () => ({ Component: (await import('@/pages/NotFound')).NotFound }) },
    ],
  },
]);
