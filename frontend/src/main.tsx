import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { RouterProvider } from 'react-router/dom';
import '@fontsource/red-hat-display/500.css';
import '@fontsource/red-hat-display/700.css';
import '@fontsource/red-hat-mono/400.css';
import '@fontsource/red-hat-mono/500.css';
import '@fontsource/red-hat-text/400.css';
import '@fontsource/red-hat-text/500.css';
import '@fontsource/red-hat-text/600.css';
import '@fontsource/red-hat-text/700.css';
import { ApiError } from '@/api/client';
import { ThemeProvider } from '@/components/ThemeProvider';
import { router } from './router';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      refetchOnWindowFocus: false,
      // 4xx are answers (not ready, not found, validation) — do not retry them
      retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
    },
  },
});

const root = document.getElementById('root');
if (!root) throw new Error('#root missing');

createRoot(root).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
);
