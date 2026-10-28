/**
 * Mock API entry point. `mockFetch` behaves like the Django backend for every endpoint in
 * docs/API.md, backed by seeded in-memory fixtures (see store.ts). Loaded lazily by the
 * API client only when VITE_API_MOCK=1, so production bundles never include it.
 */
import type { Route } from './router';
import { dataRoutes } from './routesData';
import { simRoutes } from './routesSim';
import { MockHttpError } from './store';

const ROUTES: Route[] = [...dataRoutes, ...simRoutes];

export interface MockResponse {
  status: number;
  body: unknown;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function mockFetch(method: string, path: string, query: URLSearchParams, body: unknown): Promise<MockResponse> {
  await sleep(70 + Math.random() * 180);
  const clean = path.endsWith('/') ? path : `${path}/`;
  for (const [m, re, handler, status] of ROUTES) {
    if (m !== method) continue;
    const match = clean.match(re);
    if (!match) continue;
    try {
      // structuredClone detaches responses from the store, like a real network boundary
      return { status: status ?? 200, body: structuredClone(handler(match, query, body, Date.now())) };
    } catch (err) {
      if (err instanceof MockHttpError) return { status: err.status, body: err.body };
      console.error('[mock api]', method, path, err);
      return { status: 500, body: { detail: err instanceof Error ? err.message : 'mock handler failed' } };
    }
  }
  return { status: 404, body: { detail: `no mock route for ${method} ${path}` } };
}
