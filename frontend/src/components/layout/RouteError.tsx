import { isRouteErrorResponse, Link, useRouteError } from 'react-router';

export function RouteError() {
  const err = useRouteError();
  const msg = isRouteErrorResponse(err) ? `${err.status} ${err.statusText}` : err instanceof Error ? err.message : 'Unknown error';
  return (
    <div role="alert" className="mx-auto mt-24 max-w-lg rounded-lg border border-line bg-surface p-6 text-center">
      <h1 className="text-base font-semibold text-ink">Something broke while rendering this page</h1>
      <p className="mt-2 text-xs text-muted">{msg}</p>
      <Link to="/" className="mt-4 inline-block text-xs text-accent-strong underline">
        Back to the overview
      </Link>
    </div>
  );
}
