import { isRouteErrorResponse, Link, useRouteError } from 'react-router';

export function RouteError() {
  const err = useRouteError();
  const msg = isRouteErrorResponse(err) ? `${err.status} ${err.statusText}` : err instanceof Error ? err.message : 'Unknown error';
  return (
    <div role="alert" className="mx-auto mt-24 max-w-lg rounded border border-line bg-surface p-6 text-center">
      <h1 className="font-display text-lg font-bold tracking-tight text-ink">Something broke while rendering this page</h1>
      <p className="mt-2 text-xs text-muted">{msg}</p>
      <Link to="/" className="cta-link mt-4 text-[13px]">
        Back to the overview
      </Link>
    </div>
  );
}
