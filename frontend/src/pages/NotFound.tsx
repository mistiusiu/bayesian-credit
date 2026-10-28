import { Link } from 'react-router';

export function NotFound() {
  return (
    <div className="mx-auto mt-20 max-w-md text-center">
      <h1 className="text-base font-semibold text-ink">Page not found</h1>
      <p className="mt-1 text-xs text-muted">This route is not part of the console.</p>
      <Link to="/" className="mt-3 inline-block text-xs text-accent-strong underline">
        Go to the overview
      </Link>
    </div>
  );
}
