import { Link } from 'react-router';

export function NotFound() {
  return (
    <div className="mx-auto mt-20 max-w-md text-center">
      <div className="eyebrow">404</div>
      <h1 className="mt-1 font-display text-2xl font-bold tracking-tight text-ink">Page not found</h1>
      <p className="mt-1 text-xs text-muted">This route is not part of the console.</p>
      <Link to="/" className="cta-link mt-4 text-[13px]">
        Go to the overview
      </Link>
    </div>
  );
}
