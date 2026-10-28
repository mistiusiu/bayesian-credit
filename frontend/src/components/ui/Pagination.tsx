import { fmt } from '@/lib/format';
import { Button } from './Button';

export function Pagination({ offset, limit, count, onChange }: { offset: number; limit: number; count: number; onChange: (offset: number) => void }) {
  const from = count === 0 ? 0 : offset + 1;
  const to = Math.min(count, offset + limit);
  return (
    <nav aria-label="Pagination" className="flex items-center justify-between gap-3 text-xs text-muted">
      <span className="num">
        {fmt.int(from)}–{fmt.int(to)} of {fmt.int(count)}
      </span>
      <div className="flex gap-1.5">
        <Button size="sm" variant="secondary" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>
          Previous
        </Button>
        <Button size="sm" variant="secondary" disabled={to >= count} onClick={() => onChange(offset + limit)}>
          Next
        </Button>
      </div>
    </nav>
  );
}
