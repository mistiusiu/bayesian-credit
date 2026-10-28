import { cn } from '@/lib/cn';

/** Multi-select toggle chips (aria-pressed). */
export function Chips<T extends string>({
  options,
  value,
  onChange,
  label,
  size = 'md',
}: {
  options: { value: T; label: string; title?: string }[];
  value: T[];
  onChange: (v: T[]) => void;
  label: string;
  size?: 'sm' | 'md';
}) {
  const toggle = (v: T) => onChange(value.includes(v) ? value.filter((x) => x !== v) : [...value, v]);
  return (
    <div role="group" aria-label={label} className="flex flex-wrap gap-1.5">
      {options.map((o) => {
        const on = value.includes(o.value);
        return (
          <button
            key={o.value}
            type="button"
            aria-pressed={on}
            title={o.title}
            onClick={() => toggle(o.value)}
            className={cn(
              'rounded-full border font-medium whitespace-nowrap transition-colors',
              size === 'sm' ? 'h-6 px-2 text-2xs' : 'h-7 px-2.5 text-xs',
              on ? 'border-accent-strong bg-accent-wash text-ink' : 'border-line-strong text-muted hover:text-ink',
            )}
          >
            {on && <span aria-hidden="true">✓ </span>}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
