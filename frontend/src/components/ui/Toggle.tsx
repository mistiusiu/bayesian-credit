import { cn } from '@/lib/cn';

export function Toggle({ checked, onChange, label, description }: { checked: boolean; onChange: (v: boolean) => void; label: string; description?: string }) {
  return (
    <label className="inline-flex cursor-pointer items-start gap-2.5 select-none">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn('relative mt-0.5 h-4.5 w-8 shrink-0 rounded-full border transition-colors', checked ? 'border-accent bg-accent' : 'border-line-strong bg-surface-3')}
      >
        <span className={cn('absolute top-[1px] left-0 size-3.5 rounded-full bg-white shadow transition-transform', checked ? 'translate-x-[15px]' : 'translate-x-[1px]')} />
      </button>
      <span className="text-[13px] leading-5">
        <span className="text-ink">{label}</span>
        {description && <span className="block text-xs text-muted">{description}</span>}
      </span>
    </label>
  );
}
