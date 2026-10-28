import { useId, type ReactNode, type SelectHTMLAttributes, type InputHTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

/** Label + control + hint/error. Pass a render function to receive the generated id. */
export function Field({
  label,
  hint,
  error,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: string[] | string | null;
  children: (id: string, describedBy: string | undefined, invalid: boolean) => ReactNode;
  className?: string;
}) {
  const id = useId();
  const errs = !error ? [] : Array.isArray(error) ? error : [error];
  const descId = hint || errs.length ? `${id}-desc` : undefined;
  return (
    <div className={cn('flex min-w-0 flex-col gap-1', className)}>
      <label htmlFor={id} className="text-xs font-medium text-ink-2">
        {label}
      </label>
      {children(id, descId, errs.length > 0)}
      {(hint || errs.length > 0) && (
        <p id={descId} className={cn('text-2xs', errs.length ? 'text-delta-bad' : 'text-muted')}>
          {errs.length ? errs.join(' ') : hint}
        </p>
      )}
    </div>
  );
}

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input type="text" className={cn('field-input', className)} {...rest} />;
}

/** Number input bound to number | null (empty → null). */
export function NumberInput({
  value,
  onValue,
  className,
  ...rest
}: Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'type'> & { value: number | null; onValue: (v: number | null) => void }) {
  return (
    <input
      type="number"
      inputMode="decimal"
      className={cn('field-input', className)}
      value={value ?? ''}
      onChange={(e) => onValue(e.target.value === '' ? null : Number(e.target.value))}
      {...rest}
    />
  );
}

export function Select<T extends string>({
  value,
  onValue,
  options,
  className,
  ...rest
}: Omit<SelectHTMLAttributes<HTMLSelectElement>, 'value' | 'onChange'> & { value: T; onValue: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <select className={cn('field-input', className)} value={value} onChange={(e) => onValue(e.target.value as T)} {...rest}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function FormError({ message }: { message: string | null | undefined }) {
  if (!message) return null;
  return (
    <p role="alert" className="rounded-md border border-critical/40 bg-critical/10 px-3 py-2 text-xs text-ink">
      {message}
    </p>
  );
}
