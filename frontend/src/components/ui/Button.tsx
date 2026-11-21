import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { Icon, type IconName } from './Icon';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'link';

/** CTA styles: solid brand red primary, outlined secondary, quiet ghost, blue text link with an arrow. */
const VARIANT: Record<Variant, string> = {
  primary: 'bg-brand text-white hover:bg-brand-strong border border-transparent font-semibold',
  secondary: 'bg-transparent text-ink border border-ink/45 hover:border-ink hover:bg-surface-2',
  ghost: 'bg-transparent text-ink-2 border border-transparent hover:bg-surface-2 hover:text-ink',
  danger: 'bg-transparent text-ink border border-critical/60 hover:border-critical hover:bg-critical/10',
  link: 'cta-link border border-transparent px-0.5!',
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: 'sm' | 'md';
  icon?: IconName;
  loading?: boolean;
  children?: ReactNode;
}

export function Button({ variant = 'secondary', size = 'md', icon, loading = false, className, children, disabled, type = 'button', ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        'inline-flex items-center justify-center gap-1.5 rounded font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-50',
        size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8 px-3 text-[13px]',
        VARIANT[variant],
        className,
      )}
      {...rest}
    >
      {loading ? (
        <span className="size-3.5 animate-spin rounded-full border-2 border-current border-r-transparent" aria-hidden="true" />
      ) : (
        icon && <Icon name={icon} size={14} />
      )}
      {children}
    </button>
  );
}
