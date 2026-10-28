import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { Icon, type IconName } from './Icon';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';

const VARIANT: Record<Variant, string> = {
  primary: 'bg-accent text-white hover:bg-accent-strong border border-transparent',
  secondary: 'bg-surface text-ink border border-line-strong hover:bg-surface-2',
  ghost: 'bg-transparent text-ink-2 border border-transparent hover:bg-surface-2 hover:text-ink',
  danger: 'bg-surface text-ink border border-critical/60 hover:bg-critical/10',
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
        'inline-flex items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-50',
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
