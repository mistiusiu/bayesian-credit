import type { ReactNode } from 'react';
import type { Stage } from '@/api/types';
import { cn } from '@/lib/cn';
import { STAGE_BLURB, STAGE_LABEL, STAGES } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { statusTone, TONE_COLOR, type Tone } from '@/lib/palette';
import { Icon, type IconName } from './Icon';

const TONE_ICON: Record<Tone, IconName> = {
  good: 'check',
  warning: 'alert',
  serious: 'alert',
  critical: 'x',
  neutral: 'circle',
  info: 'circle',
};

/** Status glyph: colour carries the state, the shape repeats it for colour-blind readers. */
export function StatusGlyph({ tone, pulse = false }: { tone: Tone; pulse?: boolean }) {
  if (tone === 'info')
    return (
      <span className="relative inline-flex size-2 shrink-0" aria-hidden="true">
        {pulse && <span className="absolute inline-flex size-full animate-ping rounded-full opacity-60" style={{ background: TONE_COLOR.info }} />}
        <span className="relative inline-flex size-2 rounded-full" style={{ background: TONE_COLOR.info }} />
      </span>
    );
  return <Icon name={TONE_ICON[tone]} size={12} strokeWidth={2.2} style={{ color: TONE_COLOR[tone] }} className="shrink-0" />;
}

export function Badge({ tone = 'neutral', children, className, title, pulse }: { tone?: Tone; children: ReactNode; className?: string; title?: string; pulse?: boolean }) {
  return (
    <span
      title={title}
      className={cn('inline-flex h-5 items-center gap-1.5 whitespace-nowrap rounded-full border border-line bg-surface px-2 text-2xs font-medium text-ink-2', className)}
    >
      <StatusGlyph tone={tone} pulse={pulse} />
      {children}
    </span>
  );
}

/** Badge for a lifecycle status string ("running", "ready", "awaiting_review", …). */
export function StatusBadge({ status, label, className }: { status: string; label?: string; className?: string }) {
  const tone = statusTone(status);
  return (
    <Badge tone={tone} pulse={status === 'running'} className={className}>
      {label ?? fmt.label(status)}
    </Badge>
  );
}

/** Table 3 stage: label plus an ordinal position marker (lifecycle order, one hue). */
export function StageBadge({ stage }: { stage: Stage }) {
  const pos = STAGES.indexOf(stage);
  return (
    <span
      title={STAGE_BLURB[stage]}
      className={cn(
        'inline-flex h-5 items-center gap-1.5 whitespace-nowrap rounded border px-1.5 text-2xs font-semibold tracking-wide uppercase',
        stage === 'pre_decision' ? 'border-accent-strong/50 bg-accent-wash text-ink' : 'border-line text-ink-2',
      )}
    >
      <span className="inline-flex gap-px" aria-hidden="true">
        {STAGES.map((s, i) => (
          <span key={s} className={cn('h-2 w-1 rounded-[1px]', i <= pos ? 'bg-accent-strong' : 'bg-surface-3')} />
        ))}
      </span>
      {STAGE_LABEL[stage]}
    </span>
  );
}

export function Pill({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn('inline-flex h-5 items-center rounded bg-surface-2 px-1.5 text-2xs font-medium text-ink-2', className)}>{children}</span>;
}
