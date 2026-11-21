import { useId } from 'react';
import { MOCK_MODE } from '@/api/client';
import { Icon } from '@/components/ui/Icon';
import { fmt } from '@/lib/format';
import { useSelection } from '@/lib/selection';
import { useTheme } from '@/lib/theme';

function Selector({ label, value, options, onChange, disabled }: { label: string; value: number | null; options: { id: number; label: string }[]; onChange: (id: number) => void; disabled?: boolean }) {
  const id = useId();
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <label htmlFor={id} className="text-2xs font-medium tracking-wide text-muted uppercase">
        {label}
      </label>
      <select
        id={id}
        className="field-input h-7 max-w-[15rem] min-w-[8rem] truncate text-xs xl:max-w-[19rem]"
        value={value ?? ''}
        disabled={disabled || options.length === 0}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {options.length === 0 && <option value="">none</option>}
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

const suffix = (status: string) => (status === 'ready' || status === 'completed' ? '' : ` · ${fmt.label(status).toLowerCase()}`);

export function TopBar() {
  const sel = useSelection();
  const { theme, toggle } = useTheme();
  return (
    // Masthead: near-black chrome in both themes (dark token scope), like the side nav.
    <header data-theme="dark" className="sticky top-0 z-30 flex min-h-14 items-center gap-4 border-b border-line bg-chrome px-6 py-2">
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-4 gap-y-1">
        <Selector label="Dataset" value={sel.datasetId} options={sel.datasets.map((d) => ({ id: d.id, label: `#${d.id} ${d.name}${suffix(d.status)}` }))} onChange={sel.setDataset} />
        <Selector label="Fit" value={sel.fitId} options={sel.fits.map((f) => ({ id: f.id, label: `#${f.id} ${f.name}${suffix(f.status)}` }))} onChange={sel.setFit} />
        <Selector label="Simulation" value={sel.simId} options={sel.sims.map((s) => ({ id: s.id, label: `#${s.id} ${s.name}${suffix(s.status)}` }))} onChange={sel.setSim} />
      </div>
      {MOCK_MODE && (
        <span className="eyebrow hidden rounded-sm border border-line-strong px-1.5 py-0.5 lg:inline" title="VITE_API_MOCK=1: data are seeded fixtures generated in the browser">
          Mock data
        </span>
      )}
      <button
        type="button"
        onClick={toggle}
        aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
        className="rounded p-1.5 text-ink-2 hover:bg-surface-2 hover:text-ink"
      >
        <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16} />
      </button>
    </header>
  );
}
