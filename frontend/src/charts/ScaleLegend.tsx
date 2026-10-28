/** Continuous scale legend (sequential or diverging) with end / mid labels. */
export function ScaleLegend({ stops, labels, title }: { stops: string[]; labels: [string, string] | [string, string, string]; title: string }) {
  const gradient = `linear-gradient(90deg, ${stops.map((s, i) => `${s} ${(i / (stops.length - 1)) * 100}%`).join(', ')})`;
  return (
    <div className="inline-flex flex-col gap-1 text-2xs text-muted" aria-label={`${title} colour scale`}>
      <span>{title}</span>
      <span className="block h-2 w-44 rounded-sm border border-line" style={{ background: gradient }} />
      <span className="num flex w-44 justify-between">
        {labels.map((l) => (
          <span key={l}>{l}</span>
        ))}
      </span>
    </div>
  );
}
