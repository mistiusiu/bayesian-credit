import { useNavigate } from 'react-router';
import type { Tone } from '@/lib/palette';
import { TONE_COLOR } from '@/lib/palette';

export interface StageInfo {
  key: string;
  title: string;
  metric: string;
  to: string;
  tone?: Tone | null;
}

const W = 1120;
const H = 480;
const NW = 196;
const NH = 60;
/** Node centres around the loop (clockwise from Borrower Context). */
const POS: [number, number][] = [
  [215, 98],
  [480, 58],
  [745, 98],
  [828, 240],
  [745, 382],
  [480, 422],
  [215, 382],
  [132, 240],
];

/** Point where the segment from a box centre towards (tx, ty) leaves the box (plus a gap). */
function exitPoint(cx: number, cy: number, tx: number, ty: number, gap = 6): [number, number] {
  const dx = tx - cx;
  const dy = ty - cy;
  const sx = dx === 0 ? Infinity : (NW / 2 + gap) / Math.abs(dx);
  const sy = dy === 0 ? Infinity : (NH / 2 + gap) / Math.abs(dy);
  const s = Math.min(sx, sy);
  return [cx + dx * s, cy + dy * s];
}

function Node({ info, x, y, w = NW, h = NH, dashed = false }: { info: StageInfo; x: number; y: number; w?: number; h?: number; dashed?: boolean }) {
  const navigate = useNavigate();
  return (
    <a
      href={info.to}
      onClick={(e) => {
        e.preventDefault();
        void navigate(info.to);
      }}
      aria-label={`${info.title}: ${info.metric}`}
      className="group outline-none"
    >
      <rect
        x={x - w / 2}
        y={y - h / 2}
        width={w}
        height={h}
        rx={9}
        fill="var(--surface-1)"
        stroke={dashed ? 'var(--text-axis)' : 'var(--border-strong)'}
        strokeDasharray={dashed ? '4 3' : undefined}
        className="transition-colors group-hover:fill-[var(--surface-2)] group-focus-visible:stroke-[var(--focus)] group-focus-visible:[stroke-width:2]"
      />
      <text x={x - w / 2 + 12} y={y - 7} fontSize={13.5} fontWeight={600} fill="var(--text-primary)">
        {info.title}
      </text>
      <text x={x - w / 2 + 12} y={y + 14} fontSize={12} fill="var(--text-muted)" className="num">
        {info.metric.length > 29 ? `${info.metric.slice(0, 28)}…` : info.metric}
      </text>
      {info.tone && <circle cx={x + w / 2 - 13} cy={y - 11} r={4.5} fill={TONE_COLOR[info.tone]} stroke="var(--surface-1)" strokeWidth={2} />}
    </a>
  );
}

/**
 * The closed loop of §3.2 as a live diagram: eight stages, the PSI Sentinel spanning the
 * loop (frame + breaker on the adaptation edge) and the shadow engine running alongside
 * the historical policy (dashed = counterfactual). Every node links to its page.
 */
export function LoopDiagram({
  stages,
  sentinel,
  shadow,
  historical,
  center,
}: {
  stages: StageInfo[];
  sentinel: { label: string; tone: Tone; to: string; breakerOpen: boolean };
  shadow: StageInfo;
  historical: StageInfo;
  center: [string, string];
}) {
  const navigate = useNavigate();
  const edges = POS.map((p, i) => {
    const q = POS[(i + 1) % POS.length];
    const [x1, y1] = exitPoint(p[0], p[1], q[0], q[1]);
    const [x2, y2] = exitPoint(q[0], q[1], p[0], p[1]);
    return { i, x1, y1, x2, y2 };
  });
  const breakerEdge = edges[6];
  const bx = (breakerEdge.x1 + breakerEdge.x2) / 2;
  const by = (breakerEdge.y1 + breakerEdge.y2) / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="group" aria-label="Closed-loop architecture with live status" className="block">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--text-axis)" />
        </marker>
      </defs>
      <a
        href={sentinel.to}
        onClick={(e) => {
          e.preventDefault();
          void navigate(sentinel.to);
        }}
        aria-label={`PSI Sentinel: ${sentinel.label}`}
        className="group outline-none"
      >
        <rect x={14} y={14} width={950} height={H - 28} rx={18} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} className="group-focus-visible:stroke-[var(--focus)]" />
        <rect x={34} y={4} width={344} height={22} rx={11} fill="var(--surface-1)" stroke="var(--border-strong)" />
        <circle cx={50} cy={15} r={4.5} fill={TONE_COLOR[sentinel.tone]} />
        <text x={62} y={19} fontSize={12} fontWeight={600} fill="var(--text-primary)">
          PSI Sentinel
        </text>
        <text x={144} y={19} fontSize={11.5} fill="var(--text-muted)" className="num">
          {sentinel.label}
        </text>
      </a>
      {edges.map((e) => (
        <line key={e.i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke="var(--text-axis)" strokeWidth={1.5} markerEnd="url(#arrow)" />
      ))}
      <g aria-hidden="true">
        <rect x={bx - 28} y={by - 11} width={56} height={22} rx={5} fill="var(--surface-1)" stroke={sentinel.breakerOpen ? TONE_COLOR.critical : 'var(--border-strong)'} />
        <text x={bx} y={by + 4} textAnchor="middle" fontSize={10.5} fill={sentinel.breakerOpen ? 'var(--text-primary)' : 'var(--text-muted)'}>
          {sentinel.breakerOpen ? 'frozen' : 'breaker'}
        </text>
      </g>
      {stages.map((s, i) => (
        <Node key={s.key} info={s} x={POS[i][0]} y={POS[i][1]} />
      ))}
      <text x={480} y={232} textAnchor="middle" fontSize={15} fontWeight={600} fill="var(--text-primary)">
        {center[0]}
      </text>
      <text x={480} y={254} textAnchor="middle" fontSize={12.5} fill="var(--text-muted)" className="num">
        {center[1]}
      </text>
      <Node info={historical} x={1032} y={140} w={160} dashed={false} />
      <Node info={shadow} x={1032} y={320} w={160} dashed />
      <line x1={1032} y1={172} x2={1032} y2={286} stroke="var(--text-axis)" strokeWidth={1.5} markerEnd="url(#arrow)" />
      <path d={`M952 320 C 900 320, 880 360, 845 372`} fill="none" stroke="var(--text-axis)" strokeWidth={1.5} strokeDasharray="5 4" markerEnd="url(#arrow)" />
      <text x={1032} y={372} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)">
        counterfactual arms
      </text>
      <text x={1032} y={386} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)">
        per borrower
      </text>
    </svg>
  );
}
