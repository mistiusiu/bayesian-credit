import type { SVGProps } from 'react';

const PATHS = {
  overview: 'M8 2.5a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11Zm0 0v2m5.5 3.5h-2m-3.5 5.5v-2M2.5 8h2M8 8l2.5-2.5',
  data: 'M2.5 4c0-.8 2.5-1.5 5.5-1.5s5.5.7 5.5 1.5v8c0 .8-2.5 1.5-5.5 1.5S2.5 12.8 2.5 12V4Zm0 0c0 .8 2.5 1.5 5.5 1.5s5.5-.7 5.5-1.5M2.5 8c0 .8 2.5 1.5 5.5 1.5s5.5-.7 5.5-1.5',
  risk: 'M2.5 13.5c1.5 0 2-9 5.5-9s4 9 5.5 9M2 13.5h12',
  mctm: 'M4 4.5a1.5 1.5 0 1 1 0 .01M12 4.5a1.5 1.5 0 1 1 0 .01M4 11.5a1.5 1.5 0 1 1 0 .01M12 11.5a1.5 1.5 0 1 1 0 .01M5.5 4.5h5M4 6v4M12 6v4M5.3 5.3l5.4 5.4',
  bandit: 'M3 13.5V8m3.3 5.5V5m3.4 8.5v-4m3.3 4V3M2 13.5h12',
  shadow: 'M2.5 8s2-4.5 5.5-4.5S13.5 8 13.5 8 11.5 12.5 8 12.5 2.5 8 2.5 8Zm5.5-2a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z',
  sentinel: 'M8 2 3 4v4c0 3 2.2 5.2 5 6 2.8-.8 5-3 5-6V4L8 2Zm-2 6 1.5 1.5L10.5 6.5',
  evaluation: 'M3.5 2.5h9v11h-9v-11Zm2 3h5m-5 2.5h5m-5 2.5h3',
  sun: 'M8 5a3 3 0 1 1 0 6 3 3 0 0 1 0-6Zm0-3.5v1.5m0 10v1.5M1.5 8H3m10 0h1.5M3.4 3.4l1 1m7.2 7.2 1 1m0-9.2-1 1m-7.2 7.2-1 1',
  moon: 'M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5Z',
  check: 'M3.5 8.5 6.5 11.5 12.5 4.5',
  alert: 'M8 2.5 14 13H2L8 2.5Zm0 4v3m0 2v.01',
  x: 'M4 4l8 8m0-8-8 8',
  half: 'M8 2.5a5.5 5.5 0 1 0 0 11V2.5Z',
  play: 'M5 3.5v9l7-4.5-7-4.5Z',
  pause: 'M5 3.5v9m6-9v9',
  stop: 'M4 4h8v8H4z',
  bolt: 'M9 1.5 3.5 9H8l-1 5.5L12.5 7H8l1-5.5Z',
  download: 'M8 2.5v8m-3.5-3.5L8 10.5 11.5 7M3 13.5h10',
  chevronDown: 'M4 6l4 4 4-4',
  chevronRight: 'M6 4l4 4-4 4',
  plus: 'M8 3v10M3 8h10',
  minus: 'M3 8h10',
  info: 'M8 2.5a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11ZM8 7v4m0-6v.01',
  lock: 'M4.5 7.5h7v6h-7v-6Zm1.5 0V5.5a2 2 0 1 1 4 0v2',
  table: 'M2.5 3.5h11v9h-11v-9Zm0 3h11m-11 3h11M6.5 3.5v9',
  chart: 'M2.5 13.5h11M4 11l3-3.5 2.5 2L13 5',
  refresh: 'M13 8a5 5 0 1 1-1.5-3.5M13 2.5v2.5h-2.5',
  external: 'M9.5 2.5h4v4m0-4L8 8m-1.5-4.5h-3v9h9v-3',
  circle: 'M8 3a5 5 0 1 1 0 10A5 5 0 0 1 8 3Z',
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
