import { useEffect, useRef, useState } from 'react';

/** Content width of an element, tracked with ResizeObserver (for pixel-exact custom SVG charts). */
export function useElementWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(Math.round(w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}
