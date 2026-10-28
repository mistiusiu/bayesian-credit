import { useCallback } from 'react';
import { useSearchParams } from 'react-router';

/** A single URL search param as React state (replace-navigation, other params preserved). */
export function useParam(key: string, fallback: string): [string, (v: string | null) => void];
export function useParam(key: string, fallback: null): [string | null, (v: string | null) => void];
export function useParam(key: string, fallback: string | null): [string | null, (v: string | null) => void] {
  const [params, setParams] = useSearchParams();
  const value = params.get(key) ?? fallback;
  const set = useCallback(
    (v: string | null) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (v === null || v === '') next.delete(key);
          else next.set(key, v);
          return next;
        },
        { replace: true },
      ),
    [key, setParams],
  );
  return [value, set];
}

/** Numeric id param (null when absent / invalid). */
export function toId(v: string | null): number | null {
  if (!v) return null;
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : null;
}
