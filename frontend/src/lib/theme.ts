import { createContext, useContext } from 'react';

export type Theme = 'dark' | 'light';

export interface ThemeState {
  theme: Theme;
  toggle: () => void;
}

export const ThemeContext = createContext<ThemeState>({ theme: 'dark', toggle: () => undefined });

export const useTheme = () => useContext(ThemeContext);

export function readTheme(): Theme {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

export function applyTheme(t: Theme): void {
  document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem('bc-theme', t);
  } catch {
    /* storage unavailable (private mode) */
  }
}
