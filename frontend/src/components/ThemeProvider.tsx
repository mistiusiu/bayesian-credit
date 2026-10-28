import { useMemo, useState, type ReactNode } from 'react';
import { applyTheme, readTheme, ThemeContext, type Theme } from '@/lib/theme';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(readTheme);
  const value = useMemo(
    () => ({
      theme,
      toggle: () => {
        const next: Theme = theme === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        setTheme(next);
      },
    }),
    [theme],
  );
  return <ThemeContext value={value}>{children}</ThemeContext>;
}
