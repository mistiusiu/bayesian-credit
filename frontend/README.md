# Bayesian Credit — research console

React front end for the shadow lending engine: borrower context → hierarchical Beta-Beta risk
→ MCTM credit states → risk-adjusted reward → contextual Thompson sampling → delayed outcomes
→ posterior update, with the PSI Sentinel circuit breaker and the counterfactual shadow
engine. Vite · React 19 · TypeScript (strict) · Tailwind CSS v4 · Recharts 3 · React Router 8 ·
TanStack Query 5. No UI kit: the small component set lives in `src/components/ui`.

The API contract is [`src/api/types.ts`](src/api/types.ts) (authoritative) and
[`../docs/API.md`](../docs/API.md) (human-readable, with examples).

## Run

Requires Node ≥ 22.12 and pnpm.

```bash
pnpm install
pnpm dev          # http://localhost:5173 — mock API in the browser (VITE_API_MOCK=1, .env.development)
pnpm dev:live     # same UI against Django on :8010 through the /api proxy (.env.live)
```

| Script | What it does |
|---|---|
| `pnpm dev` | Vite dev server, **mock mode** (seeded fixtures, no backend needed) |
| `pnpm dev:live` | Vite dev server, live API via the `/api` → `http://127.0.0.1:8010` proxy |
| `pnpm typecheck` | `tsc --noEmit` for the app and the Vite config |
| `pnpm lint` | ESLint (type-aware typescript-eslint, React Hooks / Compiler rules, react-refresh), zero warnings |
| `pnpm build` | typecheck + production build (live API; the mock is not bundled) |
| `pnpm build:demo` | typecheck + static build that ships the mock API (`.env.demo`) |
| `pnpm preview` | serve `dist/` (also proxies `/api`) |

Proxy target: set `VITE_API_PROXY_TARGET` (e.g. `http://localhost:8001`) to point dev /
preview elsewhere. The API base path is `VITE_API_BASE` (default `/api/v1`).

### Mock mode

`src/mocks` implements every endpoint of the contract in the browser. It generates a
Lending Club–like loan index from a seeded PRNG (200k loans, 2012–2018, LC status mix, sub-grade
pricing, grade-driven default, vintage-aware outcomes, a FICO/DTI shift at 2016-06) and
derives everything else from it: hierarchical posteriors for the 84 groups, holdout Brier /
AUC / ECE, recovery fit, the MCTM, per-borrower shadow reports, simulations (five policies,
breaker trip at 2016-06, a pending review, a chaos kill with checkpoint recovery), PSI bucket
tables and the Table 11 report. It also serves the model extras of docs/API.md §13.1: the
structured leaf prior (an offset logistic regression for the policy effect), the lending
guardrails with sub-grade reference rates, prepayment ρ by grade × term, recovery by term,
the portfolio decomposition / λ sweep / recommendation mix, and the report `details`. New
datasets, fits, simulations and portfolio runs progress in wall-clock time, so polling,
progress bars, pause / resume / chaos and the review workflow can be exercised. State resets on
reload. Mock-only modelling shortcuts are commented in the mock files; they are not part of
the contract.

## Structure

```
src/
  api/
    types.ts          authoritative JSON contract (every request/response shape)
    client.ts         fetch wrapper, ApiError, typed endpoint functions (`api`), query keys (`qk`), hooks
  mocks/              in-browser API: index.ts (router entry) · routes*.ts · store.ts (state, jobs, sim state machine)
                      loans.ts (seeded loan index) · riskModel.ts · policyEffect.ts · fitViews.ts · mctm.ts
                      shadow.ts · portfolio.ts · simulation.ts · simViews.ts · sentinel.ts · report.ts
                      reportDetails.ts · prng.ts · stats.ts · finance.ts
  lib/
    format.ts         all number / date formatting (Intl: currency, percent, compact, epochs)
    palette.ts        colour roles (validated categorical slots, sequential / diverging scales, status tones)
    domain.ts         labels, defaults, arm-grid builder, guardrail blurbs shared by UI and mocks
    counterfactual.ts uplift fractions, bucket labels, decomposition steps, ρ fallback
    guardrails.ts     browser re-check of G1–G4 (why an arm is not admissible)
    selection.ts      dataset → fit → simulation selection persisted in the URL
    params.ts, epoch.ts, theme.ts, loanFilters.ts, selectionMix.ts, useElementWidth.ts, cn.ts
  components/
    ui/               Card, StatTile, Badge, Button, Tabs, Segmented, Toggle, Chips, Field, DataTable,
                      Pagination, Progress (bar, meter), States (loading / empty / error), Disclosure, Icon
    layout/           AppShell, SideNav (running-job dots), TopBar (selectors, theme), PageHeader, RouteError
    counterfactual/   uplift decomposition, λ sweep, recommendation mix, eligibility, selection-mix table
    fit/              policy-effect card, prepayment ρ heatmap, guardrail strip
    LoanFilters.tsx, ThemeProvider.tsx
  charts/             ChartFrame (title · legend · axis labels · caption · table view), Legend, ChartTooltip,
                      kit.ts (axes, grid, domains), PolicyLines, SeriesLines, Heatmap, IntervalBar,
                      Density, Calibration, StripPlot, Occupancy, PosteriorBands, GroupedBars, Bars,
                      Dumbbell, DotWhisker, Waterfall, BreakerStrip, ScaleLegend, EndLabels
  pages/              overview · data · risk · mctm · bandit · shadow · sentinel · evaluation (one folder each)
  router.tsx          routes (pages are code-split)
  main.tsx            QueryClient, theme, router
```

### Conventions

* Every query has loading, empty and error states (`QueryState`); refetches keep the previous
  frame dimmed instead of flashing a skeleton (`placeholderData: keepPreviousData`).
* Running jobs and live simulations are polled every 1.5 s via `refetchInterval`, which stops
  as soon as the resource settles.
* The selection (`?dataset=&fit=&sim=`) lives in the URL; page state that should survive a
  reload or be shareable (tab, selected loan, group, feature, epoch…) also lives in the URL.
* Charts follow the dataviz method: categorical colour follows the entity (policy colours and
  credit-state colours were validated with the palette validator in light and dark),
  sequential = one blue ramp, diverging = red · neutral · blue, status colours only with an
  icon + label; 2px lines, hairline solid grids, one y-axis per chart, legends for ≥ 2 series,
  direct end labels only where they do not collide; every chart has axis/unit labels, a
  one-line caption and a data-table view.
* Dark theme is the default; the light theme is stored in `localStorage` (`bc-theme`).
* Numbers use tabular figures in tables and axes; money in USD, rates and shares as percent
  of a fraction (the API always sends fractions).

## Adding an endpoint

1. **Contract** — add the request / response interfaces to `src/api/types.ts` (snake_case,
   ids as numbers, epochs `"YYYY-MM"`, money in USD, rates as fractions, `null` for missing),
   and document the endpoint in `docs/API.md` with an example payload.
2. **Client** — add a function to `api` in `src/api/client.ts`, a key to `qk`, and a hook
   (`useQuery` for reads; `useMutation` + `invalidateQueries` for writes). Use
   `refetchInterval` if the resource can be running, `placeholderData: keepPreviousData` if its
   parameters change interactively.
3. **Mock** — add a route `[method, /^\/path\/(\d+)\/$/, handler, status?]` to
   `src/mocks/routesData.ts` or `routesSim.ts` returning the new shape (throw `MockHttpError`
   / use `badRequest`, `conflict`, `notFound` for errors).
4. **UI** — render it inside `QueryState`, format numbers with `fmt`, chart it through
   `ChartFrame` (caption + table view), keep components under ~250 lines.
5. Run `pnpm typecheck && pnpm lint && pnpm build`.
