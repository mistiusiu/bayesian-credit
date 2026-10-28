import { useCallback } from 'react';
import { useSearchParams } from 'react-router';
import { useDatasets, useFits, useSimulations } from '@/api/client';
import type { Dataset, Fit, Simulation } from '@/api/types';
import { toId } from './params';

const SELECTION_KEYS = ['dataset', 'fit', 'sim'] as const;

export interface Selection {
  datasetId: number | null;
  fitId: number | null;
  simId: number | null;
  dataset: Dataset | null;
  fit: Fit | null;
  sim: Simulation | null;
  datasets: Dataset[];
  fits: Fit[];
  sims: Simulation[];
  loading: boolean;
  setDataset: (id: number | null) => void;
  setFit: (id: number | null) => void;
  setSim: (id: number | null) => void;
}

/**
 * The dataset → fit → simulation selection shared by every page, persisted in the URL
 * (?dataset=&fit=&sim=). Missing params resolve to the most recent ready resource
 * without rewriting the URL (derived state, no effects).
 */
export function useSelection(): Selection {
  const [params, setParams] = useSearchParams();
  const dsQ = useDatasets();
  const datasets = dsQ.data?.results ?? [];
  const pDs = toId(params.get('dataset'));
  const datasetId = pDs ?? datasets.find((d) => d.status === 'ready')?.id ?? datasets[0]?.id ?? null;

  const fitQ = useFits(datasetId, datasetId !== null);
  const fits = datasetId === null ? [] : (fitQ.data?.results ?? []);
  const pFit = toId(params.get('fit'));
  const fitValid = pFit !== null && (!fitQ.data || fits.some((f) => f.id === pFit));
  const fitId = fitValid ? pFit : (fits.find((f) => f.status === 'ready')?.id ?? fits[0]?.id ?? null);

  const simQ = useSimulations(fitId, fitId !== null);
  const sims = fitId === null ? [] : (simQ.data?.results ?? []);
  const pSim = toId(params.get('sim'));
  const simValid = pSim !== null && (!simQ.data || sims.some((s) => s.id === pSim));
  const simId = simValid ? pSim : (sims.find((s) => s.status === 'completed')?.id ?? sims[0]?.id ?? null);

  const write = useCallback(
    (patch: Partial<Record<(typeof SELECTION_KEYS)[number], number | null>>, clear: (typeof SELECTION_KEYS)[number][]) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const k of clear) next.delete(k);
          for (const [k, v] of Object.entries(patch)) {
            if (v === null || v === undefined) next.delete(k);
            else next.set(k, String(v));
          }
          return next;
        },
        { replace: false },
      ),
    [setParams],
  );

  return {
    datasetId,
    fitId,
    simId,
    dataset: datasets.find((d) => d.id === datasetId) ?? null,
    fit: fits.find((f) => f.id === fitId) ?? null,
    sim: sims.find((s) => s.id === simId) ?? null,
    datasets,
    fits,
    sims,
    loading: dsQ.isPending || (datasetId !== null && fitQ.isPending) || (fitId !== null && simQ.isPending),
    setDataset: useCallback((id) => write({ dataset: id }, ['fit', 'sim']), [write]),
    setFit: useCallback((id) => write({ fit: id }, ['sim']), [write]),
    setSim: useCallback((id) => write({ sim: id }, []), [write]),
  };
}

/** Search string carrying only the selection params (for nav links). */
export function selectionSearch(params: URLSearchParams): string {
  const next = new URLSearchParams();
  for (const k of SELECTION_KEYS) {
    const v = params.get(k);
    if (v) next.set(k, v);
  }
  const s = next.toString();
  return s ? `?${s}` : '';
}
