import type { LoanQuery } from '@/api/client';

export type LoanFilterState = Required<Pick<LoanQuery, 'grade' | 'purpose' | 'home_ownership'>> & { resolved: 'all' | 'true' | 'false'; search: string };

export const emptyLoanFilters: LoanFilterState = { grade: [], purpose: [], home_ownership: [], resolved: 'all', search: '' };

export function toLoanQuery(f: LoanFilterState, offset: number, limit: number): LoanQuery {
  return {
    limit,
    offset,
    grade: f.grade,
    purpose: f.purpose,
    home_ownership: f.home_ownership,
    resolved: f.resolved === 'all' ? null : f.resolved === 'true',
    search: f.search.trim() || undefined,
  };
}

