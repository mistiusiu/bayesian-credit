/** Request validation for the mock API (DRF-style field errors → 400). */
import { isEpoch } from '@/lib/epoch';
import { MockHttpError } from './store';

export class Errors {
  private errs: Record<string, string[]> = {};
  add(field: string, msg: string) {
    (this.errs[field] ??= []).push(msg);
  }
  check(cond: boolean, field: string, msg: string) {
    if (!cond) this.add(field, msg);
  }
  throwIfAny() {
    if (Object.keys(this.errs).length) throw new MockHttpError(400, this.errs);
  }
}

export const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
export const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
export const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);
export const epochOk = (v: unknown) => v === undefined || v === null || (typeof v === 'string' && isEpoch(v));

export function body(raw: unknown): Record<string, unknown> {
  if (!isObj(raw)) throw new MockHttpError(400, { detail: 'JSON object body required' });
  return raw;
}

export function notFound(what: string): never {
  throw new MockHttpError(404, { detail: `${what} not found` });
}

export function conflict(detail: string): never {
  throw new MockHttpError(409, { detail });
}

export function badRequest(field: string, msg: string): never {
  throw new MockHttpError(400, { [field]: [msg] });
}
