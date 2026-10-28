/**
 * Seeded pseudo-random numbers for the mock API. Every fixture is a pure function of
 * (resource id, seed) so reloading the page reproduces identical numbers.
 */

/** FNV-1a 32-bit hash of a string (seed derivation). */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Mix several integers / strings into one 32-bit seed. */
export function seedOf(...parts: (string | number)[]): number {
  return hashString(parts.join('␟'));
}

export class Rng {
  private s: number;
  private spare: number | null = null;

  constructor(seed: number | string) {
    this.s = (typeof seed === 'string' ? hashString(seed) : seed) >>> 0 || 0x9e3779b9;
  }

  /** mulberry32 — uniform in [0, 1). */
  next(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(a = 0, b = 1): number {
    return a + (b - a) * this.next();
  }

  /** Integer in [a, b] inclusive. */
  int(a: number, b: number): number {
    return a + Math.floor(this.next() * (b - a + 1));
  }

  bernoulli(p: number): boolean {
    return this.next() < p;
  }

  /** Standard normal via Marsaglia polar method. */
  normal(mu = 0, sd = 1): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return mu + sd * v;
    }
    let u: number;
    let v: number;
    let s: number;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * m;
    return mu + sd * u * m;
  }

  lognormal(mu: number, sigma: number): number {
    return Math.exp(this.normal(mu, sigma));
  }

  /** Gamma(k, 1) via Marsaglia–Tsang. */
  gamma(k: number): number {
    if (k < 1) {
      const u = this.next();
      return this.gamma(1 + k) * Math.pow(u, 1 / k);
    }
    const d = k - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let v: number;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.next();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  beta(a: number, b: number): number {
    const x = this.gamma(a);
    const y = this.gamma(b);
    return x / (x + y);
  }

  poisson(lambda: number): number {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= this.next();
    } while (p > L);
    return k - 1;
  }

  /** Index drawn from unnormalised weights. */
  weighted(weights: readonly number[]): number {
    let total = 0;
    for (const w of weights) total += w;
    let u = this.next() * total;
    for (let i = 0; i < weights.length; i++) {
      u -= weights[i];
      if (u <= 0) return i;
    }
    return weights.length - 1;
  }

  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length)];
  }
}
