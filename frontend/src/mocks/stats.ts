/** Beta-distribution and small statistical helpers used by the mock risk model. */

const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function lnGamma(z: number): number {
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  const x = z - 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function lnBeta(a: number, b: number): number {
  return lnGamma(a) + lnGamma(b) - lnGamma(a + b);
}

export function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) {
    if ((x === 0 && a < 1) || (x === 1 && b < 1)) return Infinity;
    if ((x === 0 && a === 1) || (x === 1 && b === 1)) return Math.exp(-lnBeta(a, b));
    return 0;
  }
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lnBeta(a, b));
}

/** Continued fraction for the regularised incomplete beta (Numerical Recipes betacf). */
function betacf(x: number, a: number, b: number): number {
  const MAXIT = 200;
  const EPS = 3e-14;
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularised incomplete beta I_x(a, b) = Beta CDF. */
export function betaCdf(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return (bt * betacf(x, a, b)) / a;
  return 1 - (bt * betacf(1 - x, b, a)) / b;
}

/** Beta quantile by safeguarded bisection on the CDF (60 iterations ≈ 1e-18 width). */
export function betaQuantile(q: number, a: number, b: number): number {
  if (q <= 0) return 0;
  if (q >= 1) return 1;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (betaCdf(mid, a, b) < q) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/** Shortest interval with the given mass (golden-section search over the lower tail). */
export function betaHdi(a: number, b: number, mass = 0.95): [number, number] {
  if (a <= 1 || b <= 1) {
    // monotone / J-shaped densities: the HDI touches a boundary
    if (a <= 1 && b > 1) return [0, betaQuantile(mass, a, b)];
    if (b <= 1 && a > 1) return [betaQuantile(1 - mass, a, b), 1];
  }
  const width = (u: number) => betaQuantile(u + mass, a, b) - betaQuantile(u, a, b);
  const gr = (Math.sqrt(5) - 1) / 2;
  let lo = 0;
  let hi = 1 - mass;
  let c = hi - gr * (hi - lo);
  let d = lo + gr * (hi - lo);
  for (let i = 0; i < 28; i++) {
    if (width(c) < width(d)) hi = d;
    else lo = c;
    c = hi - gr * (hi - lo);
    d = lo + gr * (hi - lo);
  }
  const u = 0.5 * (lo + hi);
  return [betaQuantile(u, a, b), betaQuantile(u + mass, a, b)];
}

/** Fast approximate HDI (normal approximation on the logit scale) for bulk tables. */
export function betaHdiApprox(a: number, b: number, mass = 0.95): [number, number] {
  if (a + b > 400) {
    const m = a / (a + b);
    const s = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)));
    const z = normalQuantile(0.5 + mass / 2);
    return [Math.max(0, m - z * s), Math.min(1, m + z * s)];
  }
  return betaHdi(a, b, mass);
}

export function betaMean(a: number, b: number): number {
  return a / (a + b);
}

export function betaStd(a: number, b: number): number {
  const s = a + b;
  return Math.sqrt((a * b) / (s * s * (s + 1)));
}

/** Density on a grid of n points over [0, 1] (endpoints nudged inside to stay finite). */
export function betaPdfGrid(a: number, b: number, n = 101): { x: number; density: number }[] {
  const out: { x: number; density: number }[] = [];
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    const xs = Math.min(Math.max(x, 0.002), 0.998);
    const d = betaPdf(xs, a, b);
    out.push({ x, density: Number.isFinite(d) ? Math.round(d * 1e5) / 1e5 : 0 });
  }
  return out;
}

/** Acklam's inverse normal CDF. */
export function normalQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

export function quantileSorted(sorted: ArrayLike<number>, q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function round(x: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}
