export const pct = (v: number | null | undefined, d = 0) => (v == null || !isFinite(v) ? "–" : `${(v * 100).toFixed(d)}%`);
export const f3 = (v: number | null | undefined) => (v == null || !isFinite(v) ? "–" : v.toFixed(3));
export const f2 = (v: number | null | undefined) => (v == null || !isFinite(v) ? "–" : v.toFixed(2));
export const num = (v: number | null | undefined) => (v == null ? "–" : Math.round(v).toLocaleString("en-IN"));
export const signed = (v: number, d = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
export const pp = (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)} pts`;
export const ci = (c?: number[] | null, d = 3) => (c && c.length === 2 ? `[${c[0].toFixed(d)}, ${c[1].toFixed(d)}]` : "");
export const compact = (v: number) => Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);
export const short = (s: string, n = 34) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
