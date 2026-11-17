import * as Popover from "@radix-ui/react-popover";
import clsx from "clsx";
import { motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Info, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";
import { G } from "../lib/glossary";
import { useMode } from "../lib/mode";

export const C = {
  lime: "#B6F24A", limeDim: "#5E8A17", ink: "#15171B", grid: "rgba(255,255,255,.05)", mute: "#8D929C", soft: "#C9CDD4",
  red: "#FF5D5D", amber: "#F6B44B", azure: "#4D8DFF", violet: "#A78BFA", teal: "#3DD9B3", white: "#F4F5F6", gray: "#3A3E47",
};
export const PHASE_COLOR: Record<string, string> = { "PBL-1": C.azure, "PBL-2": C.violet, "PBL-3": C.lime };
export const SERIES = [C.lime, C.azure, C.violet, C.amber, C.teal, C.red, "#F472B6", "#94A3B8"];
export const REGIME = ["#B6F24A", "#4D8DFF", "#F6B44B", "#A78BFA", "#3DD9B3", "#FF5D5D"];

/* ---------------------------------------------------------------- info popover */
export function InfoTip({ term, text, size = 14 }: { term?: string; text?: string; size?: number }) {
  const { mode } = useMode();
  const [open, setOpen] = useState(false);
  const t = term ? G[term] : undefined;
  if (!t && !text) return null;
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          aria-label={t ? `About ${t.title}` : "More information"}
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
          className="inline-flex shrink-0 items-center justify-center rounded-full text-mute transition hover:text-lime focus:outline-none focus-visible:ring-1 focus-visible:ring-lime"
        >
          <Info size={size} strokeWidth={2} />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content side="top" align="center" sideOffset={8} collisionPadding={12}
          onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
          className="z-50 w-[320px] rounded-xl border border-white/10 bg-ink-800/95 p-3.5 text-[12.5px] leading-relaxed text-soft shadow-2xl backdrop-blur-xl">
          {t ? (
            <>
              <div className="mb-1 text-[13px] font-semibold text-white">{t.title}</div>
              <p>{t.plain}</p>
              <div className={clsx("mt-2.5 rounded-lg border p-2.5 font-mono text-[11px] leading-relaxed",
                mode === "scientist" ? "border-lime/25 bg-lime/[0.06] text-lime-300" : "border-white/5 bg-ink-900/70 text-mute")}>
                <div className="mb-1 font-sans text-[10px] uppercase tracking-[0.14em] opacity-70">Technical</div>
                {t.tech}
              </div>
            </>
          ) : <p>{text}</p>}
          <Popover.Arrow className="fill-ink-800" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

/* ---------------------------------------------------------------- cards */
export function Card({ title, info, explain, action, children, className, glass, pad = true, delay = 0 }: {
  title?: ReactNode; info?: string; explain?: ReactNode; action?: ReactNode; children?: ReactNode;
  className?: string; glass?: boolean; pad?: boolean; delay?: number;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay, ease: "easeOut" }}
      className={clsx(glass ? "card-glass" : "card", pad && "p-5", className)}>
      {(title || action) && (
        <div className="mb-1 flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            {title && <h3 className="truncate text-[15px] font-semibold tracking-tight text-white">{title}</h3>}
            {info && <InfoTip term={info} />}
          </div>
          {action}
        </div>
      )}
      {explain && <p className="mb-3 text-[13px] leading-snug text-mute">{explain}</p>}
      {children}
    </motion.section>
  );
}

export function Delta({ value, suffix = "", good = "up", digits = 1, raw }: { value: number; suffix?: string; good?: "up" | "down"; digits?: number; raw?: string }) {
  const up = value >= 0;
  const positive = good === "up" ? up : !up;
  return (
    <span className={clsx("inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[11.5px] font-semibold num",
      positive ? "bg-lime/15 text-lime" : "bg-danger/15 text-danger")}>
      <span className="text-[9px]">{up ? "▲" : "▼"}</span>{raw ?? `${up ? "+" : "−"}${Math.abs(value).toFixed(digits)}${suffix}`}
    </span>
  );
}

export function Kpi({ label, value, delta, icon, info, foot, spark, accent, onClick, delay = 0 }: {
  label: string; value: ReactNode; delta?: ReactNode; icon?: ReactNode; info?: string; foot?: ReactNode;
  spark?: number[]; accent?: boolean; onClick?: () => void; delay?: number;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay }}
      onClick={onClick}
      className={clsx("card group relative overflow-hidden p-5", onClick && "cursor-pointer transition hover:border-lime/30")}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 label-caps">{icon}<span>{label}</span>{info && <InfoTip term={info} size={13} />}</div>
        <span className={clsx("grid h-8 w-8 place-items-center rounded-full transition",
          accent ? "bg-lime text-ink-950" : "bg-ink-700 text-soft group-hover:bg-lime group-hover:text-ink-950")}>
          <ArrowUpRight size={16} strokeWidth={2.4} />
        </span>
      </div>
      <div className="mt-4 flex items-end gap-2.5">
        <div className="text-[34px] font-semibold leading-none tracking-tight num">{value}</div>
        {delta && <div className="mb-1">{delta}</div>}
      </div>
      {foot && <div className="mt-2.5 text-[12.5px] text-mute">{foot}</div>}
      {spark && <div className="pointer-events-none absolute bottom-0 right-0 h-14 w-32 opacity-80"><Spark data={spark} fill /></div>}
    </motion.div>
  );
}

/* ---------------------------------------------------------------- small pieces */
export function Segmented<T extends string>({ options, value, onChange, size = "md" }: {
  options: { value: T; label: ReactNode }[]; value: T; onChange: (v: T) => void; size?: "sm" | "md";
}) {
  return (
    <div className="inline-flex flex-wrap items-center gap-1 rounded-xl border border-white/[0.06] bg-ink-900/80 p-1">
      {options.map((o) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={clsx("relative rounded-lg font-medium transition", size === "sm" ? "px-2.5 py-1 text-[12px]" : "px-3.5 py-1.5 text-[13px]",
            value === o.value ? "text-ink-950" : "text-mute hover:text-white")}>
          {value === o.value && <motion.span layoutId={`seg-${options.map((x) => x.value).join("")}`} className="absolute inset-0 rounded-lg bg-lime" transition={{ type: "spring", duration: 0.4 }} />}
          <span className="relative">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

export function Badge({ children, tone = "lime" }: { children: ReactNode; tone?: "lime" | "red" | "amber" | "azure" | "violet" | "gray" }) {
  const t = { lime: "bg-lime/15 text-lime border-lime/25", red: "bg-danger/15 text-danger border-danger/25", amber: "bg-amber/15 text-amber border-amber/25",
    azure: "bg-azure/15 text-azure border-azure/25", violet: "bg-violet/15 text-violet border-violet/25", gray: "bg-white/5 text-soft border-white/10" }[tone];
  return <span className={clsx("inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-wider", t)}>{children}</span>;
}

export const bandTone = (b: string) => (b === "Review now" ? "red" : b === "Standard" ? "amber" : "gray") as "red" | "amber" | "gray";
export const agreeTone = (a: string) => (a === "Models agree" ? "lime" : a === "Mixed" ? "amber" : "red") as "lime" | "amber" | "red";

export function PhaseTag({ phase }: { phase: string }) {
  return <Badge tone={phase === "PBL-1" ? "azure" : phase === "PBL-2" ? "violet" : "lime"}>{phase}</Badge>;
}

export function Spark({ data, color = C.lime, fill, height = "100%", strokeWidth = 1.6 }: { data: number[]; color?: string; fill?: boolean; height?: string | number; strokeWidth?: number }) {
  if (!data?.length) return null;
  const w = 100, h = 32, max = Math.max(...data, 1e-9), min = Math.min(...data, 0);
  const pts = data.map((v, i) => [(i / Math.max(data.length - 1, 1)) * w, h - ((v - min) / (max - min || 1)) * (h - 3) - 1.5]);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
  const id = `sg${Math.round(Math.random() * 1e9)}`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" width="100%" height={height}>
      {fill && <defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".35" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>}
      {fill && <path d={`${d} L${w},${h} L0,${h} Z`} fill={`url(#${id})`} />}
      <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

/** The lime "Insights" card from the reference design: one big plain-language sentence. */
export function InsightCard({ children, kicker = "Insight", foot, progress }: { children: ReactNode; kicker?: string; foot?: ReactNode; progress?: number }) {
  return (
    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.4 }}
      className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-lime-300 via-lime to-lime-600 p-6 text-ink-950 shadow-glow">
      <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-white/25 blur-2xl" />
      <Sparkles className="absolute right-6 top-6 opacity-70" size={26} />
      <span className="inline-flex items-center gap-1.5 rounded-full bg-ink-950 px-3 py-1 text-[12px] font-semibold text-lime"><Sparkles size={12} />{kicker}</span>
      <div className="relative mt-4 text-[22px] font-semibold leading-[1.3] tracking-tight">{children}</div>
      {foot && <div className="relative mt-3 text-[13px] font-medium text-ink-950/70">{foot}</div>}
      {progress != null && (
        <div className="mt-5 flex gap-1.5">
          {Array.from({ length: 10 }).map((_, i) => (
            <span key={i} className={clsx("h-2.5 w-6 rounded-sm", i < Math.round(progress * 10) ? "bg-ink-950" : "bg-ink-950/25")} />
          ))}
        </div>
      )}
    </motion.div>
  );
}
export const Hl = ({ children }: { children: ReactNode }) => <span className="mx-0.5 inline-block rounded-full border-2 border-ink-950 px-2 leading-tight">{children}</span>;

/** Plain-language bullet points. */
export function Takeaway({ points, title = "What this tells us" }: { points: ReactNode[]; title?: string }) {
  return (
    <div className="rounded-xl border border-lime/15 bg-lime/[0.04] p-4">
      <div className="mb-2 flex items-center gap-2 text-[12px] font-semibold uppercase tracking-[0.14em] text-lime"><Sparkles size={13} />{title}</div>
      <ul className="space-y-1.5 text-[13.5px] leading-relaxed text-soft">
        {points.map((p, i) => <li key={i} className="flex gap-2"><span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-lime" /><span>{p}</span></li>)}
      </ul>
    </div>
  );
}

export function Gauge({ value, label, sub, low = "Dismiss", high = "Report", marks }: { value: number; label: string; sub?: ReactNode; low?: string; high?: string; marks?: { v: number; label: string }[] }) {
  // semicircle: 0 at the left, 1 at the right
  const r = 80, cx = 100, cy = 96;
  const pt = (v: number, rr = r) => [cx - rr * Math.cos(Math.PI * v), cy - rr * Math.sin(Math.PI * v)];
  const arc = (a: number, b: number) => { const [x1, y1] = pt(a), [x2, y2] = pt(b); return `M${x1},${y1} A${r},${r} 0 0 1 ${x2},${y2}`; };
  const v = Math.max(0, Math.min(1, value));
  const [nx, ny] = pt(v, r - 16);
  const segs = [[0, 0.33, "#3DD9B3"], [0.335, 0.66, "#F6B44B"], [0.665, 1, "#FF5D5D"]] as const;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" className="w-full max-w-[280px]">
        {segs.map(([a, b, c]) => <path key={a} d={arc(a, b)} stroke={c} strokeWidth="12" fill="none" strokeLinecap="butt" opacity={0.9} />)}
        {Array.from({ length: 21 }).map((_, i) => { const [x1, y1] = pt(i / 20, r + 9), [x2, y2] = pt(i / 20, r + (i % 5 ? 12 : 15)); return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#3A3E47" strokeWidth="1" />; })}
        {marks?.map((m) => { const [x, y] = pt(m.v, r - 2); return <circle key={m.label} cx={x} cy={y} r="3" fill="#F4F5F6" stroke="#08090B" strokeWidth="1.5"><title>{m.label}</title></circle>; })}
        <motion.line x1={cx} y1={cy} initial={{ x2: pt(0, r - 16)[0], y2: pt(0, r - 16)[1] }} animate={{ x2: nx, y2: ny }} transition={{ type: "spring", stiffness: 60, damping: 12 }}
          stroke="#F4F5F6" strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="7" fill="#F4F5F6" /><circle cx={cx} cy={cy} r="3" fill="#08090B" />
        <text x="14" y="116" fill="#8D929C" fontSize="9">{low}</text>
        <text x="186" y="116" fill="#8D929C" fontSize="9" textAnchor="end">{high}</text>
      </svg>
      <div className="-mt-1 text-[26px] font-bold tracking-tight">{label}</div>
      {sub && <div className="text-[12.5px] text-mute">{sub}</div>}
    </div>
  );
}

export function Skeleton({ h = 200, className }: { h?: number; className?: string }) {
  return <div className={clsx("skeleton", className)} style={{ height: h }} />;
}

export function PageLoading() {
  return (
    <div className="grid gap-4 md:grid-cols-4">
      {[0, 1, 2, 3].map((i) => <Skeleton key={i} h={130} />)}
      <Skeleton h={340} className="md:col-span-3" /><Skeleton h={340} />
    </div>
  );
}

export function TrendIcon({ up }: { up: boolean }) {
  return up ? <ArrowUpRight size={14} className="text-lime" /> : <ArrowDownRight size={14} className="text-danger" />;
}

/** Recharts tooltip in the dashboard style. */
export function ChartTip({ active, payload, label, fmt, title }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-ink-800/95 px-3 py-2 text-[12px] shadow-2xl backdrop-blur-xl">
      {(title ?? label) != null && <div className="mb-1 font-semibold text-white">{title ?? label}</div>}
      {payload.filter((p: any) => p.value != null).map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2 text-soft">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color || p.payload?.fill || C.lime }} />
          <span className="text-mute">{p.name}</span>
          <span className="ml-auto pl-3 font-semibold text-white num">{fmt ? fmt(p.value, p) : typeof p.value === "number" ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

export function SectionTitle({ children, sub, right }: { children: ReactNode; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 mt-2 flex flex-wrap items-end justify-between gap-3">
      <div><h2 className="text-[18px] font-semibold tracking-tight">{children}</h2>{sub && <p className="mt-0.5 text-[13px] text-mute">{sub}</p>}</div>
      {right}
    </div>
  );
}

export function Stat({ label, value, info, tone }: { label: ReactNode; value: ReactNode; info?: string; tone?: string }) {
  return (
    <div className="inset px-3.5 py-3">
      <div className="flex items-center gap-1.5 text-[11.5px] text-mute">{label}{info && <InfoTip term={info} size={12} />}</div>
      <div className="mt-1 text-[19px] font-semibold tracking-tight num" style={tone ? { color: tone } : undefined}>{value}</div>
    </div>
  );
}

/** Horizontal bar with an optional interval whisker, used for leaderboards. */
export function MeterRow({ label, value, max, lo, hi, color = C.lime, right, base, onClick, active }: {
  label: ReactNode; value: number; max: number; lo?: number; hi?: number; color?: string; right?: ReactNode; base?: number; onClick?: () => void; active?: boolean;
}) {
  const w = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  return (
    <div onClick={onClick} className={clsx("grid grid-cols-[minmax(120px,1.1fr)_2fr_auto] items-center gap-3 rounded-lg px-2 py-1.5", onClick && "cursor-pointer hover:bg-white/[0.03]", active && "bg-white/[0.04]")}>
      <div className="truncate text-[13px] text-soft">{label}</div>
      <div className="relative h-5 rounded-md bg-ink-700/70">
        <motion.div initial={{ width: 0 }} animate={{ width: w(value) }} transition={{ duration: 0.6, ease: "easeOut" }} className="absolute inset-y-0 left-0 rounded-md" style={{ background: color, opacity: 0.85 }} />
        {lo != null && hi != null && <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-white/80" style={{ left: w(lo), width: `calc(${w(hi)} - ${w(lo)})` }} />}
        {base != null && <div className="absolute inset-y-[-3px] w-[2px] bg-danger" style={{ left: w(base) }} title="no-skill line" />}
      </div>
      <div className="min-w-[56px] text-right text-[13px] font-semibold num">{right ?? value.toFixed(3)}</div>
    </div>
  );
}
