import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen, Boxes, CalendarRange, ChevronsLeft, ChevronsRight, Database, FlaskConical, GitBranch, LayoutGrid,
  ListOrdered, Menu, Search, ScanSearch, Sparkles, Swords, X,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useData } from "../lib/data";
import { f3, pct } from "../lib/fmt";
import { useMode } from "../lib/mode";
import { Badge } from "./ui";

export const NAV: { group: string; items: { to: string; label: string; icon: any; title: string; sub: string; badge?: string }[] }[] = [
  { group: "Operations", items: [
    { to: "/", label: "Overview", icon: LayoutGrid, title: "Overview", sub: "How the alert queue looks today, and how much work the model saves." },
    { to: "/queue", label: "Alert Queue", icon: ListOrdered, title: "Alert Queue", sub: "Every unseen alert, ranked by risk. Work it from the top." },
    { to: "/alert", label: "Alert Detail", icon: ScanSearch, title: "Alert Detail", sub: "Why one alert is ranked where it is." },
  ] },
  { group: "Patterns", items: [
    { to: "/clusters", label: "Cluster Explorer", icon: Boxes, title: "Cluster Explorer", sub: "Behavioural profiles found without looking at outcomes." },
    { to: "/sequences", label: "Sequence Analysis", icon: GitBranch, title: "Sequence Analysis", sub: "Hidden behavioural regimes in each client's year of transactions." },
  ] },
  { group: "Models", items: [
    { to: "/arena", label: "Model Arena", icon: Swords, badge: "UPDATED", title: "Model Arena", sub: "Every model from all three reviews, judged on the same unseen quarters." },
    { to: "/ensembles", label: "Ensemble Insights", icon: FlaskConical, badge: "NEW", title: "Ensemble Insights", sub: "How combining many models works, and whether it pays off here." },
  ] },
  { group: "Data", items: [
    { to: "/data", label: "Data Insights", icon: Database, title: "Data Insights", sub: "What the data looks like before any model touches it." },
    { to: "/about", label: "About & Method", icon: BookOpen, title: "About & Methodology", sub: "How AdaptiveAML was built and kept honest." },
  ] },
];

function Logo({ collapsed }: { collapsed: boolean }) {
  return (
    <NavLink to="/" className="flex items-center gap-2.5 px-1">
      <div className="relative grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-ink-700 ring-1 ring-white/10">
        <svg viewBox="0 0 32 32" className="h-5 w-5"><path d="M7 24 L16 8 L25 24" stroke="#B6F24A" strokeWidth="3.4" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
        <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-lime" />
      </div>
      {!collapsed && <div className="leading-tight"><div className="text-[15px] font-bold tracking-tight">Adaptive<span className="text-lime">AML</span></div><div className="text-[10.5px] text-mute">Alert triage · v3.0</div></div>}
    </NavLink>
  );
}

function ModeSwitch({ collapsed }: { collapsed: boolean }) {
  const { mode, setMode } = useMode();
  if (collapsed)
    return (
      <button onClick={() => setMode(mode === "analyst" ? "scientist" : "analyst")} title="Switch view mode"
        className="mx-auto grid h-10 w-10 place-items-center rounded-xl bg-ink-700 text-[11px] font-bold text-lime">{mode === "analyst" ? "AN" : "DS"}</button>
    );
  return (
    <div className="rounded-xl border border-white/[0.06] bg-ink-900 p-1">
      <div className="grid grid-cols-2 gap-1">
        {(["analyst", "scientist"] as const).map((m) => (
          <button key={m} onClick={() => setMode(m)}
            className={clsx("relative rounded-lg py-2 text-[12.5px] font-medium transition", mode === m ? "text-ink-950" : "text-mute hover:text-white")}>
            {mode === m && <motion.span layoutId="mode-pill" className="absolute inset-0 rounded-lg bg-lime" />}
            <span className="relative">{m === "analyst" ? "Analyst" : "Data scientist"}</span>
          </button>
        ))}
      </div>
      <p className="px-2 pb-1 pt-2 text-[11px] leading-snug text-mute">{mode === "analyst" ? "Plain language, key numbers only." : "Intervals, tests, parameters and formulas shown."}</p>
    </div>
  );
}

function Sidebar({ collapsed, setCollapsed, onNavigate }: { collapsed: boolean; setCollapsed: (b: boolean) => void; onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col gap-5 p-4">
      <div className="flex items-center justify-between"><Logo collapsed={collapsed} /></div>
      <nav className="flex-1 space-y-5 overflow-y-auto pr-1">
        {NAV.map((g) => (
          <div key={g.group}>
            {!collapsed && <div className="mb-1.5 px-3 text-[10.5px] font-medium uppercase tracking-[0.16em] text-mute/80">{g.group}</div>}
            <div className="space-y-0.5">
              {g.items.map((it) => (
                <NavLink key={it.to} to={it.to} end={it.to === "/"} onClick={onNavigate} title={collapsed ? it.label : undefined}
                  className={({ isActive }) => clsx("group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] transition",
                    isActive ? "bg-ink-700 text-white ring-1 ring-white/[0.06]" : "text-mute hover:bg-white/[0.03] hover:text-white", collapsed && "justify-center px-0")}>
                  {({ isActive }) => (
                    <>
                      {isActive && <motion.span layoutId="nav-active" className="absolute -left-4 top-2 bottom-2 w-1 rounded-r bg-lime" />}
                      <it.icon size={18} className={isActive ? "text-lime" : ""} strokeWidth={1.9} />
                      {!collapsed && <span className="flex-1">{it.label}</span>}
                      {!collapsed && it.badge && <Badge tone={it.badge === "NEW" ? "lime" : "azure"}>{it.badge}</Badge>}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
        {!collapsed && (
          <NavLink to="/ensembles" onClick={onNavigate}
            className="relative block overflow-hidden rounded-2xl border border-white/[0.07] bg-gradient-to-b from-ink-750 to-[#0f2a22] p-4 text-center">
            <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-teal/25 to-transparent blur-xl" />
            <div className="relative mx-auto mb-2 grid h-9 w-9 place-items-center rounded-full bg-white/10"><Sparkles size={16} className="text-lime" /></div>
            <div className="relative text-[13px] font-semibold">New: ensemble models</div>
            <p className="relative mt-1 text-[11.5px] leading-snug text-mute">Forests, boosting and a stacked committee — and an honest verdict on whether they help.</p>
            <span className="relative mt-3 inline-block rounded-lg bg-white px-3 py-1.5 text-[12px] font-semibold text-ink-950">Explore →</span>
          </NavLink>
        )}
      </nav>
      <ModeSwitch collapsed={collapsed} />
      <button onClick={() => setCollapsed(!collapsed)} className="hidden items-center justify-center gap-2 rounded-lg py-1.5 text-[12px] text-mute hover:text-white lg:flex">
        {collapsed ? <ChevronsRight size={16} /> : <><ChevronsLeft size={16} /> Collapse</>}
      </button>
    </div>
  );
}

function Ticker() {
  const { data: s } = useData<any>("summary.json");
  if (!s) return <div className="h-9 border-b border-white/[0.05] bg-ink-900" />;
  const champ = s.champion_row;
  const items: [string, string, string?][] = [
    ["IN SERVICE", `${s.champion} · PR-AUC ${f3(champ.pr_auc)}`, s.champion_phase],
    ["CHALLENGER", `${s.challenger}`, s.promoted ? "▲ promoted" : "not promoted"],
    ["BEST ENSEMBLE", `${s.best_test_row.model} · ${f3(s.best_test_row.pr_auc)}`, `${s.best_test_row.diff_vs_pbl1 >= 0 ? "▲" : "▼"} ${Math.abs(s.best_test_row.diff_vs_pbl1).toFixed(3)} vs PBL-1 (${s.best_test_row.verdict_vs_pbl1})`],
    ["BASE RATE", pct(s.base_rate, 1), "random ranking"],
    ["CATCH @20%", pct(champ.recall_at_budget, 1), `▲ ${(champ.recall_at_budget / s.budget).toFixed(2)}× random`],
    ["HIT RATE", pct(champ.precision_at_budget, 1), "of reviewed alerts"],
    ["SAFE TO DEFER", pct(champ.safe_deprioritised, 1), `${pct(champ.safe_recall, 0)} of reports kept`],
    ["TEST QUEUE", `${s.n_test.toLocaleString()} alerts`, "2021 Q3–Q4"],
    ["MODELS COMPARED", `${s.n_candidates}`, "across 3 reviews"],
    ["VERDICT", s.verdict.toUpperCase(), ""],
  ];
  const row = items.map(([k, v, d], i) => (
    <span key={i} className="flex items-center gap-2 whitespace-nowrap px-5 text-[11.5px]">
      <span className="font-semibold tracking-[0.12em] text-mute">{k}</span>
      <span className="font-semibold text-white num">{v}</span>
      {d && <span className={clsx("num", d.startsWith("▲") ? "text-lime" : d.startsWith("▼") ? "text-danger" : "text-mute")}>{d}</span>}
      <span className="ml-3 h-1 w-1 rounded-full bg-white/20" />
    </span>
  ));
  return (
    <div className="relative flex h-9 items-center overflow-hidden border-b border-white/[0.05] bg-ink-900">
      <div className="z-10 flex h-full shrink-0 items-center gap-2 border-r border-white/[0.06] bg-ink-900 px-4 text-[11px] font-bold tracking-[0.14em] text-lime">
        <span className="h-1.5 w-1.5 animate-pulseDot rounded-full bg-lime" />LIVE ON TEST QUARTERS
      </div>
      <div className="flex animate-ticker hover:[animation-play-state:paused]">{row}{row}</div>
    </div>
  );
}

function SearchBox() {
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const ref = useRef<HTMLInputElement>(null);
  const { data: queue } = useData<any[]>("queue.json");
  const [err, setErr] = useState("");
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "/" && document.activeElement?.tagName !== "INPUT") { e.preventDefault(); ref.current?.focus(); } };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
  const go = () => {
    const id = parseInt(q.replace(/\D/g, ""), 10);
    if (!id) return;
    if (queue && !queue.some((r) => r.id === id)) { setErr("Not in the test queue"); setTimeout(() => setErr(""), 2200); return; }
    nav(`/alert/${id}`); setQ("");
  };
  return (
    <div className="relative hidden md:block">
      <div className="flex h-10 w-[290px] items-center gap-2 rounded-xl border border-white/[0.07] bg-ink-850 px-3 text-mute focus-within:border-lime/40">
        <Search size={15} />
        <input ref={ref} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go()}
          placeholder="Jump to alert ID…" className="w-full bg-transparent text-[13px] text-white placeholder:text-mute focus:outline-none" />
        <span className="kbd">/</span>
      </div>
      {err && <div className="absolute left-0 top-12 z-20 rounded-lg bg-danger/90 px-3 py-1.5 text-[12px] text-white">{err}</div>}
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobile, setMobile] = useState(false);
  const loc = useLocation();
  const { mode } = useMode();
  const page = NAV.flatMap((g) => g.items).find((it) => (it.to === "/" ? loc.pathname === "/" : loc.pathname.startsWith(it.to))) ?? NAV[0].items[0];
  useEffect(() => { window.scrollTo({ top: 0 }); setMobile(false); }, [loc.pathname]);
  return (
    <div className="flex min-h-full flex-col">
      <Ticker />
      <div className="flex flex-1">
        <aside className={clsx("sticky top-0 hidden h-screen shrink-0 border-r border-white/[0.05] bg-ink-900/80 transition-[width] lg:block", collapsed ? "w-[84px]" : "w-[264px]")}>
          <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
        </aside>
        <AnimatePresence>
          {mobile && (
            <motion.div className="fixed inset-0 z-40 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="absolute inset-0 bg-black/60" onClick={() => setMobile(false)} />
              <motion.aside initial={{ x: -280 }} animate={{ x: 0 }} exit={{ x: -280 }} className="absolute inset-y-0 left-0 w-[270px] border-r border-white/10 bg-ink-900">
                <button className="absolute right-3 top-4 text-mute" onClick={() => setMobile(false)}><X size={18} /></button>
                <Sidebar collapsed={false} setCollapsed={() => {}} onNavigate={() => setMobile(false)} />
              </motion.aside>
            </motion.div>
          )}
        </AnimatePresence>
        <main className="glow-bg min-w-0 flex-1">
          <header className="sticky top-0 z-30 border-b border-white/[0.05] bg-ink-950/80 backdrop-blur-xl">
            <div className="mx-auto flex max-w-[1520px] items-center gap-4 px-5 py-4 lg:px-8">
              <button className="lg:hidden" onClick={() => setMobile(true)}><Menu size={20} /></button>
              <div className="min-w-0 flex-1">
                <h1 className="truncate text-[22px] font-semibold tracking-tight">{page.title}</h1>
                <p className="truncate text-[13px] text-mute">{page.sub}</p>
              </div>
              <SearchBox />
              <div className="hidden h-10 items-center gap-2 rounded-xl border border-white/[0.07] bg-ink-850 px-3 text-[13px] text-soft xl:flex">
                <CalendarRange size={15} className="text-mute" />Test · 2021 Q3–Q4
              </div>
              <div className={clsx("grid h-10 w-10 place-items-center rounded-xl text-[11.5px] font-bold", mode === "analyst" ? "bg-ink-700 text-soft" : "bg-lime text-ink-950")} title={mode === "analyst" ? "Analyst mode" : "Data scientist mode"}>
                {mode === "analyst" ? "AN" : "DS"}
              </div>
            </div>
          </header>
          <motion.div key={loc.pathname} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.25 }}
            className="mx-auto max-w-[1520px] px-5 pb-16 pt-6 lg:px-8">
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  );
}
