import clsx from "clsx";
import { ArrowDownUp, ChevronLeft, ChevronRight, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Area, AreaChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useNavigate } from "react-router-dom";
import { Badge, C, Card, ChartTip, InfoTip, PageLoading, Segmented, Spark, Stat, Takeaway, agreeTone, bandTone } from "../components/ui";
import { useData } from "../lib/data";
import { num, pct } from "../lib/fmt";
import { DS, useMode } from "../lib/mode";

type Row = { id: number; q: string; risk: number; rank: number; band: string; agree: string; y: number; reason: string; spark: number[] };

export default function Queue() {
  const { data: rows } = useData<Row[]>("queue.json");
  const { data: s } = useData<any>("summary.json");
  const { mode } = useMode();
  const nav = useNavigate();
  const [band, setBand] = useState("All");
  const [agree, setAgree] = useState("All");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ k: keyof Row; dir: 1 | -1 }>({ k: "rank", dir: 1 });
  const [page, setPage] = useState(0);
  const [budget, setBudget] = useState(20);

  const curve = useMemo(() => {
    if (!rows) return [];
    const total = rows.reduce((a, r) => a + r.y, 0);
    const sorted = [...rows].sort((a, b) => a.rank - b.rank);
    let c = 0;
    const out: { x: number; caught: number; random: number }[] = [{ x: 0, caught: 0, random: 0 }];
    sorted.forEach((r, i) => { c += r.y; if ((i + 1) % 33 === 0 || i === sorted.length - 1) out.push({ x: (i + 1) / sorted.length, caught: c / total, random: (i + 1) / sorted.length }); });
    return out;
  }, [rows]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    let r = rows;
    if (band !== "All") r = r.filter((x) => x.band === band);
    if (agree !== "All") r = r.filter((x) => x.agree === agree);
    if (q) r = r.filter((x) => String(x.id).includes(q.trim()));
    return [...r].sort((a, b) => ((a[sort.k] as number) > (b[sort.k] as number) ? 1 : -1) * sort.dir);
  }, [rows, band, agree, q, sort]);

  if (!rows || !s) return <PageLoading />;
  const total = rows.reduce((a, r) => a + r.y, 0);
  const k = Math.ceil((budget / 100) * rows.length);
  const caught = [...rows].sort((a, b) => a.rank - b.rank).slice(0, k).reduce((a, r) => a + r.y, 0);
  const per = 25, pages = Math.ceil(filtered.length / per);
  const view = filtered.slice(page * per, page * per + per);
  const head = (key: keyof Row, label: string, cls = "") => (
    <th className={clsx("px-3 py-3 font-medium", cls)}>
      <button onClick={() => setSort({ k: key, dir: sort.k === key ? (-sort.dir as 1 | -1) : key === "risk" ? -1 : 1 })} className="inline-flex items-center gap-1 hover:text-white">
        {label}<ArrowDownUp size={11} className={sort.k === key ? "text-lime" : "opacity-40"} />
      </button>
    </th>
  );

  return (
    <div className="space-y-5">
      <div className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
        <Card title="How much of the queue do you have time for?" info="recall_budget"
          explain="Drag the slider to choose how much of the queue analysts can open. The curve shows how many reportable alerts that finds, compared with opening alerts at random (dashed).">
          <div className="h-[230px]">
            <ResponsiveContainer>
              <AreaChart data={curve} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                <defs><linearGradient id="qg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={C.lime} stopOpacity={0.35} /><stop offset="1" stopColor={C.lime} stopOpacity={0} /></linearGradient></defs>
                <XAxis dataKey="x" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v)} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTip title=" " fmt={(v: number) => pct(v, 1)} />} labelFormatter={() => ""} />
                <Area dataKey="random" name="Random order" stroke={C.mute} strokeDasharray="4 4" fill="none" />
                <Area dataKey="caught" name="Model order" stroke={C.lime} strokeWidth={2.2} fill="url(#qg)" />
                <ReferenceLine x={budget / 100} stroke="#fff" strokeOpacity={0.5} strokeDasharray="3 3" />
                <ReferenceDot x={budget / 100} y={caught / total} r={6} fill={C.lime} stroke="#08090B" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex items-center gap-4">
            <span className="text-[12px] text-mute">Review</span>
            <input type="range" min={5} max={60} step={5} value={budget} onChange={(e) => setBudget(+e.target.value)} className="flex-1 accent-[#B6F24A]" />
            <span className="w-12 text-right text-[15px] font-semibold num">{budget}%</span>
          </div>
        </Card>
        <div className="grid content-start gap-4">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Alerts opened" value={num(k)} />
            <Stat label="Reportable found" value={`${num(caught)}`} tone={C.lime} />
            <Stat label="Share of all" value={pct(caught / total)} info="recall_budget" />
          </div>
          <Takeaway points={[
            <>Opening <b className="text-white">{budget}%</b> of the queue ({num(k)} alerts) finds <b className="text-white">{pct(caught / total)}</b> of all reportable cases — random review would find about {budget}%.</>,
            <>Every alert opened is a real analyst hour, so the left part of the curve is where the value is: it climbs fastest there.</>,
            <>The default budget used everywhere in this dashboard is <b className="text-white">20%</b>.</>,
          ]} />
          <DS><div className="inset p-3 text-[12px] text-mute">Rank-based top-k on calibrated champion scores ({s.champion}); ties broken by queue order. Outcome labels are used only to draw this curve, never to rank.</div></DS>
        </div>
      </div>

      <Card pad={false}>
        <div className="flex flex-wrap items-center gap-3 border-b border-white/[0.05] p-4">
          <Segmented value={band} onChange={(v) => { setBand(v); setPage(0); }} options={["All", "Review now", "Standard", "Low priority"].map((v) => ({ value: v, label: v }))} />
          <Segmented size="sm" value={agree} onChange={(v) => { setAgree(v); setPage(0); }} options={[{ value: "All", label: "Any agreement" }, { value: "Models agree", label: "Agree" }, { value: "Mixed", label: "Mixed" }, { value: "Models disagree", label: "Disagree" }]} />
          <InfoTip term="agreement" />
          <div className="ml-auto flex h-9 items-center gap-2 rounded-xl border border-white/[0.07] bg-ink-900 px-3">
            <Search size={14} className="text-mute" />
            <input value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} placeholder="Filter by ID" className="w-28 bg-transparent text-[13px] focus:outline-none" />
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[920px] text-left text-[13px]">
            <thead className="text-[11.5px] uppercase tracking-wider text-mute">
              <tr className="border-b border-white/[0.05]">
                {head("rank", "Rank", "pl-5")}{head("id", "Alert")}{head("risk", "Risk")}
                <th className="px-3 py-3 font-medium">Band</th><th className="px-3 py-3 font-medium">Agreement</th>
                <th className="px-3 py-3 font-medium">Main reason</th><th className="px-3 py-3 font-medium">Last 12 months</th>
                {mode === "scientist" && <th className="px-3 py-3 font-medium">Outcome</th>}
              </tr>
            </thead>
            <tbody>
              {view.map((r) => (
                <tr key={r.id} onClick={() => nav(`/alert/${r.id}`)} className="cursor-pointer border-b border-white/[0.04] transition hover:bg-white/[0.025]">
                  <td className="py-2.5 pl-5 pr-3 text-mute num">#{r.rank}</td>
                  <td className="px-3 py-2.5 font-medium">{r.id}<span className="ml-2 text-[11px] text-mute">{r.q}</span></td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2"><span className="w-10 font-semibold num">{pct(r.risk)}</span>
                      <div className="h-1.5 w-20 rounded-full bg-ink-700"><div className="h-full rounded-full" style={{ width: `${Math.min(100, r.risk * 160)}%`, background: r.band === "Review now" ? C.lime : r.band === "Standard" ? "#6E8F2E" : C.gray }} /></div></div>
                  </td>
                  <td className="px-3 py-2.5"><Badge tone={bandTone(r.band)}>{r.band}</Badge></td>
                  <td className="px-3 py-2.5"><Badge tone={agreeTone(r.agree)}>{r.agree.replace("Models ", "")}</Badge></td>
                  <td className="max-w-[260px] truncate px-3 py-2.5 text-soft">{r.reason}</td>
                  <td className="px-3 py-2.5"><div className="h-7 w-28"><Spark data={r.spark} color={r.band === "Review now" ? C.lime : C.mute} /></div></td>
                  {mode === "scientist" && <td className="px-3 py-2.5">{r.y ? <Badge tone="red">Reported</Badge> : <Badge tone="gray">Dismissed</Badge>}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between p-4 text-[12.5px] text-mute">
          <span>{num(filtered.length)} alerts{mode === "scientist" && <> · outcome shown for evaluation only, never used for ranking</>}</span>
          <div className="flex items-center gap-2">
            <button disabled={page === 0} onClick={() => setPage(page - 1)} className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 disabled:opacity-30"><ChevronLeft size={15} /></button>
            <span className="num">{page + 1} / {Math.max(pages, 1)}</span>
            <button disabled={page >= pages - 1} onClick={() => setPage(page + 1)} className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 disabled:opacity-30"><ChevronRight size={15} /></button>
          </div>
        </div>
      </Card>
    </div>
  );
}
