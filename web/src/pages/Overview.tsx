import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Clock, Crosshair, Inbox, MessageSquareText, Send, ShieldCheck, Target } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine, LabelList } from "recharts";
import { useNavigate } from "react-router-dom";
import { Badge, C, Card, ChartTip, Delta, Hl, InfoTip, InsightCard, Kpi, PageLoading, PHASE_COLOR, PhaseTag, Segmented, Spark, agreeTone, bandTone } from "../components/ui";
import { useData } from "../lib/data";
import { ci, f3, num, pct } from "../lib/fmt";
import { DS } from "../lib/mode";

export default function Overview() {
  const { data: s } = useData<any>("summary.json");
  const nav = useNavigate();
  const [metric, setMetric] = useState<"rate" | "lift">("rate");
  const [askOpen, setAskOpen] = useState<number | null>(0);
  if (!s) return <PageLoading />;
  const ch = s.champion_row, p1 = s.pbl1_row;
  const lift = ch.recall_at_budget / s.budget;
  const top10 = s.deciles[0];
  const bands = ["Review now", "Standard", "Low priority"].map((b) => ({ name: b, value: s.bands[b] ?? 0, rate: s.band_rates[b] ?? 0 }));
  const bandColor: Record<string, string> = { "Review now": C.lime, Standard: "#6E8F2E", "Low priority": "#2F3338" };
  const decData = s.deciles.map((d: any) => ({ ...d, name: d.decile === 1 ? "Top 10%" : `${(d.decile - 1) * 10}–${d.decile * 10}%`, lift: d.rate / s.base_rate }));
  const phaseMax = Math.max(...s.phases.map((p: any) => p.pr_auc_ci[1]));

  const questions = [
    { q: "How much work does this save?", a: <>Reviewing only the top <b>{pct(s.budget)}</b> of the queue ({num(s.reviewed)} alerts) finds <b>{num(s.caught)}</b> of the {num(s.n_reported)} alerts that end up reported — <b>{pct(ch.recall_at_budget)}</b>, against {pct(s.budget)} if alerts were opened in random order. At the bottom, <b>{pct(ch.safe_deprioritised)}</b> of alerts can wait while still keeping {pct(ch.safe_recall)} of reportable cases in view.</> },
    { q: "Did the ensembles help?", a: <>Not reliably. The best new model on early-2021 data, <b>{s.challenger}</b>, was not clearly better than the incumbent {s.pbl1_champion} (gap {s.valid_gap.gap >= 0 ? "+" : "−"}{Math.abs(s.valid_gap.gap).toFixed(3)}, range {ci(s.valid_gap.ci)}), so the <b>{s.champion}</b> keeps running the queue. On the test quarters the strongest ensemble, <b>{s.best_test_row.model}</b>, scores {f3(s.best_test_row.pr_auc)} vs {f3(p1.pr_auc)} — a gap of {ci(s.best_test_row.diff_vs_pbl1_ci)} that includes zero. The data, not the model family, is the limit.</> },
    { q: "Which alerts should I trust least?", a: <>The ones where the six models disagree. Alerts marked <b>“Models disagree”</b> ({num(s.agreement.find((a: any) => a.agreement === "Models disagree")?.n)}) get very different scores depending on which model you ask — open them and use judgement rather than the score alone.</> },
    { q: "Why not just use accuracy?", a: <>Because only {pct(s.base_rate)} of alerts get reported. A model that dismisses everything would be {pct(1 - s.base_rate)} “accurate” and catch nothing. We use PR-AUC, which only rewards putting real cases near the top.</> },
  ];

  return (
    <div className="space-y-5">
      {/* KPI row */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Alerts in queue" icon={<Inbox size={14} />} value={num(s.n_test)} info="temporal" delay={0}
          foot={<>From 2021 Q3–Q4 · <b className="text-white">{num(s.n_reported)}</b> later reported</>} onClick={() => nav("/queue")} />
        <Kpi label="Caught in top 20%" icon={<Target size={14} />} value={pct(ch.recall_at_budget)} info="recall_budget" accent delay={0.05}
          delta={<Delta value={(ch.recall_at_budget - s.budget) * 100} suffix=" pts" good="up" digits={0} />} foot={`${lift.toFixed(2)}× what random review finds`} onClick={() => nav("/queue")} />
        <Kpi label="Hit rate when reviewing" icon={<Crosshair size={14} />} value={pct(ch.precision_at_budget)} info="precision_budget" delay={0.1}
          delta={<Delta value={(ch.precision_at_budget - s.base_rate) * 100} suffix=" pts" digits={0} />} foot={`vs ${pct(s.base_rate)} if picked at random`} />
        <Kpi label="Safe to defer" icon={<Clock size={14} />} value={pct(ch.safe_deprioritised)} info="safe" delay={0.15}
          foot={`while keeping ${pct(ch.safe_recall)} of reportable alerts`} />
      </div>

      {/* ledger + bar + donut */}
      <div className="grid gap-4 xl:grid-cols-[300px_1fr_330px]">
        <Card title="Signal ledger" info="prauc" explain="Headline numbers for the model now running the queue." delay={0.1}>
          <div className="space-y-2">
            {[
              ["Ranking quality (PR-AUC)", f3(ch.pr_auc), <Delta key="a" value={ch.pr_auc - s.base_rate} raw={`+${(ch.pr_auc - s.base_rate).toFixed(3)}`} />, "prauc"],
              ["Lift over random", `${ch.lift.toFixed(2)}×`, <Delta key="b" value={ch.lift - 1} raw={`+${((ch.lift - 1) * 100).toFixed(0)}%`} />, "lift"],
              ["ROC-AUC", f3(ch.roc_auc), <Delta key="c" value={ch.roc_auc - 0.5} raw={`+${(ch.roc_auc - 0.5).toFixed(3)}`} />, "rocauc"],
              ["Best ensemble (test)", f3(s.best_test_row.pr_auc),
                <Badge key="d" tone={s.best_test_row.verdict_vs_pbl1 === "better" ? "lime" : "amber"}>{s.best_test_row.verdict_vs_pbl1}</Badge>, "paired"],
              ["Brier score", f3(ch.brier), null, "brier"],
            ].map(([k, v, d, t]: any) => (
              <div key={k} className="inset flex items-center justify-between px-3 py-2.5">
                <div>
                  <div className="flex items-center gap-1.5 text-[12px] text-soft"><span className="h-2 w-2 rounded-sm bg-lime" />{k}<InfoTip term={t} size={12} /></div>
                  <div className="mt-0.5 text-[18px] font-semibold num">{v}</div>
                </div>
                {d}
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11.5px] leading-snug text-mute">Running the queue: <b className="text-soft">{s.champion}</b> ({s.champion_phase}). Challenger {s.challenger} {s.promoted ? "promoted" : "not promoted"} <InfoTip term="champion" size={11} /><DS><> · 95% CI {ci(ch.pr_auc_ci)}</></DS></p>
        </Card>

        <Card delay={0.15}
          title={<span className="flex items-center gap-2.5"><span className="text-[26px] font-semibold num">{pct(top10.rate)}</span><Delta value={(top10.rate / s.base_rate - 1) * 100} suffix="%" digits={0} /></span>}
          action={<Segmented size="sm" value={metric} onChange={setMetric} options={[{ value: "rate", label: "Reported share" }, { value: "lift", label: "Lift" }]} />}
          explain={<>Alerts split into ten equal slices by risk. The top slice is reported <b className="text-white">{(top10.rate / s.base_rate).toFixed(1)}×</b> as often as average — the model is putting the right alerts first.</>}>
          <div className="h-[290px]">
            <ResponsiveContainer>
              <BarChart data={decData} margin={{ top: 28, right: 6, left: -12, bottom: 0 }}>
                <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 10.5 }} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => (metric === "rate" ? pct(v) : `${v.toFixed(1)}×`)} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => (metric === "rate" ? pct(v, 1) : `${v.toFixed(2)}×`)} />} />
                <ReferenceLine y={metric === "rate" ? s.base_rate : 1} stroke={C.red} strokeDasharray="4 4" label={{ value: "random", fill: C.red, fontSize: 10, position: "insideTopRight" }} />
                <Bar dataKey={metric} name={metric === "rate" ? "Reported" : "Lift"} radius={[10, 10, 10, 10]} maxBarSize={52}>
                  {decData.map((d: any, i: number) => <Cell key={i} fill={i === 0 ? C.lime : "#24272D"} stroke={i === 0 ? C.lime : "#2F333A"} />)}
                  <LabelList dataKey={metric} position="top" content={({ x, y, width, index, value }: any) => index === 0 ? (
                    <g><rect x={x + width / 2 - 26} y={y - 24} width="52" height="18" rx="9" fill={C.lime} /><text x={x + width / 2} y={y - 11} textAnchor="middle" fontSize="10.5" fontWeight="700" fill="#08090B">{metric === "rate" ? pct(value) : `${value.toFixed(1)}×`}</text></g>) : null} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Queue by priority" info="band" explain="How the unseen alerts split, and how often each band was really reported." delay={0.2}>
          <div className="flex items-center gap-3">
            <div className="relative h-[180px] w-[180px] shrink-0">
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={bands} dataKey="value" innerRadius={58} outerRadius={84} paddingAngle={3} stroke="none" startAngle={90} endAngle={-270}>
                    {bands.map((b) => <Cell key={b.name} fill={bandColor[b.name]} />)}
                  </Pie>
                  <Tooltip content={<ChartTip fmt={(v: number) => num(v)} />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
                <div><div className="text-[22px] font-semibold num">{num(s.n_test)}</div><div className="text-[11px] text-mute">alerts</div></div>
              </div>
            </div>
            <div className="flex-1 space-y-2.5">
              {bands.map((b) => (
                <div key={b.name}>
                  <div className="flex items-center gap-2 text-[12.5px]"><span className="h-2.5 w-2.5 rounded-full" style={{ background: bandColor[b.name] }} />{b.name}</div>
                  <div className="ml-[18px] text-[11.5px] text-mute num">{num(b.value)} · {pct(b.rate)} reported</div>
                </div>
              ))}
            </div>
          </div>
          <div className="inset mt-4 p-3 text-[12.5px] leading-relaxed text-soft">
            Alerts in <b className="text-white">Review now</b> are reported <b className="text-lime">{(s.band_rates["Review now"] / s.band_rates["Low priority"]).toFixed(1)}×</b> as often as <b className="text-white">Low priority</b> ones.
          </div>
        </Card>
      </div>

      {/* insight + heatmap + ask */}
      <div className="grid gap-4 xl:grid-cols-[1fr_1.25fr_1fr]">
        <InsightCard kicker="Insights" progress={ch.recall_at_budget}
          foot={`${num(s.caught)} of ${num(s.n_reported)} reportable alerts found after opening ${num(s.reviewed)}.`}>
          Reviewing the top <Hl>{pct(s.budget)}</Hl> of the queue finds {pct(ch.recall_at_budget)} of all reportable alerts — {lift.toFixed(1)}× what random review would.
        </InsightCard>

        <Card title="Does the sequence view agree?" info="hmm" delay={0.15}
          explain="Each row is a behavioural regime from the sequence model; columns are risk slices. Brighter = reported more often.">
          <Heat s={s} />
        </Card>

        <Card title="Ask about this data" delay={0.2} explain="Answers use the figures on this page." action={<MessageSquareText size={16} className="text-mute" />}>
          <div className="min-h-[150px]">
            <AnimatePresence mode="wait">
              {askOpen != null && (
                <motion.div key={askOpen} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="inset p-3.5 text-[13px] leading-relaxed text-soft">
                  <div className="mb-1.5 text-[12px] font-semibold text-lime">{questions[askOpen].q}</div>{questions[askOpen].a}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {questions.map((q, i) => (
              <button key={i} onClick={() => setAskOpen(i)}
                className={clsx("rounded-full border px-3 py-1.5 text-[12px] transition", askOpen === i ? "border-lime/50 bg-lime/10 text-lime" : "border-white/10 text-soft hover:border-white/25")}>{q.q}</button>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-2 rounded-xl border border-white/[0.07] bg-ink-900 px-3 py-2 text-[12.5px] text-mute">
            <span className="flex-1">Pick a question above</span><span className="grid h-7 w-7 place-items-center rounded-lg bg-lime text-ink-950"><Send size={13} /></span>
          </div>
        </Card>
      </div>

      {/* top alerts + journey */}
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Card title="Highest-risk alerts right now" info="risk" explain="Open any alert to see the behaviours behind its score." delay={0.1}
          action={<button onClick={() => nav("/queue")} className="flex items-center gap-1 text-[12.5px] text-lime">Full queue <ArrowRight size={14} /></button>}>
          <div className="divide-y divide-white/[0.05]">
            {s.top.map((a: any) => (
              <button key={a.id} onClick={() => nav(`/alert/${a.id}`)} className="flex w-full items-center gap-4 py-2.5 text-left transition hover:bg-white/[0.02]">
                <div className="w-9 text-[12px] text-mute num">#{a.rank}</div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 text-[13.5px] font-medium">Alert {a.id}<Badge tone={bandTone(a.band)}>{a.band}</Badge></div>
                  <div className="truncate text-[12px] text-mute">{a.reason}</div>
                </div>
                <div className="hidden h-8 w-24 sm:block"><Spark data={a.spark} /></div>
                <Badge tone={agreeTone(a.agree)}>{a.agree.replace("Models ", "")}</Badge>
                <div className="w-14 text-right text-[16px] font-semibold num">{pct(a.risk)}</div>
              </button>
            ))}
          </div>
        </Card>

        <Card title="Three reviews, one question" info="ci" delay={0.15}
          explain="The best model from each review (picked on early-2021 data), scored on the same unseen alerts. Bars show the plausible range.">
          <div className="space-y-3">
            {s.phases.map((p: any, i: number) => (
              <div key={p.phase} className="inset p-3.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2"><PhaseTag phase={p.phase} /><span className="text-[13.5px] font-medium">{p.model}</span></div>
                  <span className="text-[18px] font-semibold num" style={{ color: PHASE_COLOR[p.phase] }}>{f3(p.pr_auc)}</span>
                </div>
                <div className="relative mt-3 h-2 rounded-full bg-ink-700">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${(p.pr_auc / phaseMax) * 100}%` }} transition={{ duration: 0.8, delay: i * 0.12 }}
                    className="absolute inset-y-0 left-0 rounded-full" style={{ background: PHASE_COLOR[p.phase] }} />
                  <div className="absolute -top-1 h-4 border-x-2 border-white/70" style={{ left: `${(p.pr_auc_ci[0] / phaseMax) * 100}%`, width: `${((p.pr_auc_ci[1] - p.pr_auc_ci[0]) / phaseMax) * 100}%` }} />
                  <div className="absolute -top-1.5 h-5 w-[2px] bg-danger" style={{ left: `${(s.base_rate / phaseMax) * 100}%` }} title="random" />
                </div>
                <div className="mt-2 flex justify-between text-[11.5px] text-mute">
                  <span>{["Classic classifiers", "+ behavioural profiles & sequences", "Ensembles & stacking"][i]}</span>
                  <span className="num">catches {pct(p.recall_at_budget)} @ 20%{p.phase !== "PBL-1" && <> · {p.verdict_vs_pbl1 === "better" ? "better" : p.verdict_vs_pbl1 === "worse" ? "worse" : "tied"} vs PBL-1</>}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-2 text-[12px] text-mute"><ShieldCheck size={14} className="text-lime" />Same training year, same test quarters, same bootstrap for every number.</div>
        </Card>
      </div>
    </div>
  );
}

function Heat({ s }: { s: any }) {
  const cells = s.heat.flatMap((r: any) => r.cells.map((c: any) => c.rate)).filter((v: any) => v != null);
  const max = Math.max(...cells), min = Math.min(...cells);
  const [hover, setHover] = useState<{ r: string; d: number; c: any } | null>(null);
  return (
    <div>
      <div className="mb-2 flex items-center justify-end gap-1.5 text-[10.5px] text-mute">Low {[0.15, 0.35, 0.55, 0.8, 1].map((o) => <span key={o} className="h-2.5 w-2.5 rounded-sm" style={{ background: `rgba(182,242,74,${o})` }} />)} High</div>
      <div className="space-y-1.5">
        {s.heat.map((r: any) => (
          <div key={r.regime} className="flex items-center gap-2">
            <div className="w-10 shrink-0 text-[11px] font-semibold text-soft" title={r.regime}>{r.regime.split(" · ")[0]}</div>
            <div className="grid flex-1 grid-cols-10 gap-1">
              {r.cells.map((c: any, d: number) => {
                const o = c.rate == null ? 0 : 0.12 + 0.88 * ((c.rate - min) / (max - min || 1));
                return <div key={d} onMouseEnter={() => setHover({ r: r.regime, d, c })} onMouseLeave={() => setHover(null)}
                  className={clsx("h-6 rounded-[5px] transition", c.rate == null ? "hatch bg-ink-800" : "", hover?.r === r.regime && hover?.d === d && "ring-2 ring-white")}
                  style={c.rate != null ? { background: `rgba(182,242,74,${o})` } : undefined} />;
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="ml-12 mt-1.5 flex justify-between text-[10.5px] text-mute"><span>highest risk</span><span>lowest risk</span></div>
      <div className="inset mt-3 grid grid-cols-3 divide-x divide-white/5 px-1 py-2 text-[11.5px]">
        {hover ? (
          <><div className="px-2"><div className="text-mute">Regime</div><div className="truncate font-semibold">{hover.r}</div></div>
            <div className="px-2"><div className="text-mute">Risk slice</div><div className="font-semibold">{hover.d === 0 ? "Top 10%" : `${hover.d * 10}–${hover.d * 10 + 10}%`}</div></div>
            <div className="px-2"><div className="text-mute">Reported</div><div className="font-semibold">{hover.c.rate == null ? `too few (${hover.c.n})` : `${pct(hover.c.rate)} of ${hover.c.n}`}</div></div></>
        ) : (
          <><div className="px-2"><div className="text-mute">Read across</div><div className="font-semibold">Fades left → right</div></div>
            <div className="px-2"><div className="text-mute">Read down</div><div className="font-semibold">Regimes differ less</div></div>
            <div className="px-2"><div className="text-mute">Hover</div><div className="font-semibold">for exact rates</div></div></>
        )}
      </div>
    </div>
  );
}
