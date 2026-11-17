import clsx from "clsx";
import { ChevronLeft, ChevronRight, Eye } from "lucide-react";
import { useMemo, useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { useNavigate, useParams } from "react-router-dom";
import { Badge, C, Card, ChartTip, Gauge, InfoTip, PageLoading, REGIME, Segmented, Stat, Takeaway, agreeTone, bandTone } from "../components/ui";
import { bucketOf, useData } from "../lib/data";
import { f2, num, pct, short } from "../lib/fmt";
import { DS, useMode } from "../lib/mode";

export default function AlertDetail() {
  const { id: idParam } = useParams();
  const nav = useNavigate();
  const { mode } = useMode();
  const { data: s } = useData<any>("summary.json");
  const { data: queue } = useData<any[]>("queue.json");
  const id = idParam ? parseInt(idParam, 10) : s?.top?.[0]?.id;
  const { data: bucket } = useData<any>(id ? bucketOf(id) : null);
  const { data: seq } = useData<any>("sequences.json");
  const { data: clu } = useData<any>("clusters.json");
  const [view, setView] = useState<"ensemble" | "transparent">("ensemble");
  const [reveal, setReveal] = useState(false);
  const a = bucket?.[String(id)];
  const byRank = useMemo(() => (queue ? Object.fromEntries(queue.map((r) => [r.rank, r.id])) : {}), [queue]);

  if (!s || !queue || !a) return <PageLoading />;
  const go = (rank: number) => byRank[rank] && nav(`/alert/${byRank[rank]}`);
  const up = a.shap.top.filter((t: any) => t.v > 0), down = a.shap.top.filter((t: any) => t.v < 0);
  const share = a.rank / a.n;
  const ratio = a.risk / s.base_rate;

  // waterfall in log-odds: base -> each factor -> rest -> final
  let run = a.shap.base;
  const wf = [{ name: "Average alert", lo: Math.min(0, run), hi: Math.max(0, run), v: run, kind: "base" }];
  [...a.shap.top].sort((x: any, y: any) => Math.abs(y.v) - Math.abs(x.v)).forEach((t: any) => {
    const nx = run + t.v; wf.push({ name: short(t.label, 30), lo: Math.min(run, nx), hi: Math.max(run, nx), v: t.v, kind: t.v > 0 ? "up" : "down" }); run = nx;
  });
  const nx = run + a.shap.rest; wf.push({ name: "All other behaviours", lo: Math.min(run, nx), hi: Math.max(run, nx), v: a.shap.rest, kind: a.shap.rest > 0 ? "up" : "down" }); run = nx;
  wf.push({ name: "This alert", lo: Math.min(0, run), hi: Math.max(0, run), v: run, kind: "final" });
  const wfData = wf.map((d) => ({ ...d, range: [d.lo, d.hi] }));
  const lrData = [...a.lr].reverse();

  const weeks = [...a.weekly].reverse().map((n: number, i: number) => ({ w: i - 51, n }));
  const regimes = a.regimes as number[][] | undefined;

  return (
    <div className="space-y-5">
      {/* header */}
      <div className="card flex flex-wrap items-center gap-4 px-5 py-4">
        <div className="flex items-center gap-2">
          <button onClick={() => go(a.rank - 1)} disabled={a.rank <= 1} className="grid h-9 w-9 place-items-center rounded-xl border border-white/10 disabled:opacity-30" title="Higher-ranked alert"><ChevronLeft size={16} /></button>
          <button onClick={() => go(a.rank + 1)} disabled={a.rank >= a.n} className="grid h-9 w-9 place-items-center rounded-xl border border-white/10 disabled:opacity-30" title="Lower-ranked alert"><ChevronRight size={16} /></button>
        </div>
        <div>
          <div className="text-[20px] font-semibold tracking-tight">Alert {a.id}</div>
          <div className="text-[12.5px] text-mute">Raised in {a.quarter} · ranked <b className="text-white">#{num(a.rank)}</b> of {num(a.n)}</div>
        </div>
        <Badge tone={bandTone(a.band)}>{a.band}</Badge>
        <Badge tone={agreeTone(a.agreement)}>{a.agreement}</Badge>
        <div className="ml-auto flex items-center gap-3">
          <Segmented size="sm" value={view} onChange={setView} options={[{ value: "ensemble", label: "Ensemble reasons" }, { value: "transparent", label: "Transparent model" }]} />
          <button onClick={() => setReveal(!reveal)} className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-[12.5px] text-soft hover:border-white/25"><Eye size={14} />{reveal ? (a.y ? "Reported" : "Dismissed") : "Reveal outcome"}</button>
        </div>
      </div>

      <Takeaway title="In plain words" points={[
        <>This alert is in the <b className="text-white">top {Math.max(1, Math.round(share * 100))}%</b> of the queue with a <b className="text-white">{pct(a.risk)}</b> chance of being reported — {ratio >= 1 ? `${ratio.toFixed(1)}× the average alert` : `below the ${pct(s.base_rate)} average`}.</>,
        up.length ? <>Pushing it up: <b className="text-white">{up.slice(0, 3).map((t: any) => t.label.toLowerCase()).join("; ")}</b>.</> : <>No single behaviour pushes it up strongly.</>,
        down.length ? <>Holding it down: {down.slice(0, 2).map((t: any) => t.label.toLowerCase()).join("; ")}.</> : null,
        <>{a.agreement === "Models agree" ? "All six models broadly agree on this score, so it can be taken at face value." : a.agreement === "Mixed" ? "The models partly disagree; treat the score as a guide." : "The models disagree strongly — use judgement, not the score alone."}</>,
      ].filter(Boolean)} />

      <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
        <Card title="Risk" info="risk" explain="Where this alert sits between ‘almost surely dismissed’ and ‘very likely reported’.">
          <Gauge value={Math.min(1, a.risk / 0.6)} label={pct(a.risk)} sub={<>Bayesian model's range: {pct(a.bayes.lo)}–{pct(a.bayes.hi)} <InfoTip term="bayes_band" size={12} /></>}
            low="Dismiss" high="Report" />
          <div className="mt-4 text-[12px] font-medium text-soft">How each model voted <InfoTip term="agreement" size={12} /></div>
          <div className="relative mt-3 h-[150px]">
            {a.votes.map((v: any, i: number) => (
              <div key={v.model} className="absolute left-0 right-0 flex items-center gap-2" style={{ top: i * 24 }}>
                <span className="w-[92px] truncate text-[11px] text-mute">{v.model.replace(" (profiles + sequences)", "").replace("Logistic Regression", "Logistic reg.")}</span>
                <div className="relative h-[3px] flex-1 rounded bg-ink-700">
                  <div className="absolute -top-[5px] h-[13px] w-[13px] -translate-x-1/2 rounded-full border-2 border-ink-950" style={{ left: `${Math.min(100, (v.p / 0.6) * 100)}%`, background: i >= 3 ? C.lime : i === 5 ? C.violet : C.azure }} />
                </div>
                <span className="w-9 text-right text-[11px] num">{pct(v.p)}</span>
              </div>
            ))}
          </div>
          <DS><p className="text-[11px] text-mute">Rank spread across models (sd of percentile) {f2(a.committee_sd)} · dots on a 0–60% risk scale · Bayesian LR 95% interval on the logit.</p></DS>
        </Card>

        {view === "ensemble" ? (
          <Card title="What drove the score" info="shap"
            explain={<>Start from an average alert (left); each bar is one behaviour pushing risk up (lime) or down (red). The last bar is this alert. <span className="text-mute/80">Reasons come from the boosted-tree model (exact SHAP); it orders alerts much like the {s.champion} that ranks the queue (rank agreement {s.explainer_rank_corr.toFixed(2)}).</span></>}>
            <div className="h-[360px]">
              <ResponsiveContainer>
                <BarChart data={wfData} margin={{ top: 8, right: 10, left: 0, bottom: 70 }}>
                  <XAxis dataKey="name" interval={0} angle={-28} textAnchor="end" height={80} tickLine={false} axisLine={false} tick={{ fontSize: 10.5 }} />
                  <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(1)} label={mode === "scientist" ? { value: "log-odds", angle: -90, position: "insideLeft", fill: C.mute, fontSize: 10 } : undefined} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,.15)" />
                  <Tooltip cursor={false} content={({ active, payload }: any) => active && payload?.length ? (
                    <div className="rounded-xl border border-white/10 bg-ink-800/95 px-3 py-2 text-[12px]"><div className="font-semibold">{payload[0].payload.name}</div>
                      <div className="text-mute">{payload[0].payload.kind === "base" || payload[0].payload.kind === "final" ? `score ${payload[0].payload.v.toFixed(2)}` : `${payload[0].payload.v > 0 ? "+" : ""}${payload[0].payload.v.toFixed(3)} log-odds`}</div></div>) : null} />
                  <Bar dataKey="range" radius={[4, 4, 4, 4]}>
                    {wfData.map((d, i) => <Cell key={i} fill={d.kind === "up" ? C.lime : d.kind === "down" ? C.red : d.kind === "final" ? "#F4F5F6" : C.gray} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <DS>
              <div className="mt-1 grid grid-cols-2 gap-2 md:grid-cols-4">
                {a.shap.top.slice(0, 4).map((t: any) => <div key={t.f} className="inset px-3 py-2 text-[11.5px]"><div className="truncate text-mute">{t.label}</div><div className="num">value {t.x == null ? "missing" : t.x.toFixed(2)} · z {t.z == null ? "–" : t.z.toFixed(2)}</div></div>)}
              </div>
            </DS>
          </Card>
        ) : (
          <Card title="Reasons from the transparent model" info="lr_contrib" explain="The logistic regression's own reasons. They are exact, and usually agree with the ensemble on the big factors.">
            <div className="h-[360px]">
              <ResponsiveContainer>
                <BarChart data={lrData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <XAxis type="number" tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="label" width={230} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                  <ReferenceLine x={0} stroke="rgba(255,255,255,.2)" />
                  <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => `${v > 0 ? "+" : ""}${v.toFixed(3)}`} />} />
                  <Bar dataKey="v" name="contribution" radius={4}>{lrData.map((d: any, i: number) => <Cell key={i} fill={d.v > 0 ? C.azure : C.red} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Activity over the year" explain="Transactions per week in the year before the alert (right = just before the alert).">
          <div className="h-[200px]">
            <ResponsiveContainer>
              <BarChart data={weeks} margin={{ top: 6, right: 4, left: -22, bottom: 0 }}>
                <XAxis dataKey="w" tickLine={false} axisLine={false} tickFormatter={(v) => (v === 0 ? "alert" : `${v}w`)} interval={12} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => `${v} transactions`} />} labelFormatter={(v: any) => (v === 0 ? "Week before alert" : `${-v} weeks earlier`)} />
                <Bar dataKey="n" name="Transactions" radius={[3, 3, 0, 0]}>{weeks.map((d, i) => <Cell key={i} fill={i >= 48 ? C.lime : "#2C3036"} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 grid grid-cols-3 gap-3">
            <Stat label="Last 30 days" value={num(a.activity.actual)} />
            <Stat label="Expected" value={`${num(a.activity.expected)}`} />
            <Stat label="Normal range" value={`${num(a.activity.lo)}–${num(a.activity.hi)}`} tone={a.activity.actual > a.activity.hi ? C.amber : undefined} />
          </div>
          <p className="mt-2 text-[12px] text-mute">{a.activity.actual > a.activity.hi ? "Busier than expected from the client's own history." : a.activity.actual < a.activity.lo ? "Quieter than expected from the client's own history." : "Within the range expected from the client's own history."}</p>
        </Card>

        <Card title="Behavioural regimes over time" info="hmm" explain="The sequence model's reading of the client's year, transaction by transaction. Each colour is one hidden regime.">
          {regimes && seq ? (
            <>
              <RegimeStrip regimes={regimes} names={seq.regime_names} />
              <div className="mt-4 grid grid-cols-3 gap-3">
                <Stat label="Regime switches" value={num(regimes.length - 1)} />
                <Stat label="Mostly in" value={a.hmm ? `R${a.hmm.occ.indexOf(Math.max(...a.hmm.occ)) + 1}` : "–"} />
                <Stat label="Suspicion score" info="llr" value={a.hmm ? `top ${Math.max(1, Math.round((1 - a.hmm.llr_pct) * 100))}%` : "–"} />
              </div>
              {a.hmm && <p className="mt-2 text-[12px] text-mute">Its transaction sequence looks more like past <b className="text-white">{a.hmm.llr > 0 ? "reported" : "dismissed"}</b> alerts than {pct(a.hmm.llr > 0 ? a.hmm.llr_pct : 1 - a.hmm.llr_pct)} of the queue.</p>}
            </>
          ) : <p className="text-[13px] text-mute">Too few usable transactions to read a sequence for this alert.</p>}
        </Card>
      </div>

      <Card title="Which behavioural profile is it in?" info="gmm" explain="Every alert placed on a map of look-alikes (closer = more similar). This alert is the white star.">
        <div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
          <div className="h-[300px]">
            {clu && (
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 6, right: 6, left: -20, bottom: 0 }}>
                  <XAxis type="number" dataKey="x" tickLine={false} axisLine={false} tick={false} name="map axis 1" />
                  <YAxis type="number" dataKey="y" tickLine={false} axisLine={false} tick={false} name="map axis 2" />
                  <ZAxis range={[14, 14]} />
                  {[0, 1, 2].map((k) => <Scatter key={k} data={clu.scatter.filter((p: any) => p.k === k)} fill={[C.azure, C.violet, C.amber][k]} fillOpacity={0.35} isAnimationActive={false} />)}
                  <Scatter data={[{ x: a.cluster.pc1, y: a.cluster.pc2 }]} fill="#fff" shape="star" isAnimationActive={false}><ZAxis range={[260, 260]} /></Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
          <div className="space-y-3">
            <div className="inset p-3.5">
              <div className="text-[11.5px] text-mute">Broad profile</div>
              <div className="mt-0.5 text-[15px] font-semibold">P{a.cluster.kmeans + 1}</div>
              <div className="text-[12px] capitalize text-soft">{a.cluster.km_name}</div>
              <div className="mt-1 text-[12px] text-mute">{pct(a.cluster.km_rate)} of 2020 alerts like this were reported</div>
            </div>
            <div className="inset p-3.5">
              <div className="mb-2 text-[11.5px] text-mute">Fine-grained membership (soft) <InfoTip term="gmm" size={12} /></div>
              {a.cluster.gmm_top.map((g: any) => (
                <div key={g.k} className="mb-1.5 flex items-center gap-2 text-[12px]">
                  <span className="w-8">G{g.k + 1}</span>
                  <div className="h-2 flex-1 rounded-full bg-ink-700"><div className="h-full rounded-full bg-violet" style={{ width: `${g.p * 100}%` }} /></div>
                  <span className="w-10 text-right num">{pct(g.p)}</span>
                </div>
              ))}
            </div>
            <p className="text-[12px] text-mute">Profiles were found without looking at outcomes, so they describe <i>how</i> a client behaves, not whether they are suspicious.</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

function RegimeStrip({ regimes, names }: { regimes: number[][]; names: string[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const used = Array.from(new Set(regimes.map((r) => r[2]))).sort();
  // regimes: [start_day_before, end_day_before, state, n]; x = 365 - day (left = a year ago)
  return (
    <div>
      <div className="relative h-12 overflow-hidden rounded-xl bg-ink-800">
        {regimes.map((r, i) => {
          const x0 = ((365 - r[0]) / 365) * 100, x1 = ((365 - r[1]) / 365) * 100;
          return <div key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            className={clsx("absolute inset-y-0 transition", hover === i && "brightness-125")}
            style={{ left: `${x0}%`, width: `${Math.max(0.35, x1 - x0)}%`, background: REGIME[r[2]], opacity: hover == null || hover === i ? 0.9 : 0.45 }} />;
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10.5px] text-mute"><span>a year before</span><span>6 months</span><span>alert</span></div>
      <div className="mt-2 min-h-[20px] text-[12px] text-soft">
        {hover != null ? <>{names[regimes[hover][2]]} · {regimes[hover][3]} transactions · {Math.round(regimes[hover][0])}→{Math.round(regimes[hover][1])} days before</> :
          <span className="flex flex-wrap gap-x-3 gap-y-1">{used.map((k) => <span key={k} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: REGIME[k] }} />{names[k]}</span>)}</span>}
      </div>
    </div>
  );
}
