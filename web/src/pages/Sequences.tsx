import clsx from "clsx";
import { motion } from "framer-motion";
import { ArrowRight, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useNavigate } from "react-router-dom";
import { Badge, C, Card, ChartTip, InfoTip, PageLoading, REGIME, Segmented, Stat, Takeaway } from "../components/ui";
import { bucketOf, useData } from "../lib/data";
import { f3, num, pct } from "../lib/fmt";
import { DS } from "../lib/mode";

const TABS = [{ value: "regimes", label: "Regimes explained" }, { value: "risk", label: "Which regimes are riskier" }, { value: "moves", label: "How clients move" },
  { value: "client", label: "One client's year" }, { value: "score", label: "Suspicion score" }, { value: "choice", label: "Model choice" }];
const CH = ["cash", "card", "wire", "international"], SZ = ["small", "medium", "large"], GP = ["within the hour", "on the same day", "a day or more apart"];

export default function Sequences() {
  const { data: s } = useData<any>("sequences.json");
  const [tab, setTab] = useState("regimes");
  if (!s) return <PageLoading />;
  const r21 = s.regime_risk.filter((r: any) => r.period === "2021");
  const hiK = r21.reduce((a: any, b: any) => (b.rate > a.rate ? b : a)).k, loK = r21.reduce((a: any, b: any) => (b.rate < a.rate ? b : a)).k;
  const stay = s.A.reduce((a: number, row: number[], i: number) => a + row[i], 0) / s.K;
  const llr = s.standalone[0];
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Hidden regimes" value={s.K} info="hmm" />
        <Stat label="Transactions read" value={`${(s.n_transactions / 1e6).toFixed(1)}M`} />
        <Stat label="Stay in same regime, next transaction" value={pct(stay, 1)} />
        <Stat label="Suspicion score alone (PR-AUC)" value={f3(llr.pr_auc)} info="llr" tone={C.amber} />
      </div>
      <Takeaway points={[
        <>Reading each client's transactions <i>in order</i>, the model finds <b className="text-white">{s.K} recurring behaviour regimes</b> — like ‘rapid small card debits’ or ‘spread-out large wire transfers’.</>,
        <><b className="text-white">{s.regime_names[hiK]}</b> is the riskiest regime ({pct(r21.find((r: any) => r.k === hiK).rate)} reported when it dominates a client's year) and <b className="text-white">{s.regime_names[loK]}</b> the calmest ({pct(r21.find((r: any) => r.k === loK).rate)}).</>,
        <>On its own, the sequence suspicion score beats random ({f3(llr.pr_auc)} vs {f3(s.base_rate_test)}) but is weaker than the summary-based models, and adds nothing on top of them.</>,
      ]} />
      <div className="sticky top-[86px] z-20 overflow-x-auto py-1"><Segmented value={tab} onChange={setTab} options={TABS} /></div>
      {tab === "regimes" && <Regimes s={s} r21={r21} />}
      {tab === "risk" && <Risk s={s} />}
      {tab === "moves" && <Moves s={s} />}
      {tab === "client" && <Client s={s} />}
      {tab === "score" && <Score s={s} />}
      {tab === "choice" && <Choice s={s} />}
    </div>
  );
}

function describe(st: any) {
  const m = st.marginals;
  const am = (a: number[]) => a.indexOf(Math.max(...a));
  const c = am(m.channel), z = am(m.size), g = am(m.gap);
  return <>Mostly <b className="text-white">{CH[c]}</b> transactions ({pct(m.channel[c])}), usually <b className="text-white">{SZ[z]}</b> amounts ({pct(m.size[z])}), typically <b className="text-white">{GP[g]}</b> ({pct(m.gap[g])}). {pct(m.direction[0])} are money coming in.</>;
}

function Regimes({ s, r21 }: any) {
  const [k, setK] = useState(0);
  const st = s.states[k];
  const overall = s.overall_2021;
  const rows: any[] = [];
  ([["Channel", ["Cash", "Card", "Wire", "International"], "channel"], ["Amount", ["Small", "Medium", "Large"], "size"], ["Timing", ["< 1 hour apart", "Same day", "≥ 1 day apart"], "gap"], ["Direction", ["Credit (in)", "Debit (out)"], "direction"]] as const)
    .forEach(([grp, labels, key]) => labels.forEach((l, i) => rows.push({ name: `${grp}: ${l}`, regime: st.marginals[key][i], all: st.marginals[key][i] / Math.max(st.lifts[key][i], 1e-9) })));
  const rate = r21.find((r: any) => r.k === k);
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {s.states.map((x: any, i: number) => (
          <button key={i} onClick={() => setK(i)} className={clsx("card-glass p-4 text-left transition", k === i ? "ring-1" : "opacity-75 hover:opacity-100")} style={k === i ? { boxShadow: `0 0 0 1px ${REGIME[i]}` } : undefined}>
            <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-sm" style={{ background: REGIME[i] }} /><span className="text-[12px] font-bold">R{i + 1}</span></div>
            <div className="mt-1.5 text-[13px] font-semibold leading-snug">{x.name}</div>
            <div className="mt-2 text-[11.5px] text-mute">reported {pct(r21.find((r: any) => r.k === i).rate)} · stays ~{Math.round(x.expected_duration)} txns</div>
          </button>
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_1.4fr]">
        <Card glass>
          <div className="flex items-center gap-2"><span className="h-3.5 w-3.5 rounded" style={{ background: REGIME[k] }} /><span className="text-[17px] font-semibold">{s.regime_names[k]}</span></div>
          <p className="mt-3 text-[14px] leading-relaxed text-soft">{describe(st)}</p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Stat label="Typical stay" value={`${Math.round(st.expected_duration)} txns`} />
            <Stat label="Reported when dominant (2021)" value={pct(rate.rate)} tone={rate.rate > overall ? C.amber : C.azure} />
          </div>
          <div className="mt-4 text-[12px] text-mute">Most typical transactions</div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">{st.top.slice(0, 4).map((t: any) => <Badge key={t.symbol} tone="gray">{t.symbol} · {pct(t.p)}</Badge>)}</div>
          <div className="mt-3 text-[12px] text-mute">Unusually common here</div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">{st.distinctive.map((t: any) => <Badge key={t.symbol} tone="lime">{t.symbol} · {t.lift.toFixed(1)}×</Badge>)}</div>
        </Card>
        <Card title={`R${k + 1} compared with all transactions`} explain="Coloured bar = share within this regime; grey = share among all transactions. Where colour is longer, the regime over-represents that kind of transaction.">
          <div className="h-[400px]">
            <ResponsiveContainer>
              <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" tickFormatter={(v) => pct(v)} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={170} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} />
                <Bar dataKey="all" name="all transactions" fill="#3A3E47" radius={3} barSize={7} />
                <Bar dataKey="regime" name={`R${k + 1}`} fill={REGIME[k]} radius={3} barSize={7} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Risk({ s }: any) {
  const data = s.states.map((_: any, k: number) => {
    const a = s.regime_risk.find((r: any) => r.k === k && r.period === "2020"), b = s.regime_risk.find((r: any) => r.k === k && r.period === "2021");
    return { name: `R${k + 1}`, full: s.regime_names[k], y20: a.rate, y21: b.rate, e20: [a.rate - a.ci[0], a.ci[1] - a.rate], e21: [b.rate - b.ci[0], b.ci[1] - b.rate], n: b.n };
  });
  return (
    <div className="grid gap-4 xl:grid-cols-[1.6fr_1fr]">
      <Card title="Share reported, by the regime that dominates an alert's year" explain="Blue = 2020 (training year), lime = 2021 (new alerts). A pattern visible in both years is not a fluke. Dashed line = overall 2021 rate.">
        <div className="h-[360px]">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="name" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} domain={[0, "auto"]} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} labelFormatter={(l: any) => data.find((d: any) => d.name === l)?.full} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine y={s.overall_2021} stroke={C.red} strokeDasharray="4 4" />
              <Bar dataKey="y20" name="2020" fill={C.azure} radius={[6, 6, 0, 0]}><ErrorBar dataKey="e20" width={4} stroke="#C9CDD4" /></Bar>
              <Bar dataKey="y21" name="2021" fill={C.lime} radius={[6, 6, 0, 0]}><ErrorBar dataKey="e21" width={4} stroke="#F4F5F6" /></Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="All regimes at a glance">
        {data.map((d: any, k: number) => (
          <div key={k} className="flex items-center gap-3 border-b border-white/5 py-2.5 text-[12.5px] last:border-0">
            <span className="h-3 w-3 rounded-sm" style={{ background: REGIME[k] }} />
            <span className="flex-1">{d.full}</span>
            <span className="w-14 text-right num text-mute">{num(d.n)}</span>
            <span className={clsx("w-12 text-right font-semibold num", d.y21 > s.overall_2021 ? "text-amber" : "text-soft")}>{pct(d.y21)}</span>
          </div>
        ))}
        <p className="mt-2 text-[11.5px] text-mute">“Dominates” = the regime the client spent most transactions in. Counts are 2021 alerts.</p>
      </Card>
    </div>
  );
}

function Moves({ s }: any) {
  const K = s.K, A = s.A as number[][];
  const [hover, setHover] = useState<number | null>(null);
  const pos = Array.from({ length: K }, (_, i) => { const a = Math.PI / 2 - (2 * Math.PI * i) / K; return [250 + 170 * Math.cos(a), 220 - 170 * Math.sin(a)]; });
  const off = A.flatMap((r, i) => r.map((v, j) => (i === j ? 0 : v)));
  const mx = Math.max(...off);
  return (
    <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
      <Card title="Routes between regimes" explain="Circles are regimes. Arrows show where clients go when they leave one — thicker = more common. Hover a regime to isolate its routes.">
        <svg viewBox="0 0 500 440" className="mx-auto w-full max-w-[560px]">
          <defs>{REGIME.map((c, i) => <marker key={i} id={`ah${i}`} markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 Z" fill={c} /></marker>)}</defs>
          {A.map((row, i) => row.map((v, j) => {
            if (i === j || v < 0.04 * mx) return null;
            const [x1, y1] = pos[i], [x2, y2] = pos[j];
            const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
            const nx = -dy / L, ny = dx / L, bend = 26;
            const sx = x1 + (dx / L) * 44, sy = y1 + (dy / L) * 44, ex = x2 - (dx / L) * 48, ey = y2 - (dy / L) * 48;
            const cx = (sx + ex) / 2 + nx * bend, cy = (sy + ey) / 2 + ny * bend;
            const dim = hover != null && hover !== i;
            return <motion.path key={`${i}-${j}`} d={`M${sx},${sy} Q${cx},${cy} ${ex},${ey}`} fill="none" stroke={REGIME[i]} strokeWidth={1 + 7 * (v / mx)} markerEnd={`url(#ah${i})`}
              strokeOpacity={dim ? 0.08 : 0.75} strokeDasharray="8 6" animate={{ strokeDashoffset: [28, 0] }} transition={{ repeat: Infinity, duration: 1.6 - v / mx, ease: "linear" }}><title>{`R${i + 1} → R${j + 1}: ${pct(v, 2)} per transaction`}</title></motion.path>;
          }))}
          {pos.map(([x, y], i) => (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} className="cursor-pointer">
              <circle cx={x} cy={y} r={40} fill="#15171B" stroke={REGIME[i]} strokeWidth={hover === i ? 3 : 2} />
              <text x={x} y={y - 4} textAnchor="middle" fontSize="15" fontWeight="700" fill="#F4F5F6">R{i + 1}</text>
              <text x={x} y={y + 14} textAnchor="middle" fontSize="10" fill="#8D929C">stay {pct(A[i][i], 1)}</text>
            </g>
          ))}
        </svg>
      </Card>
      <div className="space-y-4">
        <Card title="Transition table" info="hmm" explain="Row = current regime, column = next transaction's regime.">
          <div className="grid gap-[3px]" style={{ gridTemplateColumns: `40px repeat(${K}, 1fr)` }}>
            <div />{A.map((_, j) => <div key={j} className="text-center text-[11px] text-mute">R{j + 1}</div>)}
            {A.map((row, i) => (
              <>
                <div key={`l${i}`} className="flex items-center text-[11px] font-semibold" style={{ color: REGIME[i] }}>R{i + 1}</div>
                {row.map((v, j) => <div key={`${i}${j}`} className="grid h-9 place-items-center rounded-md text-[10.5px] num" style={{ background: i === j ? "rgba(182,242,74,.85)" : `rgba(182,242,74,${Math.min(0.7, (v / mx) * 0.7 + 0.04)})`, color: i === j ? "#08090B" : "#C9CDD4" }}>{i === j ? pct(v, 1) : pct(v, 2)}</div>)}
              </>
            ))}
          </div>
        </Card>
        <Takeaway points={[
          <>Clients are creatures of habit: from one transaction to the next they stay in the same regime most of the time.</>,
          <>When they do switch, a few routes dominate (thick arrows), so movement is structured rather than random.</>,
        ]} />
      </div>
    </div>
  );
}

function Client({ s }: any) {
  const nav = useNavigate();
  const [id, setId] = useState<number>(s.examples[0]?.id);
  const { data: bucket } = useData<any>(id ? bucketOf(id) : null);
  const a = bucket?.[String(id)];
  const [t, setT] = useState(365);
  const [play, setPlay] = useState(false);
  useEffect(() => { setT(365); setPlay(false); }, [id]);
  useEffect(() => {
    if (!play) return;
    const h = setInterval(() => setT((v) => { if (v <= 0) { setPlay(false); return 0; } return v - 3; }), 30);
    return () => clearInterval(h);
  }, [play]);
  const segs = (a?.regimes ?? []) as number[][];
  const shown = segs.filter((r) => r[0] >= t);
  const counts = useMemo(() => { const c = Array(s.K).fill(0); shown.forEach((r) => (c[r[2]] += r[3])); return c; }, [shown, s.K]);
  const cur = shown.length ? shown[shown.length - 1][2] : null;
  return (
    <Card title="Replay one client's year" explain="Press play to watch the sequence model read a client's year, transaction by transaction. The strip fills in the regime it infers at each point.">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <select value={id} onChange={(e) => setId(+e.target.value)} className="rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-[13px]">
          {s.examples.map((e: any) => <option key={e.id} value={e.id}>Alert {e.id} · {e.switches} switches · rank #{e.rank}</option>)}
        </select>
        <button onClick={() => { if (t <= 0) setT(365); setPlay(!play); }} className="grid h-10 w-10 place-items-center rounded-full bg-lime text-ink-950">{play ? <Pause size={16} /> : <Play size={16} />}</button>
        <span className="text-[13px] text-mute num">{t >= 365 ? "a year before the alert" : t <= 0 ? "alert raised" : `${Math.round(t)} days before the alert`}</span>
        <button onClick={() => nav(`/alert/${id}`)} className="ml-auto flex items-center gap-1 text-[12.5px] text-lime">Open alert <ArrowRight size={14} /></button>
      </div>
      {a && (
        <>
          <div className="relative h-16 overflow-hidden rounded-xl bg-ink-800">
            {shown.map((r, i) => {
              const x0 = ((365 - r[0]) / 365) * 100, x1 = ((365 - r[1]) / 365) * 100;
              return <div key={i} className="absolute inset-y-0" style={{ left: `${x0}%`, width: `${Math.max(0.35, x1 - x0)}%`, background: REGIME[r[2]], opacity: 0.9 }} />;
            })}
            <div className="absolute inset-y-0 w-[2px] bg-white" style={{ left: `${((365 - t) / 365) * 100}%` }} />
          </div>
          <div className="mt-1 flex justify-between text-[10.5px] text-mute"><span>a year before</span><span>alert</span></div>
          <div className="mt-4 grid gap-3 sm:grid-cols-5">
            {counts.map((c, k) => (
              <div key={k} className={clsx("inset p-3 transition", cur === k && "ring-1")} style={cur === k ? { boxShadow: `0 0 0 1px ${REGIME[k]}` } : undefined}>
                <div className="flex items-center gap-1.5 text-[11.5px]"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: REGIME[k] }} />R{k + 1}</div>
                <div className="mt-1 text-[18px] font-semibold num">{num(c)}</div>
                <div className="truncate text-[10.5px] text-mute">{s.states[k].name}</div>
              </div>
            ))}
          </div>
          {a.hmm && <p className="mt-3 text-[12.5px] text-soft">This sequence looks more like past <b className="text-white">{a.hmm.llr > 0 ? "reported" : "dismissed"}</b> alerts than {pct(a.hmm.llr > 0 ? a.hmm.llr_pct : 1 - a.hmm.llr_pct)} of the queue. Outcome: <DS><b>{a.y ? "reported" : "dismissed"}</b></DS><span className="text-mute"> (switch to Data scientist mode to see)</span></p>}
        </>
      )}
    </Card>
  );
}

function Score({ s }: any) {
  const h = s.llr_hist;
  const tot = (a: number[]) => a.reduce((x, y) => x + y, 0);
  const tr = tot(h.report), td = tot(h.dismiss);
  const data = h.report.map((r: number, i: number) => ({ x: (h.bins[i] + h.bins[i + 1]) / 2, report: r / tr, dismiss: h.dismiss[i] / td }));
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="The sequence suspicion score" info="llr" explain="Two HMMs — one trained on reported alerts, one on dismissed — each say how likely a client's sequence is. Right of zero = looks more like a reported alert.">
        <div className="h-[320px]">
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={0}>
              <XAxis dataKey="x" tickLine={false} axisLine={false} tickFormatter={(v) => (+v).toFixed(2)} />
              <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} labelFormatter={(v: any) => `score ${(+v).toFixed(3)}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={data.reduce((a: any, b: any) => (Math.abs(b.x) < Math.abs(a.x) ? b : a)).x} stroke="#fff" strokeOpacity={0.4} />
              <Bar dataKey="dismiss" name="dismissed" fill="#4A4F58" />
              <Bar dataKey="report" name="reported" fill={C.lime} fillOpacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="Scores straight from the sequence model" explain="Each used alone to rank late-2021 alerts, with no classifier on top.">
        {s.standalone.map((r: any) => (
          <div key={r.score} className="flex items-center justify-between border-b border-white/5 py-2.5 text-[13px] last:border-0">
            <span>{r.score}</span><span className="flex items-center gap-3 num"><span className="text-mute">ROC {f3(r.roc_auc)}</span><b>{f3(r.pr_auc)}</b></span>
          </div>
        ))}
        <p className="mt-2 text-[12px] text-mute">Random ranking scores {f3(s.base_rate_test)}. Only the outcome-based score clearly beats it.</p>
      </Card>
    </div>
  );
}

function Choice({ s }: any) {
  const hist = s.history.map((v: number, i: number) => ({ it: i + 1, ll: v }));
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title="How many regimes?" explain="Fit measured on alerts the model didn't train on (higher is better). It stops improving after five.">
        <div className="h-[280px]">
          <ResponsiveContainer>
            <LineChart data={s.k_selection} margin={{ top: 8, right: 12, left: -2, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="K" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(4)} />} labelFormatter={(v: any) => `K = ${v}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={s.K} stroke="#fff" strokeOpacity={0.4} />
              <Line dataKey="val_ll_per_txn" name="held-out fit / transaction" stroke={C.lime} strokeWidth={2} />
              <Line dataKey="train_ll_per_txn" name="training fit / transaction" stroke={C.azure} strokeDasharray="4 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="Baum–Welch training" info="hmm" explain="Each iteration of expectation-maximisation can only improve the fit; it stops when the gain becomes negligible.">
        <div className="h-[280px]">
          <ResponsiveContainer>
            <LineChart data={hist} margin={{ top: 8, right: 12, left: -2, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="it" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(4)} />} labelFormatter={(v: any) => `iteration ${v}`} />
              <Line dataKey="ll" name="log-likelihood / transaction" stroke={C.lime} strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <DS><p className="text-[11.5px] text-mute">Scaled forward–backward, pseudo-counts, {s.n_symbols} symbols (direction × channel × size tertile × gap), first/last-hour edge piles removed. Hand-written in hmm.py, verified against brute-force enumeration in tests.</p></DS>
      </Card>
    </div>
  );
}
