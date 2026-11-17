import clsx from "clsx";
import { Search } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, Cell, ComposedChart, ErrorBar, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, C, Card, ChartTip, InfoTip, PageLoading, Segmented, Stat, Takeaway } from "../components/ui";
import { useData } from "../lib/data";
import { compact, f3, num, pct, short } from "../lib/fmt";
import { DS } from "../lib/mode";

const TABS = [{ value: "structure", label: "Structure & quality" }, { value: "balance", label: "Class balance" }, { value: "separate", label: "What separates the classes" },
  { value: "bayes", label: "Evidence & belief" }, { value: "corr", label: "Relationships" }, { value: "drift", label: "Drift" }, { value: "dict", label: "Feature dictionary" }];

export default function DataInsights() {
  const { data: d } = useData<any>("data.json");
  const [tab, setTab] = useState("structure");
  if (!d) return <PageLoading />;
  const q = d.quality, e = d.eda;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Alerts" value={num(q.n_alerts)} />
        <Stat label="Transactions" value={compact(q.n_transactions)} />
        <Stat label="Reported" value={pct(q.report_rate, 1)} info="base_rate" tone={C.lime} />
        <Stat label="Transactions per alert (median)" value={num(q.txn_per_alert.median)} />
      </div>
      <div className="sticky top-[86px] z-20 overflow-x-auto py-1"><Segmented value={tab} onChange={setTab} options={TABS} /></div>
      {tab === "structure" && <Structure q={q} />}
      {tab === "balance" && <Balance e={e} />}
      {tab === "separate" && <Separate e={e} />}
      {tab === "bayes" && <Bayes e={e} />}
      {tab === "corr" && <Corr e={e} reg={d.registry} />}
      {tab === "drift" && <Drift e={e} />}
      {tab === "dict" && <Dict reg={d.registry} />}
    </div>
  );
}

function Structure({ q }: any) {
  const checks = [["Duplicate alert IDs", q.duplicate_alert_ids], ["Alerts with no transactions", q.alerts_without_transactions], ["Transactions with no alert", q.transactions_without_alert],
    ["Missing values (alerts)", Object.values(q.nulls_alerts).reduce((a: any, b: any) => a + b, 0)], ["Missing values (transactions)", Object.values(q.nulls_transactions).reduce((a: any, b: any) => a + b, 0)]];
  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <Card title="Two linked files" explain="SynthAML (Jensen et al., 2023): synthetic but modelled on a real bank's alerts.">
        <div className="space-y-3">
          <div className="inset p-3.5"><div className="text-[13px] font-semibold">Alerts</div><div className="text-[12px] text-mute">{num(q.n_alerts)} rows · alert ID, date (quarter), outcome (reported / dismissed)</div></div>
          <div className="text-center text-[11px] text-mute">joined by alert ID ↓</div>
          <div className="inset p-3.5"><div className="text-[13px] font-semibold">Transactions</div><div className="text-[12px] text-mute">{num(q.n_transactions)} rows · up to a year before each alert · direction, channel, standardised size, time</div></div>
        </div>
      </Card>
      <Card title="Split by time" info="temporal" explain="Models never see the future: they learn from 2020 and are judged on late 2021.">
        {[["Train", "2020 Q1–Q4", q.split.train, C.azure], ["Validation", "2021 Q1–Q2", q.split.valid, C.violet], ["Test", "2021 Q3–Q4", q.split.test, C.lime]].map(([r, p, n, c]: any) => (
          <div key={r} className="mb-3">
            <div className="flex justify-between text-[12.5px]"><span><b>{r}</b> <span className="text-mute">{p}</span></span><span className="num">{num(n)}</span></div>
            <div className="mt-1 h-2 rounded-full bg-ink-700"><div className="h-full rounded-full" style={{ width: `${(n / q.n_alerts) * 100}%`, background: c }} /></div>
          </div>
        ))}
        <div className="mt-2 grid h-[64px] grid-cols-8 items-end gap-1">{q.by_quarter.map((b: any) => <div key={b.quarter} className="text-center"><div className="mx-auto w-full rounded-t" style={{ height: 40 * (b.alerts / 3300), background: b.quarter.startsWith("2020") ? C.azure : b.quarter <= "2021Q2" ? C.violet : C.lime }} /><div className="mt-1 text-[9px] text-mute">{b.quarter.slice(2)}</div></div>)}</div>
      </Card>
      <Card title="Quality checks" explain="Everything that could silently break a model was checked before training.">
        {checks.map(([k, v]: any) => <div key={k} className="flex justify-between border-b border-white/5 py-2 text-[12.5px] last:border-0"><span>{k}</span><Badge tone={v === 0 ? "lime" : "red"}>{v === 0 ? "none" : num(v)}</Badge></div>)}
        <p className="mt-2 text-[11.5px] text-mute">Sizes arrive pre-standardised (mean {q.size_mean.toFixed(2)}, sd {q.size_std.toFixed(2)}), so no money amounts are ever shown.</p>
      </Card>
    </div>
  );
}

function Balance({ e }: any) {
  const data = e.balance.map((b: any) => ({ ...b, err: [b.rate - b.lo, b.hi - b.rate] }));
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="Share of alerts reported, per quarter (2020)" explain="Stable around 1 in 6 — so a model that never says ‘report’ would be 83% accurate and completely useless.">
        <div className="h-[300px]">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="quarter" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} domain={[0, 0.25]} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} />
              <Bar dataKey="rate" name="reported" fill={C.lime} radius={[8, 8, 8, 8]} barSize={60}><ErrorBar dataKey="err" width={6} stroke="#F4F5F6" /></Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="The accuracy trap" info="prauc" explain="What the ‘always dismiss’ model scores.">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Accuracy" value={pct(e.accuracy_trap.accuracy)} tone={C.lime} />
          <Stat label="Reports caught" value={pct(e.accuracy_trap.recall)} tone={C.red} />
          <Stat label="PR-AUC" value={f3(e.accuracy_trap.pr_auc)} />
          <Stat label="ROC-AUC" value={f3(e.accuracy_trap.roc_auc)} />
        </div>
        <p className="mt-3 text-[12.5px] text-soft">This is why every model in AdaptiveAML is judged on PR-AUC and on how many reportable alerts the top of the queue catches — never on accuracy.</p>
      </Card>
    </div>
  );
}

function Separate({ e }: any) {
  const feats = Object.keys(e.distributions);
  const [f, setF] = useState(feats[0]);
  const dist = e.distributions[f];
  const tr = dist.report.reduce((a: number, b: number) => a + b, 0), td = dist.dismiss.reduce((a: number, b: number) => a + b, 0);
  const data = dist.report.map((r: number, i: number) => ({ x: (dist.bins[i] + dist.bins[i + 1]) / 2, report: r / tr, dismiss: dist.dismiss[i] / td }));
  const top = e.mannwhitney.slice(0, 12).map((m: any) => ({ ...m, label: short(m.label, 30), abs: Math.abs(m.effect) }));
  const lab = e.mannwhitney.find((m: any) => m.feature === f)?.label ?? f;
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title="Biggest differences between reported and dismissed" explain="Effect size from a rank test on 2020 alerts. Wire and card size profiles dominate; activity counts barely differ.">
        <div className="h-[400px]">
          <ResponsiveContainer>
            <BarChart data={top} layout="vertical" margin={{ left: 10, right: 16 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="label" width={220} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number, p: any) => `${p.payload.effect > 0 ? "higher" : "lower"} in reported · ${v.toFixed(3)}`} />} />
              <Bar dataKey="abs" name="effect size" radius={4} barSize={12}>{top.map((m: any, i: number) => <Cell key={i} fill={m.effect > 0 ? C.lime : C.red} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11.5px] text-mute">Lime = higher in reported alerts; red = lower. <DS><>p-values Holm-adjusted across {e.mannwhitney.length} tests.</></DS></p>
      </Card>
      <Card title="One behaviour, two outcomes" action={<select value={f} onChange={(ev) => setF(ev.target.value)} className="rounded-lg border border-white/10 bg-ink-900 px-2 py-1 text-[12px]">{feats.map((k) => <option key={k} value={k}>{e.mannwhitney.find((m: any) => m.feature === k)?.label ?? k}</option>)}</select>}
        explain={`Distribution of “${lab}” for reported (lime) and dismissed (grey) alerts. The shift is real but the overlap is large — no single behaviour separates them.`}>
        <div className="h-[400px]">
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={0}>
              <XAxis dataKey="x" tickLine={false} axisLine={false} tickFormatter={(v) => (+v).toFixed(1)} />
              <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} labelFormatter={(v: any) => `value ≈ ${(+v).toFixed(2)}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="dismiss" name="dismissed" fill="#4A4F58" />
              <Bar dataKey="report" name="reported" fill={C.lime} fillOpacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

function Bayes({ e }: any) {
  const rows = e.bayes.slice(0, 10).map((b: any) => ({ ...b, label: short(b.label, 32) }));
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="How one observation changes our belief" explain={`Before looking at anything, an alert has a ${pct(e.prior)} chance of being reported (the prior). Learning that a behaviour is unusually high moves that belief (the posterior).`}>
        <div className="h-[380px]">
          <ResponsiveContainer>
            <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" tickFormatter={(v) => pct(v)} tickLine={false} axisLine={false} domain={[0, "auto"]} />
              <YAxis type="category" dataKey="label" width={230} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <ReferenceLine x={e.prior} stroke={C.red} strokeDasharray="4 4" label={{ value: "prior", fill: C.red, fontSize: 10, position: "top" }} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} />
              <Bar dataKey="posterior" name="P(report | behaviour is high)" fill={C.lime} radius={4} barSize={12} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="Bayes' rule, worked out" info="calibration">
        <div className="inset p-3 font-mono text-[12px] text-soft">P(report | high) = P(high | report) · P(report) / P(high)</div>
        {rows.slice(0, 3).map((b: any) => (
          <div key={b.feature} className="mt-3 text-[12.5px] text-soft">
            <b className="text-white">{b.label}</b> is high in {pct(b.likelihood)} of reported alerts and {pct(b.evidence)} of all alerts, so seeing it moves the chance from {pct(e.prior)} to <b className="text-lime">{pct(b.posterior)}</b>.
          </div>
        ))}
        <DS><p className="mt-3 text-[11.5px] text-mute">Fisher's discriminant on the best pair: criterion {e.fisher.criterion.toFixed(3)} — the classes overlap heavily along every single direction.</p></DS>
      </Card>
    </div>
  );
}

function Corr({ e, reg }: any) {
  const lab = (f: string) => reg.find((r: any) => r.feature === f)?.label ?? f;
  const n = e.corr.features.length;
  const [hover, setHover] = useState<[number, number] | null>(null);
  return (
    <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
      <Card title="Which behaviours move together?" explain="Rank correlation between pairs of behaviours (2020). Bright = strongly related. Many behaviours are near-duplicates, which is why compression (PCA) works well.">
        <div className="grid gap-[1px]" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
          {e.corr.matrix.flatMap((row: number[], i: number) => row.map((v, j) => (
            <div key={`${i}-${j}`} onMouseEnter={() => setHover([i, j])} className={clsx("aspect-square", hover && hover[0] === i && hover[1] === j && "ring-1 ring-white")}
              style={{ background: v >= 0 ? `rgba(182,242,74,${Math.abs(v)})` : `rgba(255,93,93,${Math.abs(v)})` }} />
          )))}
        </div>
        <div className="mt-2 min-h-[20px] text-[12px] text-soft">{hover ? <>{lab(e.corr.features[hover[0]])} × {lab(e.corr.features[hover[1]])}: <b className="num">{e.corr.matrix[hover[0]][hover[1]].toFixed(2)}</b></> : "Hover a cell"}</div>
      </Card>
      <Card title="Most strongly related pairs">
        {e.corr_top.map((p: any, i: number) => (
          <div key={i} className="flex items-center justify-between border-b border-white/5 py-2 text-[12px] last:border-0">
            <span className="text-soft">{lab(p.a)} <span className="text-mute">×</span> {lab(p.b)}</span><span className="num font-semibold">{p.rho.toFixed(2)}</span>
          </div>
        ))}
      </Card>
    </div>
  );
}

function Drift({ e }: any) {
  const rows = e.drift.slice(0, 14).map((d: any) => ({ ...d, label: short(d.label, 30), sig: d.p_adj < 0.05 }));
  const nSig = e.drift.filter((d: any) => d.p_adj < 0.05).length;
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="Has behaviour changed since training?" explain="Difference between 2020 and early-2021 alerts for each behaviour (Kolmogorov–Smirnov). Red bars changed measurably.">
        <div className="h-[420px]">
          <ResponsiveContainer>
            <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 16 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="label" width={220} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} />
              <Bar dataKey="stat" name="KS statistic" radius={4} barSize={12}>{rows.map((r: any, i: number) => <Cell key={i} fill={r.sig ? C.red : "#4A4F58"} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Takeaway points={[
        <><b className="text-white">{nSig} of {e.drift.length}</b> behaviours look measurably different in early 2021 than in 2020.</>,
        <>The biggest shifts are in history length and activity counts — exactly the features the models lean on least.</>,
        <>That is why the evaluation is time-based: a random split would hide this drift and flatter every model.</>,
      ]} />
    </div>
  );
}

function Dict({ reg }: any) {
  const [q, setQ] = useState("");
  const groups = Array.from(new Set(reg.map((r: any) => r.group))) as string[];
  const [g, setG] = useState("All");
  const rows = reg.filter((r: any) => (g === "All" || r.group === g) && (!q || `${r.label} ${r.feature}`.toLowerCase().includes(q.toLowerCase())));
  return (
    <Card title={`All ${reg.length} engineered behaviours`} explain="Every feature the models use is disclosed here, with a plain-language reason for including it."
      action={<div className="flex h-9 items-center gap-2 rounded-xl border border-white/10 bg-ink-900 px-3"><Search size={14} className="text-mute" /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" className="w-32 bg-transparent text-[12.5px] focus:outline-none" /></div>}>
      <div className="mb-3 flex flex-wrap gap-1.5">{["All", ...groups].map((x) => <button key={x} onClick={() => setG(x)} className={clsx("rounded-full border px-2.5 py-1 text-[11.5px]", g === x ? "border-lime/50 bg-lime/10 text-lime" : "border-white/10 text-mute")}>{x}</button>)}</div>
      <div className="max-h-[520px] overflow-auto">
        <table className="w-full text-[12.5px]"><thead className="sticky top-0 bg-ink-850 text-mute"><tr className="text-left"><th className="py-2">Behaviour</th><th>Group</th><th>Why it's included</th><th><DS><>Column</></DS></th></tr></thead>
          <tbody>{rows.map((r: any) => <tr key={r.feature} className="border-t border-white/5"><td className="py-2 pr-3">{r.label}</td><td className="pr-3"><Badge tone="gray">{r.group}</Badge></td><td className="pr-3 text-soft">{r.why}</td><td className="font-mono text-[11px] text-mute"><DS><>{r.feature}</></DS></td></tr>)}</tbody></table>
      </div>
    </Card>
  );
}
