import clsx from "clsx";
import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { Badge, C, Card, ChartTip, InfoTip, MeterRow, PageLoading, PHASE_COLOR, PhaseTag, SERIES, Segmented, Stat, Takeaway } from "../components/ui";
import { useData } from "../lib/data";
import { ci, f3, num, pct, short } from "../lib/fmt";
import { DS, useMode } from "../lib/mode";

const BASE_TABS = [
  { value: "board", label: "Leaderboard" }, { value: "curves", label: "Curves" }, { value: "stability", label: "Over time" },
  { value: "ablation", label: "Ablations" }, { value: "readable", label: "Readable models" }, { value: "bench", label: "vs Existing approach" },
  { value: "baseline", label: "Activity baseline" },
];

export default function Arena() {
  const { data: A } = useData<any>("arena.json");
  const { mode } = useMode();
  const [tab, setTab] = useState("board");
  if (!A) return <PageLoading />;
  const tabs = mode === "scientist" ? [...BASE_TABS, { value: "tuning", label: "Tuning" }] : BASE_TABS;
  const fin = A.final;
  const champ = fin.table.find((r: any) => r.model === fin.champion);
  const p1 = fin.table.find((r: any) => r.model === fin.pbl1_champion);
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Stat label="Models compared" value={fin.table.length} />
        <Stat label={`Champion · ${fin.champion_phase}`} value={<span className="text-[17px]">{fin.champion}</span>} info="champion" tone={C.lime} />
        <Stat label="Champion test PR-AUC" value={<>{f3(champ.pr_auc)} <span className="text-[12px] text-mute">{ci(champ.pr_auc_ci)}</span></>} info="prauc" />
        <Stat label="Gain over PBL-1 SVM" value={<>{champ.diff_vs_pbl1 >= 0 ? "+" : "−"}{Math.abs(champ.diff_vs_pbl1).toFixed(3)} <Badge tone={champ.verdict_vs_pbl1 === "better" ? "lime" : "amber"}>{champ.verdict_vs_pbl1 === "reference" ? "same" : champ.verdict_vs_pbl1}</Badge></>} info="paired" />
      </div>
      <div className="sticky top-[86px] z-20 overflow-x-auto py-1"><Segmented value={tab} onChange={setTab} options={tabs} /></div>
      {tab === "board" && <Board A={A} />}
      {tab === "curves" && <Curves A={A} />}
      {tab === "stability" && <Stability A={A} />}
      {tab === "ablation" && <Ablation A={A} />}
      {tab === "readable" && <Readable A={A} />}
      {tab === "bench" && <Bench A={A} />}
      {tab === "baseline" && <Baseline A={A} />}
      {tab === "tuning" && <Tuning A={A} />}
      {tab === "board" && <p className="text-[12px] text-mute">PBL-1 SVM reference: {f3(p1.pr_auc)} {ci(p1.pr_auc_ci)}. {fin.champion_reason}.</p>}
    </div>
  );
}

function Board({ A }: any) {
  const fin = A.final;
  const [phase, setPhase] = useState("All");
  const [sel, setSel] = useState<string | null>(null);
  const { mode } = useMode();
  const rows = fin.table.filter((r: any) => phase === "All" || r.phase === phase);
  const max = Math.max(...fin.table.map((r: any) => r.pr_auc_ci[1])) * 1.02;
  const r = sel ? fin.table.find((x: any) => x.model === sel) : null;
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="Ranking quality on unseen alerts" info="prauc" action={<Segmented size="sm" value={phase} onChange={setPhase} options={["All", "PBL-1", "PBL-2", "PBL-3"].map((v) => ({ value: v, label: v }))} />}
        explain={<>Longer bar = better at putting reportable alerts first. White line = plausible range; red tick = random ordering ({f3(fin.base_rate_test)}). Click a model for details.</>}>
        <div className="space-y-0.5">
          {rows.map((r: any) => (
            <MeterRow key={r.model} onClick={() => setSel(r.model)} active={sel === r.model}
              label={<span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: PHASE_COLOR[r.phase] }} />{r.model}{r.model === fin.champion && <Badge tone="lime">champion</Badge>}</span>}
              value={r.pr_auc} max={max} lo={r.pr_auc_ci[0]} hi={r.pr_auc_ci[1]} color={PHASE_COLOR[r.phase]} base={fin.base_rate_test} />
          ))}
        </div>
      </Card>
      <div className="space-y-4">
        {r ? (
          <Card title={r.model} action={<PhaseTag phase={r.phase} />} explain={r.kind}>
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Test PR-AUC" value={f3(r.pr_auc)} info="prauc" tone={PHASE_COLOR[r.phase]} />
              <Stat label="Chosen-on (validation)" value={f3(r.valid_pr_auc)} info="temporal" />
              <Stat label="Catch @ 20%" value={pct(r.recall_at_budget)} info="recall_budget" />
              <Stat label="Hit rate @ 20%" value={pct(r.precision_at_budget)} info="precision_budget" />
              <Stat label="ROC-AUC" value={f3(r.roc_auc)} info="rocauc" />
              <Stat label="Safe to defer" value={pct(r.safe_deprioritised)} info="safe" />
            </div>
            <div className="inset mt-3 p-3 text-[12.5px] text-soft">
              vs PBL-1 SVM: <b className="text-white">{r.diff_vs_pbl1 >= 0 ? "+" : "−"}{Math.abs(r.diff_vs_pbl1).toFixed(3)}</b> {ci(r.diff_vs_pbl1_ci)} → <b className={r.verdict_vs_pbl1 === "better" ? "text-lime" : r.verdict_vs_pbl1 === "worse" ? "text-danger" : "text-amber"}>{r.verdict_vs_pbl1}</b>
              {r.mcnemar_p_vs_pbl1 != null && <span className="block text-mute">McNemar p = {r.mcnemar_p_vs_pbl1.toFixed(3)} on the review decision <InfoTip term="mcnemar" size={11} /></span>}
            </div>
          </Card>
        ) : (
          <Takeaway points={[
            <>All {fin.table.length} models were trained on 2020 and judged on the same {num(fin.n_test)} alerts from late 2021.</>,
            <>Most of the top models are within each other's plausible range: the ranking at the top is <b className="text-white">not</b> a clear win for any family.</>,
            <>Ensembles reach the top group, but the gain over the simple PBL-1 SVM is {fin.verdict.startsWith("no") ? "not statistically reliable" : "statistically reliable"}.</>,
            <>Click any bar for the full scorecard.</>,
          ]} />
        )}
        <Card title="Phase winners" info="champion" explain="Best model of each review, picked on early-2021 data before looking at the test quarters.">
          {fin.phases.map((p: any) => (
            <div key={p.phase} className="flex items-center justify-between border-b border-white/5 py-2 text-[13px] last:border-0">
              <span className="flex items-center gap-2"><PhaseTag phase={p.phase} />{p.model}</span>
              <span className="num">{f3(p.pr_auc)} <DS><span className="text-mute">{ci(p.pr_auc_ci)}</span></DS></span>
            </div>
          ))}
        </Card>
      </div>
      {mode === "scientist" && (
        <Card title="Full table" className="xl:col-span-2">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[980px] text-[12.5px]">
              <thead className="text-mute"><tr className="text-left">{["Model", "Phase", "Valid PR", "Test PR-AUC [95%]", "ROC", "Brier", "Recall@20", "Prec@20", "Δ vs PBL-1 [95%]", "McNemar p"].map((h) => <th key={h} className="py-2 pr-3 font-medium">{h}</th>)}</tr></thead>
              <tbody>{fin.table.map((r: any) => (
                <tr key={r.model} className="border-t border-white/5">
                  <td className="py-2 pr-3">{r.model}</td><td><PhaseTag phase={r.phase} /></td><td className="num">{f3(r.valid_pr_auc)}</td>
                  <td className="num">{f3(r.pr_auc)} <span className="text-mute">{ci(r.pr_auc_ci)}</span></td><td className="num">{f3(r.roc_auc)}</td><td className="num">{f3(r.brier)}</td>
                  <td className="num">{pct(r.recall_at_budget, 1)}</td><td className="num">{pct(r.precision_at_budget, 1)}</td>
                  <td className="num">{r.diff_vs_pbl1 >= 0 ? "+" : "−"}{Math.abs(r.diff_vs_pbl1).toFixed(3)} <span className="text-mute">{ci(r.diff_vs_pbl1_ci)}</span></td>
                  <td className="num">{r.mcnemar_p_vs_pbl1 == null ? "–" : r.mcnemar_p_vs_pbl1.toFixed(3)}</td>
                </tr>))}</tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function Curves({ A }: any) {
  const fin = A.final;
  const names = Object.keys(fin.curves);
  const [kind, setKind] = useState<"pr" | "gain" | "rel">("gain");
  const [on, setOn] = useState<string[]>(names.slice(0, 4));
  const series = on.map((m) => {
    const c = fin.curves[m];
    if (kind === "rel") return { m, pts: c.reliability.map((r: any) => ({ x: r.pred, y: r.obs })) };
    const [xs, ys] = kind === "pr" ? c.pr : c.gain;
    return { m, pts: xs.map((x: number, i: number) => ({ x, y: ys[i] })) };
  });
  const explain = { pr: "Precision (how many reviewed alerts are real) against recall (how many real alerts are found). Higher curve = better.", gain: "Share of all reportable alerts found after reviewing the top x% of the queue. The dashed diagonal is random review.", rel: "Predicted risk vs what actually happened. On the diagonal = the risk score can be read as a probability." }[kind];
  return (
    <Card title="Curves" info={kind === "rel" ? "calibration" : kind === "pr" ? "prauc" : "recall_budget"}
      action={<Segmented size="sm" value={kind} onChange={setKind} options={[{ value: "gain", label: "Catch curve" }, { value: "pr", label: "Precision–recall" }, { value: "rel", label: "Calibration" }]} />}
      explain={explain}>
      <div className="mb-3 flex flex-wrap gap-2">
        {names.map((m, i) => (
          <button key={m} onClick={() => setOn(on.includes(m) ? on.filter((x) => x !== m) : [...on, m])}
            className={clsx("flex items-center gap-1.5 rounded-full border px-3 py-1 text-[12px]", on.includes(m) ? "border-white/25 text-white" : "border-white/10 text-mute")}>
            <span className="h-2 w-2 rounded-full" style={{ background: on.includes(m) ? SERIES[i % SERIES.length] : "#3A3E47" }} />{m}
          </button>
        ))}
      </div>
      <div className="h-[420px]">
        <ResponsiveContainer>
          <LineChart margin={{ top: 8, right: 16, left: -6, bottom: 4 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="x" domain={kind === "rel" ? [0, 0.6] : [0, 1]} tickFormatter={(v) => pct(v)} tickLine={false} axisLine={false} allowDuplicatedCategory={false} />
            <YAxis type="number" domain={kind === "pr" ? [0, 0.8] : kind === "rel" ? [0, 0.6] : [0, 1]} tickFormatter={(v) => pct(v)} tickLine={false} axisLine={false} />
            <Tooltip content={<ChartTip fmt={(v: number) => pct(v, 1)} />} labelFormatter={(v: any) => pct(+v, 1)} />
            {kind === "gain" && <Line data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} dataKey="y" stroke={C.mute} strokeDasharray="4 4" dot={false} name="random" isAnimationActive={false} />}
            {kind === "rel" && <Line data={[{ x: 0, y: 0 }, { x: 0.6, y: 0.6 }]} dataKey="y" stroke={C.mute} strokeDasharray="4 4" dot={false} name="perfect" isAnimationActive={false} />}
            {kind === "pr" && <ReferenceLine y={fin.base_rate_test} stroke={C.red} strokeDasharray="4 4" />}
            {kind === "gain" && <ReferenceLine x={0.2} stroke="#fff" strokeOpacity={0.3} />}
            {series.map((s) => <Line key={s.m} data={s.pts} dataKey="y" name={s.m} stroke={SERIES[names.indexOf(s.m) % SERIES.length]} strokeWidth={2} dot={kind === "rel"} isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function Stability({ A }: any) {
  const fin = A.final, p1 = A.pbl1;
  const winners = fin.phases.map((p: any) => p.model);
  const pbl1Models = Object.keys(p1.by_quarter[0]).filter((k) => !["quarter", "role", "base_rate", "n"].includes(k));
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title="Phase winners in each test quarter" explain="If a model only won because of one lucky quarter, its bars would jump around. Red line = random ordering in that quarter.">
        <div className="h-[320px]">
          <ResponsiveContainer>
            <BarChart data={fin.by_quarter} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="quarter" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={[0, "auto"]} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip cursor={false} content={<ChartTip />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {winners.map((m: string, i: number) => <Bar key={m} dataKey={m} fill={[C.azure, C.violet, C.lime][i]} radius={[6, 6, 0, 0]} />)}
              <Line dataKey="base_rate" name="random" stroke={C.red} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="PBL-1 models across all four 2021 quarters" explain="Validation quarters (Q1–Q2) were used to choose; test quarters (Q3–Q4) only to report. Scores hold up as time moves away from training.">
        <div className="h-[320px]">
          <ResponsiveContainer>
            <LineChart data={p1.by_quarter} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="quarter" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip content={<ChartTip />} />
              <Legend wrapperStyle={{ fontSize: 11.5 }} />
              {pbl1Models.map((m, i) => <Line key={m} dataKey={m} stroke={SERIES[i]} strokeWidth={m === "SVM (RBF)" ? 2.6 : 1.4} dot={{ r: 3 }} />)}
              <Line dataKey="base_rate" name="random" stroke={C.red} strokeDasharray="4 4" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

function Ablation({ A }: any) {
  const fin = A.final, p1 = A.pbl1, enr = A.enrichment;
  const [model, setModel] = useState("SVM (RBF)");
  const enrRows = enr.rows.filter((r: any) => r.model === model);
  const feat = p1.ablation_features.map((r: any) => ({ model: r.model.replace("K-Nearest Neighbours", "KNN").replace("Logistic Regression", "Log. reg."), all: r.all, behaviour: r.behaviour_only, profile: r.profile_only }));
  return (
    <div className="space-y-4">
      <Card title="Taking pieces away: what actually carries the signal?" explain="Each block removes or adds one ingredient and re-measures on the same test alerts. Gains whose range crosses zero are not real.">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {fin.feature_ablation.map((r: any) => (
            <div key={r.model} className="inset p-3.5">
              <div className="text-[12px] text-mute">{r.model}: + PBL-2 profile & sequence features</div>
              <div className={clsx("mt-1 text-[22px] font-semibold num", r.gain_ci[0] > 0 ? "text-lime" : "text-soft")}>{r.gain >= 0 ? "+" : "−"}{Math.abs(r.gain).toFixed(3)}</div>
              <div className="text-[11.5px] text-mute num">{f3(r.without)} → {f3(r.with_pbl2_features)} · {ci(r.gain_ci)}</div>
            </div>
          ))}
          {fin.ladder.slice(1).map((l: any, i: number) => (
            <div key={l.model} className="inset p-3.5">
              <div className="text-[12px] text-mute">Stack: {["PBL-1 vs SVM alone", "adding PBL-2 models", "adding PBL-3 models"][i]}</div>
              <div className={clsx("mt-1 text-[22px] font-semibold num", l.gain_ci[0] > 0 ? "text-lime" : "text-soft")}>{l.gain >= 0 ? "+" : "−"}{Math.abs(l.gain).toFixed(3)}</div>
              <div className="text-[11.5px] text-mute num">{ci(l.gain_ci)}</div>
            </div>
          )).slice(0, 2)}
        </div>
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Which family of behaviours matters? (PBL-1)" explain="Every PBL-1 model trained three ways. The size-and-frequency profile of each channel does almost all the work; behavioural summaries add little.">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <BarChart data={feat} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                <XAxis dataKey="model" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} domain={[0.15, "auto"]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip cursor={false} content={<ChartTip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={fin.base_rate_test} stroke={C.red} strokeDasharray="4 4" />
                <Bar dataKey="behaviour" name="behaviour only" fill="#4A4F58" radius={[4, 4, 0, 0]} />
                <Bar dataKey="profile" name="profile only" fill={C.azure} radius={[4, 4, 0, 0]} />
                <Bar dataKey="all" name="all features" fill={C.lime} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Do profiles and sequences help? (PBL-2)" action={<Segmented size="sm" value={model} onChange={setModel} options={["Logistic Regression", "LDA", "SVM (RBF)", "Decision Tree"].map((m) => ({ value: m, label: m.split(" ")[0] }))} />}
          explain="Each PBL-1 model retrained with cluster features, HMM features, or both added.">
          <div className="space-y-2">
            {enrRows.map((r: any) => (
              <div key={r.variant} className="inset flex items-center justify-between px-3 py-2.5 text-[13px]">
                <span>{r.variant}</span>
                <span className="flex items-center gap-3 num">{f3(r.pr_auc)}{r.variant !== "Baseline" && <span className="text-mute">{r.gain >= 0 ? "+" : "−"}{Math.abs(r.gain).toFixed(3)}</span>}<Badge tone={r.verdict === "adds signal" ? "lime" : r.verdict === "hurts" ? "red" : "gray"}>{r.verdict}</Badge></span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[12px] text-mute">Across all six models, {enr.rows.filter((r: any) => r.verdict === "adds signal").length} of {enr.rows.filter((r: any) => r.variant !== "Baseline").length} additions measurably helped.</p>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Does the burst signal help? (PBL-1)" explain="The ‘busier than expected’ feature from the activity-baseline model, removed from each classifier.">
          {p1.ablation_burst.map((r: any) => (
            <div key={r.model} className="flex items-center justify-between border-b border-white/5 py-2 text-[13px] last:border-0">
              <span>{r.model}</span><span className="num text-mute">{r.gain >= 0 ? "+" : "−"}{Math.abs(r.gain).toFixed(4)} <DS><span>{ci(r.gain_ci, 4)}</span></DS></span>
            </div>
          ))}
        </Card>
        <Card title="Handling the 1-in-6 imbalance (PBL-1)" explain="Same models with no adjustment, class weights, or synthetic oversampling (SMOTE). Ranking quality barely moves — the choice matters less than people think.">
          <div className="h-[230px]">
            <ResponsiveContainer>
              <BarChart data={groupImb(p1.ablation_imbalance)} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                <XAxis dataKey="model" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} domain={[0.15, "auto"]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip cursor={false} content={<ChartTip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="none" name="none" fill="#4A4F58" radius={[4, 4, 0, 0]} />
                <Bar dataKey="weights" name="class weights" fill={C.lime} radius={[4, 4, 0, 0]} />
                <Bar dataKey="smote" name="SMOTE" fill={C.violet} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
}
const groupImb = (rows: any[]) => {
  const out: Record<string, any> = {};
  rows.forEach((r) => { const k = r.model.replace("K-Nearest Neighbours", "KNN").replace("Logistic Regression", "Log. reg."); out[k] = { ...(out[k] || { model: k }), [r.arm]: r.pr_auc }; });
  return Object.values(out);
};

function Readable({ A }: any) {
  const { data: E } = useData<any>("ensembles.json");
  const ex = A.explain;
  const gam = E?.gam;
  const feats = gam ? Object.keys(gam.shapes) : [];
  const [f, setF] = useState<string | null>(null);
  const cur = f ?? feats[0];
  const coefs = useMemo(() => [...ex.coefs].sort((a: any, b: any) => Math.abs(b.coef) - Math.abs(a.coef)).slice(0, 14).reverse().map((c: any) => ({ ...c, label: short(c.label, 30), err: [c.coef - c.lo, c.hi - c.coef] })), [ex]);
  return (
    <div className="space-y-4">
      {gam && (
        <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
          <Card title="Generalised additive model: one curve per behaviour" info="gam"
            explain="Pick a behaviour. The curve shows how its value changes the risk (up = more likely reported), holding everything else fixed. Curves can bend — straight lines can't.">
            <div className="mb-3 flex flex-wrap gap-1.5">
              {feats.map((k) => <button key={k} onClick={() => setF(k)} className={clsx("rounded-full border px-2.5 py-1 text-[11.5px]", cur === k ? "border-lime/50 bg-lime/10 text-lime" : "border-white/10 text-mute hover:text-white")}>{short(gam.shapes[k].label, 26)}</button>)}
            </div>
            <div className="h-[320px]">
              <ResponsiveContainer>
                <LineChart data={gam.shapes[cur].x.map((x: number, i: number) => ({ x, f: gam.shapes[cur].f[i] }))} margin={{ top: 10, right: 16, left: -6, bottom: 10 }}>
                  <CartesianGrid />
                  <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2))} tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,.3)" />
                  <Tooltip content={<ChartTip fmt={(v: number) => `${v > 0 ? "+" : ""}${v.toFixed(3)} log-odds`} />} labelFormatter={(v: any) => `value ${(+v).toFixed(2)}`} />
                  <Line dataKey="f" name="effect on risk" stroke={C.lime} strokeWidth={2.6} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <div className="space-y-4">
            <Card title="Strongest curves" explain="How much each behaviour's curve moves the score across real alerts.">
              {gam.strength.slice(0, 8).map((s: any) => (
                <MeterRow key={s.feature} label={short(s.label, 26)} value={s.sd_logit} max={gam.strength[0].sd_logit * 1.05} right={s.sd_logit.toFixed(2)} onClick={() => setF(s.feature)} active={cur === s.feature} />
              ))}
            </Card>
            <Takeaway points={[
              <>The GAM sits between logistic regression (straight lines) and boosted trees (anything goes): flexible, but every effect can still be drawn and checked.</>,
              <>Test PR-AUC {f3(A.final.table.find((r: any) => r.model === "GAM (spline)")?.pr_auc)} vs {f3(A.final.table.find((r: any) => r.model === "Logistic Regression")?.pr_auc)} for logistic regression.</>,
            ]} />
            <DS><div className="inset p-3 font-mono text-[11px] text-mute">{gam.knots} knots · cubic B-splines · {gam.basis_per_feature} basis/feature · {gam.n_coefficients} coefficients · L2, C={gam.C}</div></DS>
          </div>
        </div>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Logistic regression: direction of each behaviour" info="lr_contrib" explain="Right = pushes risk up. Whiskers are 95% ranges from 200 refits; whiskers crossing zero mean the direction isn't certain.">
          <div className="h-[420px]">
            <ResponsiveContainer>
              <ComposedChart data={coefs} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="label" width={210} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <ReferenceLine x={0} stroke="rgba(255,255,255,.3)" />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} />
                <Bar dataKey="coef" name="coefficient" barSize={12} radius={4}>
                  {coefs.map((c: any, i: number) => <Cell key={i} fill={c.coef > 0 ? C.azure : C.red} />)}
                  <ErrorBar dataKey="err" direction="x" width={4} stroke="#F4F5F6" />
                </Bar>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Decision tree as rules" explain="The PBL-1 decision tree, first levels, written as if-then rules an investigator can follow.">
          <pre className="max-h-[420px] overflow-auto rounded-xl bg-ink-900 p-3 font-mono text-[11px] leading-relaxed text-soft">{ex.tree_rules.split("\n").slice(0, 40).join("\n")}</pre>
        </Card>
      </div>
    </div>
  );
}

function Bench({ A }: any) {
  const b = A.final.benchmark;
  const rows = [...b.existing.map((r: any) => ({ ...r, fam: "Existing approach" })), ...b.ours.map((r: any) => ({ ...r, fam: "AdaptiveAML" }))];
  const [metric, setMetric] = useState<"roc_auc" | "pr_auc">("roc_auc");
  const data = rows.map((r) => ({ name: r.model, v: r[metric], err: [r[metric] - r[`${metric}_ci`][0], r[`${metric}_ci`][1] - r[metric]], fam: r.fam }));
  const bestOld = Math.max(...b.existing.map((r: any) => r[metric])), bestNew = Math.max(...b.ours.map((r: any) => r[metric]));
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="Against the dataset's published approach" action={<Segmented size="sm" value={metric} onChange={setMetric} options={[{ value: "roc_auc", label: "ROC-AUC" }, { value: "pr_auc", label: "PR-AUC" }]} />}
        explain="The SynthAML paper's 56 summary features with five standard models, re-implemented and run on exactly the published split (train 2020, test all of 2021).">
        <div className="h-[380px]">
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 10, right: 10, left: -6, bottom: 50 }}>
              <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} angle={-22} textAnchor="end" tick={{ fontSize: 11 }} />
              <YAxis tickLine={false} axisLine={false} domain={[metric === "roc_auc" ? 0.5 : 0.15, "auto"]} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip cursor={false} content={<ChartTip />} />
              <Bar dataKey="v" name={metric === "roc_auc" ? "ROC-AUC" : "PR-AUC"} radius={[6, 6, 0, 0]}>
                {data.map((d, i) => <Cell key={i} fill={d.fam === "AdaptiveAML" ? C.lime : "#4A4F58"} />)}
                <ErrorBar dataKey="err" width={5} stroke="#F4F5F6" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Best existing" value={f3(bestOld)} />
          <Stat label="Best AdaptiveAML" value={f3(bestNew)} tone={C.lime} />
        </div>
        <Takeaway points={[
          <>On the published split, our models rank alerts {bestNew > bestOld ? "slightly better" : "about as well"} ({f3(bestNew)} vs {f3(bestOld)} {metric === "roc_auc" ? "ROC-AUC" : "PR-AUC"}); the ranges overlap, so this is parity rather than a breakthrough.</>,
          <>What AdaptiveAML adds is everything around the score: a time-honest protocol, calibrated risk, per-alert reasons, uncertainty, and a work queue.</>,
          <>Grey = existing approach (library defaults, as published). Lime = AdaptiveAML.</>,
        ]} />
      </div>
    </div>
  );
}

function Baseline({ A }: any) {
  const r = A.regression;
  const sc = r.scatter.actual.map((a: number, i: number) => ({ x: r.scatter.expected[i], y: a, rep: r.scatter.y[i] }));
  const best = r.table.filter((t: any) => t.r2_test != null).reduce((a: any, b: any) => (b.r2_valid > a.r2_valid ? b : a));
  return (
    <div className="space-y-4">
      <Takeaway points={[
        <>Before scoring risk, a regression model predicts how busy each client <i>should</i> be in the 30 days before the alert, from their own earlier 11 months.</>,
        <>The best model ({best.model}) explains <b className="text-white">{pct(best.r2_test)}</b> of the variation on unseen quarters. The gap between actual and expected becomes the ‘burst’ signal.</>,
        <>The burst signal turned out not to separate reported from dismissed alerts in this data (see Ablations), and we say so.</>,
      ]} />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Expected vs actual activity" explain="Each dot is an alert (log scale). On the diagonal = the client was exactly as busy as their history predicted.">
          <div className="h-[340px]">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 10, left: -10, bottom: 8 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" name="expected" tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)} />
                <YAxis type="number" dataKey="y" name="actual" tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)} />
                <ZAxis range={[12, 12]} />
                <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(2)} />} />
                <Scatter data={sc.filter((d: any) => !d.rep)} name="dismissed" fill="#5A606B" fillOpacity={0.5} isAnimationActive={false} />
                <Scatter data={sc.filter((d: any) => d.rep)} name="reported" fill={C.lime} fillOpacity={0.6} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Regression models compared" explain="Error on unseen quarters (lower is better) and share of variation explained (higher is better).">
          <div className="overflow-x-auto"><table className="w-full text-[12.5px]"><thead className="text-mute"><tr className="text-left"><th className="py-2">Model</th><th>RMSE valid</th><th>RMSE test</th><th>R² test</th></tr></thead>
            <tbody>{r.table.map((t: any) => <tr key={t.model} className={clsx("border-t border-white/5", t.model === best.model && "text-lime")}><td className="py-2">{t.model}{t.alpha != null && <span className="text-mute"> (α={t.alpha})</span>}</td><td className="num">{f3(t.rmse_valid)}</td><td className="num">{f3(t.rmse_test)}</td><td className="num">{f3(t.r2_test)}</td></tr>)}</tbody></table></div>
          <div className="mt-4 h-[170px]">
            <ResponsiveContainer>
              <LineChart data={r.lambda} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
                <XAxis dataKey="alpha" type="number" scale="log" domain={["auto", "auto"]} tickFormatter={(v) => (+v).toExponential(0)} tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip content={<ChartTip />} labelFormatter={(v: any) => `λ = ${(+v).toPrecision(2)}`} />
                <Line dataKey="rmse_train" name="train" stroke={C.azure} dot={false} />
                <Line dataKey="rmse_valid" name="validation" stroke={C.lime} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11.5px] text-mute">Ridge penalty sweep: too little = overfit, too much = underfit.</p>
        </Card>
      </div>
    </div>
  );
}

function Tuning({ A }: any) {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {Object.entries(A.tuning).map(([m, t]: any) => (
        <Card key={m} title={m} explain={`Forward-chaining over 2020 quarters · best ${JSON.stringify(t.best)}`}>
          {t.grid.map((g: any, i: number) => (
            <MeterRow key={i} label={<span className="font-mono text-[11px]">{Object.entries(g.params).map(([k, v]) => `${k}=${v}`).join(" ")}</span>} value={g.cv_pr_auc} max={0.27} right={g.cv_pr_auc.toFixed(3)} color={g.cv_pr_auc === t.cv ? C.lime : "#4A4F58"} />
          ))}
        </Card>
      ))}
    </div>
  );
}
