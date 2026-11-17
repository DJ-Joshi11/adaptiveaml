import clsx from "clsx";
import { motion } from "framer-motion";
import { Pause, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { Badge, C, Card, ChartTip, Hl, InfoTip, InsightCard, Kpi, PageLoading, PHASE_COLOR, PhaseTag, SERIES, Segmented, Stat, Takeaway } from "../components/ui";
import { useData } from "../lib/data";
import { ci, f3, num, pct, short } from "../lib/fmt";
import { DS } from "../lib/mode";

const TABS = [
  { value: "forest", label: "Random Forest" }, { value: "boost", label: "Gradient boosting" }, { value: "ada", label: "AdaBoost lab" },
  { value: "bv", label: "Bagging vs boosting" }, { value: "stack", label: "Stacking" }, { value: "bma", label: "BMA vs ensembles" },
] as const;
type Tab = (typeof TABS)[number]["value"];

export default function Ensembles() {
  const { data: e } = useData<any>("ensembles.json");
  const { data: arena } = useData<any>("arena.json");
  const [tab, setTab] = useState<Tab>("forest");
  if (!e || !arena) return <PageLoading />;
  const fin = arena.final;
  const row = (m: string) => fin.table.find((r: any) => r.model === m);
  const ens = fin.table.filter((r: any) => r.phase === "PBL-3");
  const bestEns = ens.reduce((a: any, b: any) => (b.pr_auc > a.pr_auc ? b : a));
  const chal = row(fin.challenger);
  const p1 = row(fin.pbl1_champion), stack = row("Stack: all phases");
  return (
    <div className="space-y-5">
      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr_1fr_1fr]">
        <InsightCard kicker="Verdict" foot={`${fin.table.length} models · same 2020 training year · same ${num(fin.n_test)} unseen alerts`}>
          {bestEns.verdict_vs_pbl1 === "better"
            ? <>Combining models helps: the best ensemble beats the PBL-1 SVM by <Hl>{(bestEns.diff_vs_pbl1 * 100).toFixed(1)} pts</Hl> of PR-AUC.</>
            : <>Ensembles land <Hl>level</Hl> with the PBL-1 SVM — the signal in this data, not the model family, is the ceiling.</>}
        </InsightCard>
        <Kpi label="Best ensemble on test" value={f3(bestEns.pr_auc)} info="paired" accent
          foot={<>{bestEns.model} · {bestEns.diff_vs_pbl1 >= 0 ? "+" : "−"}{Math.abs(bestEns.diff_vs_pbl1).toFixed(3)} vs SVM, {bestEns.verdict_vs_pbl1}</>} />
        <Kpi label="Stack of all phases" value={f3(stack.pr_auc)} info="stacking" foot={<>vs PBL-1 SVM {f3(p1.pr_auc)} · {stack.verdict_vs_pbl1}</>} />
        <Kpi label="AdaBoost check" value={e.adaboost.check_vs_sklearn.rank_corr.toFixed(4)} info="adaboost" foot="rank agreement of our hand-written AdaBoost with scikit-learn's" />
      </div>

      <Card title="Every ensemble against the PBL-1 champion" info="paired"
        explain="Each row is the difference in ranking quality from the PBL-1 SVM, measured on the same reshuffled alerts. Right of the line = better; a range crossing the line = no real difference.">
        <ForestPlot rows={ens} />
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Stat label={`Challenger (best new model on early 2021)`} value={<span className="text-[17px]">{fin.challenger}</span>} info="champion" />
        <Stat label="Its lead on early 2021" value={<>{fin.valid_gap_vs_pbl1.gap >= 0 ? "+" : "−"}{Math.abs(fin.valid_gap_vs_pbl1.gap).toFixed(3)} <span className="text-[12px] text-mute">{ci(fin.valid_gap_vs_pbl1.ci)}</span></>} info="paired" />
        <Stat label="Decision" value={fin.promoted ? "Promoted" : `${fin.champion} stays`} tone={fin.promoted ? C.lime : C.amber} />
      </div>
      <p className="-mt-2 text-[12.5px] text-mute">A new model only replaces the one in service if it is <i>reliably</i> better on early-2021 data. Picking the test winner after the fact ({bestEns.model}) would be cheating: on early 2021 it ranked {[...ens].sort((a: any, b: any) => b.valid_pr_auc - a.valid_pr_auc).findIndex((r: any) => r.model === bestEns.model) + 1}th of {ens.length} ensembles.</p>

      <div className="sticky top-[86px] z-20 -mx-1 overflow-x-auto px-1 py-1"><Segmented value={tab} onChange={setTab} options={TABS as any} /></div>

      {tab === "forest" && <Forest e={e} row={row} />}
      {tab === "boost" && <Boost e={e} row={row} />}
      {tab === "ada" && <Ada e={e} />}
      {tab === "bv" && <BiasVariance e={e} />}
      {tab === "stack" && <Stacking e={e} fin={fin} />}
      {tab === "bma" && <Bma e={e} row={row} />}
    </div>
  );
}

function ForestPlot({ rows }: { rows: any[] }) {
  const data = [...rows].sort((a, b) => b.diff_vs_pbl1 - a.diff_vs_pbl1);
  const lo = Math.min(...data.map((r) => r.diff_vs_pbl1_ci[0])), hi = Math.max(...data.map((r) => r.diff_vs_pbl1_ci[1]));
  const span = Math.max(Math.abs(lo), Math.abs(hi)) * 1.1;
  const x = (v: number) => `${50 + (v / span) * 50}%`;
  return (
    <div className="space-y-1">
      {data.map((r) => (
        <div key={r.model} className="grid grid-cols-[minmax(150px,220px)_1fr_150px] items-center gap-3 rounded-lg px-2 py-1.5 hover:bg-white/[0.02]">
          <div className="flex items-center gap-2 truncate text-[13px]"><span className="h-2 w-2 shrink-0 rounded-full" style={{ background: r.verdict_vs_pbl1 === "better" ? C.lime : r.verdict_vs_pbl1 === "worse" ? C.red : C.amber }} />{r.model}</div>
          <div className="relative h-6">
            <div className="absolute inset-y-0 left-1/2 w-px bg-white/25" />
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute top-1/2 h-[3px] -translate-y-1/2 rounded"
              style={{ left: x(r.diff_vs_pbl1_ci[0]), width: `calc(${x(r.diff_vs_pbl1_ci[1])} - ${x(r.diff_vs_pbl1_ci[0])})`, background: r.verdict_vs_pbl1 === "better" ? C.lime : r.verdict_vs_pbl1 === "worse" ? C.red : "#6B7079" }} />
            <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink-950" style={{ left: x(r.diff_vs_pbl1), background: "#F4F5F6" }} />
          </div>
          <div className="text-right text-[12px] text-mute num">{r.diff_vs_pbl1 >= 0 ? "+" : "−"}{Math.abs(r.diff_vs_pbl1).toFixed(3)} <span className="opacity-70">{ci(r.diff_vs_pbl1_ci)}</span></div>
        </div>
      ))}
      <div className="grid grid-cols-[minmax(150px,220px)_1fr_150px] gap-3 px-2 text-[10.5px] text-mute"><span /><div className="flex justify-between"><span>worse than PBL-1</span><span>same</span><span>better</span></div><span /></div>
    </div>
  );
}

/* ---------------------------------------------------------------- Random Forest */
function Forest({ e, row }: any) {
  const rf = e.random_forest;
  const merged = rf.mdi.slice(0, 12).map((d: any) => ({ label: short(d.label, 30), mdi: d.importance, perm: rf.permutation.find((p: any) => p.feature === d.feature)?.importance ?? 0 }));
  const r = row("Random Forest");
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Trees" value={num(rf.n_trees)} info="rf" />
        <Stat label="Out-of-bag PR-AUC" value={f3(rf.oob_pr_auc)} info="oob" />
        <Stat label="Test PR-AUC" value={f3(r.pr_auc)} info="prauc" tone={C.lime} />
        <Stat label="Average tree depth" value={rf.tree_depth_mean.toFixed(1)} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
        <Card title="Which behaviours the forest relies on" info="permutation"
          explain="Two ways of asking the same question. Lime: how much each behaviour helped trees split alerts. Blue: how much worse the forest gets when that behaviour is scrambled.">
          <div className="h-[420px]">
            <ResponsiveContainer>
              <BarChart data={merged} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="label" width={210} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(4)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="mdi" name="Split quality (impurity)" fill={C.lime} radius={4} barSize={9} />
                <Bar dataKey="perm" name="Loss when scrambled" fill={C.azure} radius={4} barSize={9} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <div className="space-y-4">
          <Card title="How a forest is built" info="bagging">
            <ol className="space-y-3 text-[13px] text-soft">
              {[
                ["Reshuffle", "Each tree gets its own random sample of 2020 alerts (drawn with replacement)."],
                ["Restrict", `At every split a tree may only look at ${typeof rf.params.max_features === "number" ? pct(rf.params.max_features) : "√p"} of the behaviours, so trees differ from each other.`],
                ["Grow", `Each tree is a CART tree (Gini impurity), stopping when a leaf holds fewer than ${rf.params.min_samples_leaf} alerts.`],
                ["Average", `The ${num(rf.n_trees)} trees vote; the share of ‘report’ votes is the score.`],
              ].map(([t, d], i) => (
                <li key={t} className="flex gap-3"><span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-lime text-[11px] font-bold text-ink-950">{i + 1}</span><span><b className="text-white">{t}.</b> {d}</span></li>
              ))}
            </ol>
          </Card>
          <Takeaway points={[
            <>The forest leans on the same <b className="text-white">wire and card size profiles</b> that drove the PBL-1 models — it finds no hidden new signal.</>,
            rf.rank_agreement_mdi_vs_permutation > 0.6
              ? <>The two importance measures agree on ranking (rank correlation <b className="text-white">{rf.rank_agreement_mdi_vs_permutation.toFixed(2)}</b>).</>
              : <>The two measures only partly agree (rank correlation <b className="text-white">{rf.rank_agreement_mdi_vs_permutation.toFixed(2)}</b>): many size features are near-copies of each other, so trees use them a lot, but scrambling any one costs little because its twins cover for it.</>,
            <>Out-of-bag score ({f3(rf.oob_pr_auc)}) is a free check, but it mixes quarters, so the time-ordered test score is the one we report.</>,
          ]} />
          <DS><div className="inset p-3 font-mono text-[11.5px] text-mute">tuned by forward-chaining: {JSON.stringify(rf.params)} · cv PR-AUC {f3(rf.cv_pr_auc)} · class_weight=balanced_subsample · mean leaves {num(rf.tree_leaves_mean)}</div></DS>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- Boosting */
function Boost({ e, row }: any) {
  const x = e.xgboost;
  const feats = Object.keys(x.shap_dependence);
  const [f, setF] = useState(feats[0]);
  const dep = x.shap_dependence[f];
  const pts = dep.x.map((v: number, i: number) => ({ x: v, y: dep.shap[i] })).filter((p: any) => p.x != null);
  const glob = x.shap_global.slice(0, 14).map((d: any) => ({ ...d, label: short(d.label, 32) }));
  const grid = x.grid;
  const depths = Array.from(new Set(grid.map((g: any) => g.params.max_depth))) as number[];
  const ns = Array.from(new Set(grid.map((g: any) => g.params.n_estimators))) as number[];
  const gmax = Math.max(...grid.map((g: any) => g.cv_pr_auc)), gmin = Math.min(...grid.map((g: any) => g.cv_pr_auc));
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="XGBoost test PR-AUC" value={f3(row("XGBoost").pr_auc)} tone={C.lime} info="xgboost" />
        <Stat label="LightGBM test PR-AUC" value={f3(row("LightGBM").pr_auc)} info="lightgbm" />
        <Stat label="Trees × depth" value={`${x.params.n_estimators} × ${x.params.max_depth}`} />
        <Stat label="SHAP adds up to score (max error)" value={x.shap_additivity_max_error.toExponential(1)} info="shap" />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="What moves the boosted score, on average" info="shap" explain="Average size of each behaviour's push on the score across all test alerts (SHAP).">
          <div className="h-[420px]">
            <ResponsiveContainer>
              <BarChart data={glob} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="label" width={220} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} />
                <Bar dataKey="mean_abs" name="mean |SHAP|" radius={4} barSize={12}>{glob.map((_: any, i: number) => <Cell key={i} fill={i < 3 ? C.lime : "#4A5A2A"} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="How one behaviour bends the score" action={<Segmented size="sm" value={f} onChange={setF} options={feats.map((k) => ({ value: k, label: short(x.shap_dependence[k].label, 18) }))} />}
          explain="Each dot is a test alert: its value of this behaviour (across) and how much that value pushed its score (up/down). A curve, not a line — something a linear model can't express.">
          <div className="h-[420px]">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 10, left: -10, bottom: 10 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" name={dep.label} tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)} />
                <YAxis type="number" dataKey="y" name="push" tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} />
                <ZAxis range={[16, 16]} />
                <ReferenceLine y={0} stroke="rgba(255,255,255,.25)" />
                <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} />
                <Scatter data={pts} fill={C.lime} fillOpacity={0.45} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Card title="Tuning, without peeking at 2021" info="temporal" explain="Each cell is the average score when training on earlier 2020 quarters and checking on the next one. Brightest cell = chosen.">
          <div className="grid gap-1.5" style={{ gridTemplateColumns: `90px repeat(${ns.length}, 1fr)` }}>
            <div />{ns.map((n) => <div key={n} className="text-center text-[11px] text-mute">{n} trees</div>)}
            {depths.map((d) => (
              <>
                <div key={`l${d}`} className="flex items-center text-[11.5px] text-mute">depth {d}</div>
                {ns.map((n) => {
                  const g = grid.find((q: any) => q.params.max_depth === d && q.params.n_estimators === n);
                  const o = 0.1 + 0.9 * ((g.cv_pr_auc - gmin) / (gmax - gmin || 1));
                  const best = g.cv_pr_auc === gmax;
                  return <div key={`${d}-${n}`} className={clsx("grid h-12 place-items-center rounded-lg text-[12px] font-semibold num", best && "ring-2 ring-white")} style={{ background: `rgba(182,242,74,${o})`, color: o > 0.55 ? "#08090B" : "#F4F5F6" }}>{g.cv_pr_auc.toFixed(3)}</div>;
                })}
              </>
            ))}
          </div>
        </Card>
        <Takeaway points={[
          <>Boosting builds small trees one after another; each new tree corrects what the sum of the earlier ones still gets wrong.</>,
          <>Shallow trees (depth {x.params.max_depth}) won: deeper trees memorised 2020 and did worse on the next quarter.</>,
          <>XGBoost and LightGBM score almost the same ({f3(row("XGBoost").pr_auc)} vs {f3(row("LightGBM").pr_auc)}), so the result is not an artefact of one library.</>,
          <>Adding the PBL-2 profile and sequence features to XGBoost: {f3(row("XGBoost + profiles + sequences").pr_auc)} — {row("XGBoost + profiles + sequences").pr_auc > row("XGBoost").pr_auc ? "a small, uncertain gain" : "no gain"}.</>,
        ]} />
      </div>
      <DS><div className="inset p-3 font-mono text-[11.5px] text-mute">XGBoost fixed: {JSON.stringify(x.fixed)} · tuned {JSON.stringify(x.params)} · cv {f3(x.cv_pr_auc)}</div></DS>
    </div>
  );
}

/* ---------------------------------------------------------------- AdaBoost lab */
function Ada({ e }: any) {
  const a = e.adaboost;
  const M = a.trace.length;
  const [m, setM] = useState(1);
  const [play, setPlay] = useState(false);
  useEffect(() => {
    if (!play) return;
    const t = setInterval(() => setM((v) => (v >= M ? (setPlay(false), M) : Math.min(M, v + (v < 20 ? 1 : v < 80 ? 3 : 8)))), 120);
    return () => clearInterval(t);
  }, [play, M]);
  const t = a.trace[m - 1];
  const snapKeys = Object.keys(a.weight_snapshots).map(Number).sort((x, y) => x - y);
  const snap = snapKeys.reduce((acc, k) => (k <= m ? k : acc), snapKeys[0]);
  const ws = a.weight_snapshots[String(snap)];
  const hist = ws.counts_reported.map((r: number, i: number) => ({ x: (ws.edges[i] + ws.edges[i + 1]) / 2, reported: r, dismissed: ws.counts_dismissed[i] }));
  const trace = useMemo(() => a.trace.map((r: any) => ({ ...r, half_alpha: r.alpha / 2 })), [a]);
  return (
    <div className="space-y-4">
      <Card title="Step through AdaBoost, one round at a time" info="adaboost"
        explain="Each round fits a one-question rule (a ‘stump’). Alerts it gets wrong become heavier, so the next rule has to pay attention to them.">
        <div className="flex flex-wrap items-center gap-4">
          <button onClick={() => { if (m >= M) setM(1); setPlay(!play); }} className="grid h-10 w-10 place-items-center rounded-full bg-lime text-ink-950">{play ? <Pause size={16} /> : <Play size={16} />}</button>
          <input type="range" min={1} max={M} value={m} onChange={(ev) => setM(+ev.target.value)} className="flex-1 accent-[#B6F24A]" />
          <span className="w-28 text-right text-[15px] font-semibold num">round {m} / {M}</span>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
          <Stat label="Weighted error εₘ" value={pct(t.eps, 1)} info="eps" />
          <Stat label="Rule's say αₘ" value={t.alpha.toFixed(3)} info="alpha" tone={C.lime} />
          <Stat label="Exponential loss" value={t.exp_loss.toFixed(3)} info="exp_loss" />
          <Stat label="Training error" value={pct(t.train_error, 1)} />
          <Stat label="Effective sample" value={pct(t.ess_share, 0)} info="ess" />
          <Stat label="Weight on reported" value={pct(t.weight_on_reported, 0)} tone={C.amber} />
        </div>
        <div className="inset mt-3 px-3.5 py-2.5 text-[12.5px] text-soft">
          Round {m}'s rule: <b className="text-white">“is {e.feature_names?.[t.feature] ?? `feature #${t.feature}`} above {t.threshold.toFixed(2)}?”</b> — wrong on {pct(t.eps, 1)} of the weighted alerts, so it gets a say of {t.alpha.toFixed(2)}.
          {m > 1 && <> Reported alerts now carry <b className="text-amber">{pct(t.weight_on_reported, 0)}</b> of the weight (they started at {pct(a.base_rate_train, 0)}).</>}
        </div>
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Error and say, round by round" explain="Rules get weaker as the easy patterns are used up (error creeps toward 50%), so each later rule counts less.">
          <div className="h-[260px]">
            <ResponsiveContainer>
              <LineChart data={trace} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
                <CartesianGrid />
                <XAxis dataKey="round" tickLine={false} axisLine={false} />
                <YAxis yAxisId="l" tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => pct(v)} />
                <YAxis yAxisId="r" orientation="right" tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine yAxisId="l" x={m} stroke="#fff" strokeOpacity={0.5} />
                <Line yAxisId="l" dataKey="eps" name="weighted error εₘ" stroke={C.amber} dot={false} strokeWidth={1.6} isAnimationActive={false} />
                <Line yAxisId="r" dataKey="alpha" name="say αₘ" stroke={C.lime} dot={false} strokeWidth={1.6} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="The loss AdaBoost is quietly minimising" info="exp_loss" explain="Exponential loss falls every single round — that is the theory working. Training error falls too, but jumpily.">
          <div className="h-[260px]">
            <ResponsiveContainer>
              <LineChart data={trace} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
                <CartesianGrid />
                <XAxis dataKey="round" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} />
                <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(4)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine x={m} stroke="#fff" strokeOpacity={0.5} />
                <Line dataKey="exp_loss" name="exponential loss" stroke={C.lime} dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line dataKey="train_error" name="training error" stroke={C.azure} dot={false} strokeWidth={1.4} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card title={`Where the weight sits after round ${snap}`} info="ess"
          explain="How heavy each training alert is (1 = its starting weight). As rounds go by, weight piles onto a few hard alerts — many of them reported ones.">
          <div className="h-[240px]">
            <ResponsiveContainer>
              <BarChart data={hist} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                <XAxis dataKey="x" tickLine={false} axisLine={false} tickFormatter={(v) => `${Math.pow(10, v).toFixed(v < 0 ? 2 : 0)}×`} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => num(v)} />} labelFormatter={(v: any) => `weight ≈ ${Math.pow(10, +v).toFixed(2)}× start`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="dismissed" stackId="a" name="dismissed" fill="#3A3E47" />
                <Bar dataKey="reported" stackId="a" name="reported" fill={C.lime} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11.5px] text-mute">Snapshots stored at rounds {snapKeys.join(", ")}; the slider shows the latest one reached.</p>
        </Card>
        <div className="space-y-4">
          <Card title="Ranking quality as rounds are added" explain="Measured on early-2021 alerts the model never trained on.">
            <div className="h-[170px]">
              <ResponsiveContainer>
                <AreaChart data={a.valid_by_round} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
                  <defs><linearGradient id="ad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={C.lime} stopOpacity={0.3} /><stop offset="1" stopColor={C.lime} stopOpacity={0} /></linearGradient></defs>
                  <XAxis dataKey="round" type="number" scale="log" domain={[1, M]} tickLine={false} axisLine={false} ticks={[1, 10, 100, M]} />
                  <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
                  <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} labelFormatter={(v: any) => `round ${v}`} />
                  <Area dataKey="valid_pr_auc" name="PR-AUC" stroke={C.lime} fill="url(#ad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Checked against the library">
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Ours (test)" value={f3(a.check_vs_sklearn.ours_test_pr_auc)} />
              <Stat label="scikit-learn (test)" value={f3(a.check_vs_sklearn.sklearn_test_pr_auc)} />
            </div>
            <p className="mt-2 text-[12px] text-mute">Rank correlation {a.check_vs_sklearn.rank_corr.toFixed(4)} — the hand-written version reproduces the library exactly. The hardest 5% of training alerts (by final weight) are {pct(a.hardest_5pct_reported_share)} reported, vs {pct(a.base_rate_train)} overall.</p>
          </Card>
        </div>
      </div>
      <DS>
        <Card title="The update rules, as implemented (adaboost.py)">
          <div className="grid gap-3 font-mono text-[12px] text-soft md:grid-cols-2">
            <div className="inset p-3">εₘ = Σₙ wₙ·I(yₘ(xₙ) ≠ tₙ) / Σₙ wₙ<br />αₘ = ln((1 − εₘ) / εₘ)<br />wₙ ← wₙ · exp(αₘ·I(yₘ(xₙ) ≠ tₙ)), then normalise</div>
            <div className="inset p-3">fₘ(x) = ½ Σₗ≤ₘ αₗ yₗ(x),  tₙ ∈ {"{−1, +1}"}<br />E = Σₙ exp(−tₙ fₘ(xₙ))  (stage-wise minimised)<br />Y(x) = sign(fₘ(x)); fₘ itself is used as the ranking score</div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">{a.stump_features.map((s: any) => <Badge key={s.feature} tone="gray">{s.label} · {s.rounds}</Badge>)}</div>
        </Card>
      </DS>
    </div>
  );
}

/* ---------------------------------------------------------------- Bagging vs boosting */
function BiasVariance({ e }: any) {
  const bv = e.bias_variance.rows;
  const minB = Math.min(...bv.map((r: any) => r.bias2_plus_noise));
  const data = bv.map((r: any) => ({ ...r, excess_bias: r.bias2_plus_noise - minB }));
  const fam = (f: string) => (f === "bagging" ? C.azure : f === "boosting" ? C.lime : "#6B7079");
  const single = bv.find((r: any) => r.model === "Single deep tree"), bag = bv.find((r: any) => r.model === "Bagging (CART)");
  const stump = bv.find((r: any) => r.model === "Single stump"), ada = bv.find((r: any) => r.model === "AdaBoost");
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Jumpiness (variance)" info="bias_variance" explain="How much a model's scores change when it is retrained on a slightly different sample. Bagging exists to shrink this.">
          <div className="h-[260px]">
            <ResponsiveContainer>
              <BarChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 30 }}>
                <XAxis dataKey="model" tickLine={false} axisLine={false} interval={0} angle={-15} textAnchor="end" tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(3)} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(4)} />} />
                <Bar dataKey="variance" name="variance" radius={[6, 6, 0, 0]}>{data.map((d: any, i: number) => <Cell key={i} fill={fam(d.family)} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Systematic error (bias)" info="bias_variance" explain="How far the average model is from the truth, beyond the noise every model shares. Boosting exists to shrink this.">
          <div className="h-[260px]">
            <ResponsiveContainer>
              <BarChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 30 }}>
                <XAxis dataKey="model" tickLine={false} axisLine={false} interval={0} angle={-15} textAnchor="end" tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(3)} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(4)} />} />
                <Bar dataKey="excess_bias" name="bias² above the best model" radius={[6, 6, 0, 0]}>{data.map((d: any, i: number) => <Cell key={i} fill={fam(d.family)} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <div className="flex flex-wrap gap-4 text-[12px] text-mute"><span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: C.azure }} />bagging</span><span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: C.lime }} />boosting</span><span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-[#6B7079]" />single model</span><span>· {e.bias_variance.replicates} retrainings on bootstrap samples of 2020, scored on {num(e.bias_variance.n_eval)} early-2021 alerts</span></div>
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card title="Does adding more models keep helping?" explain="Ranking quality on early-2021 alerts as the number of trees (or boosting rounds) grows. Bagging flattens out; boosting can keep improving — or start to overfit.">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <LineChart data={e.n_estimators_curve} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid />
                <XAxis dataKey="n" type="number" scale="log" domain={[1, 600]} ticks={[1, 2, 5, 10, 20, 50, 100, 200, 600]} tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} labelFormatter={(v: any) => `${v} trees / rounds`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line dataKey="Random Forest" stroke={C.azure} strokeWidth={2} dot={{ r: 2.5 }} connectNulls />
                <Line dataKey="Bagging (CART)" stroke={C.violet} strokeWidth={2} dot={{ r: 2.5 }} connectNulls />
                <Line dataKey="AdaBoost" stroke={C.amber} strokeWidth={2} dot={{ r: 2.5 }} connectNulls />
                <Line dataKey="XGBoost" stroke={C.lime} strokeWidth={2.4} dot={{ r: 2.5 }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Takeaway points={[
          <>Bagging cut the deep tree's jumpiness from <b className="text-white">{single.variance.toFixed(4)}</b> to <b className="text-white">{bag.variance.toFixed(4)}</b> — averaging many unstable trees steadies them.</>,
          <>Boosting cut the stump's systematic error: excess bias {(stump.bias2_plus_noise - minB).toFixed(4)} → {(ada.bias2_plus_noise - minB).toFixed(4)} — stacking weak rules makes a stronger one.</>,
          <>That is the textbook picture: <b className="text-white">bagging reduces variance, boosting reduces bias</b>.</>,
          <>Most of the squared error is shared noise no model can remove — the same reason every model here plateaus near the same PR-AUC.</>,
        ]} />
      </div>
      <DS>
        <Card title="Raw decomposition">
          <div className="overflow-x-auto"><table className="w-full text-[12.5px]"><thead className="text-mute"><tr className="text-left"><th className="py-2">Model</th><th>bias² + noise</th><th>variance</th><th>total</th><th>PR-AUC (mean ± sd over refits)</th></tr></thead>
            <tbody>{bv.map((r: any) => <tr key={r.model} className="border-t border-white/5"><td className="py-2">{r.model}</td><td className="num">{r.bias2_plus_noise.toFixed(4)}</td><td className="num">{r.variance.toFixed(4)}</td><td className="num">{r.total.toFixed(4)}</td><td className="num">{r.pr_auc_mean.toFixed(3)} ± {r.pr_auc_sd.toFixed(3)}</td></tr>)}</tbody></table></div>
          <p className="mt-2 text-[11.5px] text-mute">{e.bias_variance.note}. Models trained without class weights so probabilities are comparable; AdaBoost scores Platt-mapped on each bag.</p>
        </Card>
      </DS>
    </div>
  );
}

/* ---------------------------------------------------------------- Stacking */
function Stacking({ e, fin }: any) {
  const st = e.stacking;
  const full = st.variants["Stack: all phases"];
  const wts = full.weights.map((w: any) => ({ ...w, err: [w.weight - w.ci[0], w.ci[1] - w.weight] }));
  const ladder = fin.ladder.map((l: any, i: number) => ({ ...l, name: ["PBL-1 SVM alone", "Stack of PBL-1", "+ PBL-2 patterns", "+ PBL-3 ensembles"][i], err: [l.pr_auc - l.pr_auc_ci[0], l.pr_auc_ci[1] - l.pr_auc] }));
  const D = st.diversity;
  return (
    <div className="space-y-4">
      <Card title="How the stack is wired" info="stacking" explain="Seven models from all three reviews score every alert. A small model on top learns how much to trust each one — using only scores they gave to alerts they had not seen.">
        <StackDiagram st={st} />
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="How much the stack trusts each model" info="oof" explain="Weights learned by the top model (bigger = trusted more). Bars crossing zero mean the stack can do without that model.">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <BarChart data={wts} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="model" width={190} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <ReferenceLine x={0} stroke="rgba(255,255,255,.3)" />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number, p: any) => `${v.toFixed(3)}  ${ci(p.payload.ci)}`} />} />
                <Bar dataKey="weight" name="weight" radius={4} barSize={14}>
                  {wts.map((w: any, i: number) => <Cell key={i} fill={PHASE_COLOR[w.phase]} />)}
                  <ErrorBar dataKey="err" width={5} stroke="#F4F5F6" direction="x" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Adding each review's models, step by step" info="paired" explain="Test ranking quality as we let the stack use more of the project's models. Whiskers are 95% ranges.">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <ComposedChart data={ladder} margin={{ top: 10, right: 10, left: -6, bottom: 0 }}>
                <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} />
                <Bar dataKey="pr_auc" name="test PR-AUC" radius={[8, 8, 8, 8]} barSize={46}>
                  {ladder.map((_: any, i: number) => <Cell key={i} fill={[C.azure, C.azure, C.violet, C.lime][i]} />)}
                  <ErrorBar dataKey="err" width={8} stroke="#F4F5F6" />
                </Bar>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-[11.5px]">
            {ladder.slice(1).map((l: any) => <div key={l.name} className="inset px-2.5 py-2"><div className="text-mute">{l.name}</div><div className={clsx("font-semibold num", l.gain_ci[0] > 0 ? "text-lime" : "text-soft")}>{l.gain >= 0 ? "+" : "−"}{Math.abs(l.gain).toFixed(3)}</div><DS><div className="text-mute num">{ci(l.gain_ci)}</div></DS></div>)}
          </div>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Card title="What if we drop one model?" explain="Loss in test ranking quality when the stack is rebuilt without each model. Near zero = that model was not pulling its weight.">
          <div className="space-y-1.5">
            {fin.stack_leave_one_out.sort((a: any, b: any) => b.loss - a.loss).map((r: any) => (
              <div key={r.removed} className="grid grid-cols-[minmax(150px,1fr)_1.3fr_auto] items-center gap-3 text-[12.5px]">
                <span className="flex items-center gap-2 truncate"><PhaseTag phase={r.phase} />{r.removed}</span>
                <div className="relative h-4 rounded bg-ink-700">
                  <div className="absolute inset-y-0 left-1/2 w-px bg-white/30" />
                  <div className="absolute inset-y-0 rounded" style={{ background: r.loss > 0 ? C.lime : C.red, left: r.loss > 0 ? "50%" : `${50 + (r.loss / 0.03) * 50}%`, width: `${Math.min(50, (Math.abs(r.loss) / 0.03) * 50)}%` }} />
                </div>
                <span className="w-24 text-right num">{r.loss >= 0 ? "−" : "+"}{Math.abs(r.loss).toFixed(3)}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Do the models think alike?" explain="How similarly each pair of models ranks the same alerts (1 = identical). A committee gains most from members that disagree.">
          <div className="grid gap-[3px]" style={{ gridTemplateColumns: `120px repeat(${D.models.length}, 1fr)` }}>
            <div />{D.models.map((m: string, j: number) => <div key={j} className="truncate text-center text-[9.5px] text-mute" title={m}>{m.split(" ")[0]}</div>)}
            {D.models.map((m: string, i: number) => (
              <>
                <div key={`r${i}`} className="truncate pr-1 text-[11px] text-mute" title={m}>{short(m, 18)}</div>
                {D.spearman_valid[i].map((v: number, j: number) => <div key={`${i}-${j}`} title={`${m} vs ${D.models[j]}: ${v.toFixed(2)}`} className="grid h-8 place-items-center rounded-[4px] text-[10px] num" style={{ background: `rgba(182,242,74,${Math.max(0.05, v)})`, color: v > 0.6 ? "#08090B" : "#C9CDD4" }}>{v.toFixed(2)}</div>)}
              </>
            ))}
          </div>
        </Card>
      </div>
      <Takeaway points={[
        <>The stack is trained on <b className="text-white">{num(e.stacking_oof.rows)}</b> out-of-fold scores from {e.stacking_oof.quarters.join(", ")} — each base model scored a quarter it had not been trained on, exactly like it will at test time.</>,
        <>The PBL-1 linear models and the boosted trees agree with each other a lot, so the stack mostly re-weights the same signal rather than adding a new one.</>,
        <>The sequence score from PBL-2 is the most <i>different</i> member, but it is weak on its own, so the stack gives it little weight.</>,
      ]} />
      <DS><div className="inset p-3 font-mono text-[11.5px] text-mute">meta-learner: StandardScaler → LogisticRegression(C={full.C}) chosen by forward-chaining on OOF quarters (meta cv PR-AUC {f3(full.meta_cv_pr_auc)}); base inputs on log-odds scale (SVM: decision function; HMM: LLR); 95% weight intervals from 200 bootstrap refits.</div></DS>
    </div>
  );
}

function StackDiagram({ st }: { st: any }) {
  const groups = ["PBL-1", "PBL-2", "PBL-3"].map((p) => ({ p, models: st.base_models.filter((b: any) => b.phase === p) }));
  return (
    <div className="grid items-center gap-4 lg:grid-cols-[1.4fr_auto_1fr_auto_0.9fr]">
      <div className="space-y-2.5">
        {groups.map((g, gi) => (
          <motion.div key={g.p} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: gi * 0.1 }} className="inset p-3" style={{ borderColor: `${PHASE_COLOR[g.p]}33` }}>
            <div className="mb-2 flex items-center gap-2"><PhaseTag phase={g.p} /><span className="text-[11.5px] text-mute">{["classic classifiers", "patterns & sequences", "tree ensembles"][gi]}</span></div>
            <div className="flex flex-wrap gap-1.5">{g.models.map((m: any) => <span key={m.name} className="rounded-lg bg-ink-700 px-2 py-1 text-[11.5px]">{m.name}</span>)}</div>
          </motion.div>
        ))}
      </div>
      <Flow label="scores on unseen quarters" />
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }} className="rounded-2xl border border-lime/30 bg-lime/[0.06] p-4 text-center">
        <div className="label-caps text-lime">Meta-learner</div>
        <div className="mt-1 text-[15px] font-semibold">Logistic regression</div>
        <div className="mt-1 text-[11.5px] text-mute">learns one trust weight per model</div>
      </motion.div>
      <Flow label="final score" />
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="rounded-2xl bg-white p-4 text-center text-ink-950">
        <div className="text-[11px] font-semibold uppercase tracking-wider opacity-60">Output</div>
        <div className="text-[15px] font-bold">Risk for each alert</div>
      </motion.div>
    </div>
  );
}

function Flow({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center gap-1 text-[10.5px] text-mute">
      <svg width="70" height="16" className="rotate-90 lg:rotate-0"><defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#B6F24A" /></marker></defs>
        <motion.line x1="2" y1="8" x2="64" y2="8" stroke="#B6F24A" strokeWidth="2" strokeDasharray="5 4" markerEnd="url(#arr)" animate={{ strokeDashoffset: [18, 0] }} transition={{ repeat: Infinity, duration: 0.9, ease: "linear" }} /></svg>
      <span className="max-w-[90px] text-center leading-tight">{label}</span>
    </div>
  );
}

/* ---------------------------------------------------------------- BMA */
function Bma({ e, row }: any) {
  const b = e.bma;
  const models = b.models as string[];
  const curve = b.evidence_curve;
  const winner = models.reduce((a, m) => (b.weights[m] > b.weights[a] ? m : a), models[0]);
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card title="BMA picks a winner as evidence piles up" info="bma"
          explain="Bayesian model averaging asks ‘which ONE model is true?’. With a few alerts it hedges; with thousands it hands almost all the weight to one model.">
          <div className="h-[320px]">
            <ResponsiveContainer>
              <AreaChart data={curve} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <XAxis dataKey="n" type="number" scale="log" domain={[10, "dataMax"]} ticks={curve.map((c: any) => c.n)} tickLine={false} axisLine={false} tickFormatter={(v) => num(v)} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} domain={[0, 1]} />
                <Tooltip content={<ChartTip fmt={(v: number) => pct(v, 1)} />} labelFormatter={(v: any) => `after ${num(+v)} alerts of evidence`} />
                <Legend wrapperStyle={{ fontSize: 11.5 }} />
                {models.map((m, i) => <Area key={m} dataKey={m} stackId="1" stroke={SERIES[i]} fill={SERIES[i]} fillOpacity={0.7} />)}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <div className="space-y-4">
          <Card title="Final posterior weights" explain={`Evidence: likelihood of ${num(b.n_evidence)} alerts from 2021 Q2.`}>
            <div className="space-y-2">
              {models.map((m, i) => (
                <div key={m} className="flex items-center gap-2 text-[12.5px]">
                  <span className="w-36 truncate">{m}</span>
                  <div className="h-2 flex-1 rounded-full bg-ink-700"><motion.div initial={{ width: 0 }} animate={{ width: `${b.weights[m] * 100}%` }} className="h-full rounded-full" style={{ background: SERIES[i] }} /></div>
                  <span className="w-12 text-right num">{pct(b.weights[m], 1)}</span>
                </div>
              ))}
            </div>
          </Card>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="BMA test PR-AUC" value={f3(row("Bayesian model averaging").pr_auc)} />
            <Stat label="Stack test PR-AUC" value={f3(row("Stack: all phases").pr_auc)} tone={C.lime} />
          </div>
        </div>
      </div>
      <Card title="Same idea? No.">
        <div className="grid gap-3 md:grid-cols-2">
          <div className="inset p-4">
            <div className="mb-2 flex items-center gap-2"><Badge tone="violet">BMA</Badge><span className="text-[13px] font-semibold">Uncertainty about which model is right</span></div>
            <ul className="space-y-1.5 text-[12.5px] text-soft">
              <li>• Assumes exactly one of the candidates generated the data.</li>
              <li>• Weights = how probable each model is, given the data.</li>
              <li>• As data grows, weights collapse onto one model (here: <b className="text-white">{winner}</b>).</li>
              <li>• Never does better than the best single candidate in the long run.</li>
            </ul>
          </div>
          <div className="inset p-4">
            <div className="mb-2 flex items-center gap-2"><Badge tone="lime">Ensemble</Badge><span className="text-[13px] font-semibold">A bigger, combined model</span></div>
            <ul className="space-y-1.5 text-[12.5px] text-soft">
              <li>• Assumes no single model is right; the combination is the model.</li>
              <li>• Weights are learned to predict well (stacking) or fixed (voting).</li>
              <li>• Keeps several members even with lots of data, if they help.</li>
              <li>• Can beat every member when members make different mistakes.</li>
            </ul>
          </div>
        </div>
      </Card>
      <DS><div className="inset p-3 font-mono text-[11.5px] text-mute">p(Mₖ|D) ∝ exp(ℓₖ), equal priors; ℓₖ = Σ log p(yₙ|xₙ,Mₖ) on 2021 Q2 with Platt maps fitted on 2021 Q1. log-likelihoods: {models.map((m) => `${m.split(" ")[0]} ${b.log_likelihood[m].toFixed(1)}`).join(" · ")}. Curve averaged over 30 random orderings.</div></DS>
    </div>
  );
}
