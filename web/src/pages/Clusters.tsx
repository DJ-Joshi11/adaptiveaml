import clsx from "clsx";
import { useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { Badge, C, Card, ChartTip, InfoTip, PageLoading, SERIES, Segmented, Stat, Takeaway } from "../components/ui";
import { useData } from "../lib/data";
import { f2, f3, num, pct, short } from "../lib/fmt";
import { DS } from "../lib/mode";

const TABS = [{ value: "map", label: "Profile map" }, { value: "profiles", label: "Profiles" }, { value: "k", label: "How many profiles?" },
  { value: "latent", label: "Latent dimensions" }, { value: "tree", label: "Hierarchy" }, { value: "quality", label: "Quality" }];
const KC = [C.azure, C.violet, C.amber];

export default function Clusters() {
  const { data: c } = useData<any>("clusters.json");
  const [tab, setTab] = useState("map");
  if (!c) return <PageLoading />;
  const prof = c.profiles.kmeans;
  const rates = prof.test.map((p: any) => p.report_rate);
  const hi = prof.test[rates.indexOf(Math.max(...rates))], lo = prof.test[rates.indexOf(Math.min(...rates))];
  const ev = c.evaluation["K-Means"].test;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Broad profiles (K-Means)" value={c.kmeans.K} info="kmeans" />
        <Stat label="Fine profiles (Gaussian mixture)" value={c.gmm.K} info="gmm" />
        <Stat label="Summary directions for 80% of info" value={`${c.pca.d80} of ${c.n_features}`} info="pca" />
        <Stat label="Link to outcome (NMI)" value={f3(ev.nmi)} tone={C.amber} />
      </div>
      <Takeaway points={[
        <>Alerts fall into <b className="text-white">{c.kmeans.K} broad behavioural profiles</b>, found without ever looking at whether they were reported.</>,
        <>Profiles differ in <i>how</i> clients behave (tenure, channel mix, transaction sizes) but hardly in how often they get reported ({pct(lo.report_rate)} to {pct(hi.report_rate)} in late 2021).</>,
        <>That is a useful negative result: suspiciousness here is not a ‘type of customer’, so profiles describe context rather than risk.</>,
      ]} />
      <div className="sticky top-[86px] z-20 overflow-x-auto py-1"><Segmented value={tab} onChange={setTab} options={TABS} /></div>
      {tab === "map" && <MapTab c={c} />}
      {tab === "profiles" && <ProfilesTab c={c} />}
      {tab === "k" && <KTab c={c} />}
      {tab === "latent" && <LatentTab c={c} />}
      {tab === "tree" && <TreeTab c={c} />}
      {tab === "quality" && <QualityTab c={c} />}
    </div>
  );
}

function MapTab({ c }: any) {
  const [color, setColor] = useState<"k" | "g" | "r">("k");
  const groups = color === "k" ? [0, 1, 2] : color === "g" ? Array.from({ length: c.gmm.K }, (_, i) => i) : [0, 1];
  const col = (i: number) => (color === "r" ? (i ? C.lime : "#4A4F58") : color === "k" ? KC[i] : SERIES[i % SERIES.length]);
  return (
    <div className="grid gap-4 xl:grid-cols-[1.6fr_1fr]">
      <Card title="Map of look-alike alerts" info="pca" action={<Segmented size="sm" value={color} onChange={setColor} options={[{ value: "k", label: "Broad profile" }, { value: "g", label: "Fine profile" }, { value: "r", label: "Outcome" }]} />}
        explain="Each dot is a late-2021 alert squeezed onto two summary directions (PCA). Close dots behave alike. Colour shows the group, or the real outcome.">
        <div className="h-[460px]">
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 6, right: 6, left: -12, bottom: 6 }}>
              <CartesianGrid />
              <XAxis type="number" dataKey="x" name="direction 1" tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} />
              <YAxis type="number" dataKey="y" name="direction 2" tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} />
              <ZAxis range={[16, 16]} />
              <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(2)} />} />
              {groups.map((g) => (
                <Scatter key={g} name={color === "r" ? (g ? "reported" : "dismissed") : `${color === "k" ? "P" : "G"}${g + 1}`} isAnimationActive={false}
                  data={c.scatter.filter((p: any) => (color === "k" ? p.k : color === "g" ? p.g : p.r) === g)} fill={col(g)} fillOpacity={color === "r" && !g ? 0.35 : 0.6} />
              ))}
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <div className="space-y-4">
        <Card title="How to read it">
          <ul className="space-y-2 text-[13px] text-soft">
            <li>• <b className="text-white">Broad profile</b>: K-Means puts each alert in exactly one of {c.kmeans.K} groups.</li>
            <li>• <b className="text-white">Fine profile</b>: the Gaussian mixture finds {c.gmm.K} smaller groups and lets an alert belong partly to several.</li>
            <li>• <b className="text-white">Outcome</b>: reported alerts (lime) are sprinkled across the whole map — no region is ‘the suspicious one’.</li>
          </ul>
        </Card>
        <Card title="Direction 1 mostly measures…">
          {c.pca.loadings[0].top.slice(0, 5).map((t: any) => <div key={t.label} className="flex justify-between border-b border-white/5 py-1.5 text-[12.5px] last:border-0"><span>{t.label}</span><span className="num text-mute">{t.weight.toFixed(2)}</span></div>)}
          <p className="mt-2 text-[12px] text-mute">…so moving right on the map means bigger typical transaction sizes.</p>
        </Card>
      </div>
    </div>
  );
}

function ProfilesTab({ c }: any) {
  const [algo, setAlgo] = useState<"kmeans" | "gmm">("kmeans");
  const P = c.profiles[algo];
  const data = P.test.map((p: any) => {
    const t = P.train.find((x: any) => x.cluster === p.cluster);
    return { name: `${algo === "kmeans" ? "P" : "G"}${p.cluster + 1}`, test: p.report_rate, train: t?.report_rate, err: [p.report_rate - p.lo, p.hi - p.report_rate], n: p.size };
  });
  const base = c.evaluation.baseline_purity;
  return (
    <div className="space-y-4">
      <Card title="How often each profile was reported" action={<Segmented size="sm" value={algo} onChange={setAlgo} options={[{ value: "kmeans", label: "Broad (K-Means)" }, { value: "gmm", label: "Fine (mixture)" }]} />}
        explain="Share of alerts reported, per profile, in 2020 (blue, used to build profiles) and late 2021 (lime, new alerts). Dashed red = overall rate.">
        <div className="h-[300px]">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="name" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} />
              <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine y={1 - base} stroke={C.red} strokeDasharray="4 4" />
              <Bar dataKey="train" name="2020" fill={C.azure} radius={[6, 6, 0, 0]} />
              <Bar dataKey="test" name="late 2021" fill={C.lime} radius={[6, 6, 0, 0]}><ErrorBar dataKey="err" width={5} stroke="#F4F5F6" /></Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>
      {algo === "kmeans" && (
        <div className="grid gap-4 lg:grid-cols-3">
          {c.profiles.kmeans.train.map((p: any) => {
            const t = c.profiles.kmeans.test.find((x: any) => x.cluster === p.cluster);
            return (
              <Card key={p.cluster} glass>
                <div className="flex items-center justify-between"><Badge tone={(["azure", "violet", "amber"] as const)[p.cluster]}>P{p.cluster + 1}</Badge><span className="text-[12px] text-mute">{pct(p.share)} of alerts</span></div>
                <div className="mt-2 text-[14px] font-semibold capitalize leading-snug">{p.name}</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <Stat label="Reported 2020" value={pct(p.report_rate)} />
                  <Stat label="Reported late 2021" value={pct(t?.report_rate)} />
                </div>
                <div className="mt-3 space-y-1.5">
                  {p.signature.slice(0, 5).map((s: any) => (
                    <div key={s.feature} className="grid grid-cols-[1fr_90px] items-center gap-2 text-[11.5px]">
                      <span className="truncate text-soft">{s.label}</span>
                      <div className="relative h-2 rounded-full bg-ink-700"><div className="absolute inset-y-0 left-1/2 w-px bg-white/30" />
                        <div className="absolute inset-y-0 rounded-full" style={{ background: s.z > 0 ? C.lime : C.red, left: s.z > 0 ? "50%" : `${50 - Math.min(50, Math.abs(s.z) * 20)}%`, width: `${Math.min(50, Math.abs(s.z) * 20)}%` }} /></div>
                    </div>
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-mute">bars: how far above (lime) or below (red) the average alert</p>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

function KTab({ c }: any) {
  const em = c.gmm.em_curve.map((v: number, i: number) => ({ it: i + 1, ll: v }));
  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <Card title="K-Means: elbow and silhouette" info="kmeans" explain="Within-group spread falls as K grows (elbow); silhouette asks how well-separated groups are. Highest silhouette among sensible K wins.">
        <div className="h-[260px]">
          <ResponsiveContainer>
            <LineChart data={c.kmeans.elbow} margin={{ top: 8, right: 4, left: -6, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="k" tickLine={false} axisLine={false} />
              <YAxis yAxisId="l" tickLine={false} axisLine={false} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
              <YAxis yAxisId="r" orientation="right" tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip content={<ChartTip fmt={(v: number) => (v > 10 ? num(v) : v.toFixed(3))} />} />
              <ReferenceLine yAxisId="l" x={c.kmeans.K} stroke="#fff" strokeOpacity={0.4} />
              <Line yAxisId="l" dataKey="inertia" name="spread (inertia)" stroke={C.azure} strokeWidth={2} />
              <Line yAxisId="r" dataKey="silhouette" name="silhouette" stroke={C.lime} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="Gaussian mixture: BIC" info="gmm" explain="BIC rewards fit but charges for extra parameters. The lowest point is the chosen number of fine profiles.">
        <div className="h-[260px]">
          <ResponsiveContainer>
            <LineChart data={c.gmm.bic} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="k" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
              <Tooltip content={<ChartTip fmt={(v: number) => num(v)} />} />
              <ReferenceLine x={c.gmm.K} stroke="#fff" strokeOpacity={0.4} />
              <Line dataKey="bic" name="BIC" stroke={C.lime} strokeWidth={2} />
              <Line dataKey="aic" name="AIC" stroke={C.violet} strokeWidth={1.5} strokeDasharray="4 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="EM: every step improves the fit" explain="Expectation-maximisation never decreases the likelihood — the curve only goes up, then flattens when it converges.">
        <div className="h-[260px]">
          <ResponsiveContainer>
            <AreaChart data={em} margin={{ top: 8, right: 8, left: -6, bottom: 0 }}>
              <defs><linearGradient id="em" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={C.lime} stopOpacity={0.3} /><stop offset="1" stopColor={C.lime} stopOpacity={0} /></linearGradient></defs>
              <XAxis dataKey="it" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(3)} />} labelFormatter={(v: any) => `iteration ${v}`} />
              <Area dataKey="ll" name="log-likelihood / alert" stroke={C.lime} fill="url(#em)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

function LatentTab({ c }: any) {
  let cum = 0;
  const scree = c.pca.ratio.slice(0, 30).map((r: number, i: number) => ({ k: i + 1, r, cum: (cum += r) }));
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="How many directions does the data need?" info="pca" explain={`Bars: information captured by each summary direction. Line: running total. ${c.pca.d80} directions keep 80%.`}>
          <div className="h-[280px]">
            <ResponsiveContainer>
              <ComposedChart data={scree} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                <XAxis dataKey="k" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => pct(v)} domain={[0, 1]} />
                <Tooltip cursor={false} content={<ChartTip fmt={(v: number) => pct(v, 1)} />} labelFormatter={(v: any) => `direction ${v}`} />
                <ReferenceLine y={0.8} stroke={C.red} strokeDasharray="4 4" />
                <Bar dataKey="r" name="this direction" radius={[4, 4, 0, 0]}>{scree.map((d: any) => <Cell key={d.k} fill={d.k <= c.pca.d80 ? C.lime : "#33373E"} />)}</Bar>
                <Line dataKey="cum" name="running total" stroke={C.azure} dot={false} strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <DS><p className="text-[11.5px] text-mute">Eigen-decomposition of the covariance matches SVD of the centred data: {String(c.pca.svd_matches_eigen)}.</p></DS>
        </Card>
        <Card title="Probabilistic PCA vs factor analysis" explain="Both are ‘hidden factor’ models. Scored on alerts they did not see: higher = describes new data better.">
          <div className="h-[280px]">
            <ResponsiveContainer>
              <LineChart data={c.dim_selection} margin={{ top: 8, right: 8, left: -4, bottom: 0 }}>
                <CartesianGrid />
                <XAxis dataKey="k" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} />
                <Tooltip content={<ChartTip fmt={(v: number) => v.toFixed(2)} />} labelFormatter={(v: any) => `${v} factors`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line dataKey="ppca" name="Probabilistic PCA" stroke={C.azure} strokeWidth={2} />
                <Line dataKey="fa" name="Factor analysis" stroke={C.lime} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <Card title={`${c.fa.n_factors} hidden factors (varimax rotated)`} explain="Each factor is a theme many behaviours share. Rotation makes each factor load on few behaviours so it can be named; it does not change the fit.">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {c.fa.factors.map((f: any) => (
            <div key={f.factor} className="inset p-3.5">
              <div className="mb-2 text-[12px] font-semibold text-lime">Factor {f.factor}</div>
              {f.top.slice(0, 5).map((t: any) => (
                <div key={t.label} className="flex items-center gap-2 py-0.5 text-[12px]">
                  <span className="flex-1 truncate text-soft">{t.label}</span>
                  <div className="h-1.5 w-16 rounded-full bg-ink-700"><div className="h-full rounded-full" style={{ width: `${Math.abs(t.loading) * 100}%`, background: t.loading > 0 ? C.lime : C.red }} /></div>
                  <span className="w-10 text-right num text-mute">{t.loading.toFixed(2)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <DS><p className="mt-3 text-[11.5px] text-mute">Log-likelihood unrotated {c.fa.loglik_unrotated.toFixed(4)} = rotated {c.fa.loglik_rotated.toFixed(4)} — rotation is unidentifiable from the data (any orthogonal rotation fits equally well). Mean noise variance {c.fa.noise_variance_mean.toFixed(3)}.</p></DS>
      </Card>
    </div>
  );
}

function TreeTab({ c }: any) {
  const h = c.hierarchical, d = h.dendrogram;
  const xs = d.icoord.flat(), ys = d.dcoord.flat();
  const xmax = Math.max(...xs), ymax = Math.max(...ys);
  const W = 900, H = 320;
  const sx = (x: number) => 20 + (x / (xmax + 5)) * (W - 30), sy = (y: number) => H - 30 - (y / ymax) * (H - 50);
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="Family tree of alerts (Ward linkage)" info="kmeans" explain={`Built on a sample of ${num(h.sample)} alerts. Similar alerts join low down; the height of a join shows how different the merged groups are.`}>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          {d.icoord.map((ic: number[], i: number) => (
            <path key={i} d={`M${sx(ic[0])},${sy(d.dcoord[i][0])} L${sx(ic[1])},${sy(d.dcoord[i][1])} L${sx(ic[2])},${sy(d.dcoord[i][2])} L${sx(ic[3])},${sy(d.dcoord[i][3])}`}
              fill="none" stroke={d.dcoord[i][1] > ymax * 0.55 ? C.lime : C.azure} strokeWidth="1.6" />
          ))}
          {d.leaves.map((l: string, i: number) => <text key={i} x={sx(5 + i * 10)} y={H - 12} fontSize="9" fill="#8D929C" textAnchor="middle">{l}</text>)}
        </svg>
      </Card>
      <Card title="Which linkage and distance?" explain="‘Single’ linkage chains everything into one giant group (look at the last column), so a high silhouette there is misleading. Ward gives balanced groups.">
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead className="text-mute"><tr className="text-left"><th className="py-1.5">Linkage</th><th>Distance</th><th>Cophenetic</th><th>Silhouette</th><th>Largest group</th></tr></thead>
            <tbody>{h.table.map((r: any, i: number) => (
              <tr key={i} className={clsx("border-t border-white/5", r.linkage === "ward" && "text-lime")}>
                <td className="py-1.5">{r.linkage}</td><td>{r.metric}</td><td className="num">{f2(r.cophenetic)}</td><td className="num">{f2(r.silhouette)}</td>
                <td className={clsx("num", r.largest_cluster_share > 0.9 && "text-danger")}>{pct(r.largest_cluster_share)}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function QualityTab({ c }: any) {
  const e = c.evaluation;
  const rows = [["K-Means", e["K-Means"]], ["Gaussian mixture", e["Gaussian mixture"]]];
  const metrics = [["silhouette", "Separation (silhouette)", "Higher = tighter, better-separated groups (−1 to 1)."], ["davies_bouldin", "Overlap (Davies–Bouldin)", "Lower = groups overlap less."],
    ["purity", "Purity", `Share of alerts in their group's majority outcome. Baseline if everything is one group: ${pct(e.baseline_purity)}.`], ["nmi", "Outcome information (NMI)", "0 = groups say nothing about the outcome; 1 = groups are the outcome."], ["ari", "Outcome agreement (ARI)", "0 = no better than chance."]];
  return (
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <Card title="Measured on alerts from 2020 and late 2021" explain="The first two rows describe the groups themselves; the last three ask whether the groups line up with the outcome.">
        <div className="overflow-x-auto"><table className="w-full text-[12.5px]">
          <thead className="text-mute"><tr className="text-left"><th className="py-2">Metric</th>{rows.map(([n]) => <th key={n as string} colSpan={2}>{n as string}</th>)}</tr>
            <tr className="text-left text-[11px]"><th /><th>2020</th><th>2021</th><th>2020</th><th>2021</th></tr></thead>
          <tbody>{metrics.map(([k, l, help]) => (
            <tr key={k} className="border-t border-white/5"><td className="py-2"><span className="flex items-center gap-1.5">{l}<InfoTip text={help} size={12} /></span></td>
              {rows.map(([n, v]: any) => <><td key={`${n}a`} className="num">{f3(v.train[k])}</td><td key={`${n}b`} className="num">{f3(v.test[k])}</td></>)}</tr>))}</tbody>
        </table></div>
      </Card>
      <Takeaway points={[
        <>Groups are compact but overlapping (silhouette ≈ {f2(e["K-Means"].test.silhouette)}) — behaviour is a continuum more than a set of boxes.</>,
        <>Outcome information is almost exactly zero (NMI {f3(e["K-Means"].test.nmi)}), and purity is no better than the {pct(e.baseline_purity)} you'd get with a single group.</>,
        <>K-Means and the mixture disagree a lot about who goes where (agreement {f2(e.kmeans_vs_gmm_ari)}), another sign there is no natural grouping.</>,
        <>This is why the profile features added nothing to the risk models (see Model Arena → Ablations).</>,
      ]} />
    </div>
  );
}
