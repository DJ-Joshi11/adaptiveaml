import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Layers, Lock, Scale, Timer } from "lucide-react";
import { useState } from "react";
import { Badge, C, Card, PageLoading, PhaseTag, Takeaway } from "../components/ui";
import { useData } from "../lib/data";
import { f3, num, pct } from "../lib/fmt";
import { G } from "../lib/glossary";
import { DS } from "../lib/mode";

const STAGES = [
  ["1", "Ingest", "Load and check 20,000 alerts and 16.2M transactions", "PBL-1"], ["2", "Features", "86 disclosed behaviours per alert", "PBL-1"],
  ["3", "Explore", "Class balance, separation tests, drift, Bayes", "PBL-1"], ["4", "Baseline", "Regression: how busy should this client be?", "PBL-1"],
  ["5", "Classify", "Six classifiers, one protocol, tuned forward in time", "PBL-1"], ["6", "Evaluate", "Champion on validation, CIs on test", "PBL-1"],
  ["7", "Explain", "Per-alert reasons, coefficient intervals", "PBL-1"], ["8", "Profiles", "PCA, factor analysis, K-Means, mixtures, hierarchy", "PBL-2"],
  ["9", "Sequences", "Hand-written HMMs: regimes and suspicion score", "PBL-2"], ["10", "Feature lab", "Do profiles / sequences help the classifiers?", "PBL-2"],
  ["11", "Ensembles", "Forest, boosting, AdaBoost by hand, stacking, BMA, GAM", "PBL-3"], ["12", "Final comparison", "All 21 models, champion–challenger", "PBL-3"],
  ["13", "Export", "Static data for this dashboard", "PBL-3"],
];

const RISKS = [
  ["Synthetic data", "SynthAML is generated from a real bank's data; patterns may be smoother or weaker than reality.", "Medium", "Results framed as a method study; protocol transfers to real data unchanged."],
  ["Weak signal ceiling", "All strong models land at PR-AUC ≈ 0.28–0.29; no family breaks through.", "High", "Reported honestly; champion–challenger rule prevents promoting noise."],
  ["Drift over time", "Many behaviours shift between 2020 and 2021.", "Medium", "Time-based split; per-quarter stability checks; retrain quarterly."],
  ["Selection on noise", "With 21 candidates, the validation winner can be a lucky one (as happened with Bagging).", "Medium", "New models must beat the incumbent with a paired interval above zero."],
  ["Explanation mismatch", "The queue is ranked by the SVM, reasons come from XGBoost/LR.", "Low", "Rank agreement reported; the transparent model's reasons are always one click away."],
  ["Over-trusting scores", "Analysts may treat 0.30 as a verdict.", "Medium", "Calibrated probabilities, uncertainty band and model-agreement flag on every alert."],
  ["Clipped timestamps", "~9% of transactions sit at the window edges.", "Low", "Edge piles removed before sequence modelling."],
];

export default function About() {
  const { data: s } = useData<any>("summary.json");
  const [q, setQ] = useState("");
  if (!s) return <PageLoading />;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card glass>
          <Badge tone="lime">AdaptiveAML · v3.0</Badge>
          <h2 className="mt-3 text-[26px] font-semibold leading-tight tracking-tight">Explainable, adaptive triage of anti-money-laundering alerts.</h2>
          <p className="mt-3 text-[14px] leading-relaxed text-soft">
            Banks raise far more AML alerts than analysts can investigate, and only about {pct(s.base_rate)} end up reported. AdaptiveAML ranks each new alert by its chance of being reported, explains why, flags when the models disagree, and shows how much review effort that saves — judged only on quarters the models never saw.
          </p>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <div className="inset p-3"><div className="text-[11px] text-mute">Catch @ 20% review</div><div className="text-[20px] font-semibold text-lime num">{pct(s.champion_row.recall_at_budget)}</div></div>
            <div className="inset p-3"><div className="text-[11px] text-mute">PR-AUC (random {f3(s.base_rate)})</div><div className="text-[20px] font-semibold num">{f3(s.champion_row.pr_auc)}</div></div>
            <div className="inset p-3"><div className="text-[11px] text-mute">Models compared</div><div className="text-[20px] font-semibold num">{s.n_candidates}</div></div>
          </div>
        </Card>
        <Card title="Kept honest by design">
          {[
            [Timer, "Time-based evaluation", "Train 2020 → choose on early 2021 → report once on late 2021."],
            [Lock, "No leakage", "Every scaler, imputer, cluster model, HMM and stacker is fitted on 2020 only; stacking uses out-of-fold scores."],
            [Scale, "Fair comparison", "Same features, same split, same bootstrap for every model; paired differences and McNemar tests."],
            [CheckCircle2, "Negative results kept", "Profiles, sequences and ensembles that did not help are reported as such."],
          ].map(([I, t, d]: any) => (
            <div key={t} className="flex gap-3 border-b border-white/5 py-2.5 last:border-0"><I size={17} className="mt-0.5 shrink-0 text-lime" /><div><div className="text-[13px] font-semibold">{t}</div><div className="text-[12px] text-mute">{d}</div></div></div>
          ))}
        </Card>
      </div>

      <Card title="Three reviews, one pipeline" explain="Each review added a layer; nothing earlier was removed.">
        <div className="grid gap-3 md:grid-cols-3">
          {[
            ["PBL-1", "Supervised foundations", "Regression baseline of expected activity; six classifiers (LR, Naive Bayes, LDA, KNN, tree, SVM); calibration; bootstrap intervals; per-alert reasons; benchmark against the published approach."],
            ["PBL-2", "Patterns in behaviour", "PCA vs SVD, factor analysis, probabilistic PCA; K-Means, Gaussian mixtures by EM, hierarchical clustering; hand-written HMMs with Viterbi paths and a sequence suspicion score; feature ablation."],
            ["PBL-3", "Ensembles & final verdict", "Random Forest, bagging, XGBoost/LightGBM with TreeSHAP, AdaBoost written out by hand, bias–variance experiment, GAM, voting, OOF stacking across all reviews, Bayesian model averaging; 21-model comparison."],
          ].map(([p, t, d], i) => (
            <motion.div key={p} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }} className="inset p-4">
              <PhaseTag phase={p} /><div className="mt-2 text-[14px] font-semibold">{t}</div><p className="mt-1.5 text-[12.5px] leading-relaxed text-mute">{d}</p>
            </motion.div>
          ))}
        </div>
      </Card>

      <Card title="The pipeline" explain="python run_pipeline.py runs every stage in order; each writes its results to artifacts/.">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
          {STAGES.map(([n, t, d, p]) => (
            <div key={n} className="inset relative p-3">
              <div className="flex items-center justify-between"><span className="grid h-6 w-6 place-items-center rounded-full text-[11px] font-bold text-ink-950" style={{ background: p === "PBL-1" ? C.azure : p === "PBL-2" ? C.violet : C.lime }}>{n}</span><span className="text-[10px] text-mute">{p}</span></div>
              <div className="mt-2 text-[13px] font-semibold">{t}</div><div className="mt-0.5 text-[11.5px] leading-snug text-mute">{d}</div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card title="Risk register" explain="What could make these results misleading, and what we did about it.">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-[12.5px]">
              <thead className="text-mute"><tr className="text-left"><th className="py-2">Risk</th><th>What it means</th><th>Level</th><th>Mitigation</th></tr></thead>
              <tbody>{RISKS.map(([r, d, l, m]) => (
                <tr key={r} className="border-t border-white/5 align-top"><td className="py-2.5 pr-3 font-medium"><span className="flex items-center gap-1.5"><AlertTriangle size={13} className={l === "High" ? "text-danger" : l === "Medium" ? "text-amber" : "text-mute"} />{r}</span></td>
                  <td className="pr-3 text-soft">{d}</td><td className="pr-3"><Badge tone={l === "High" ? "red" : l === "Medium" ? "amber" : "gray"}>{l}</Badge></td><td className="text-mute">{m}</td></tr>))}</tbody>
            </table>
          </div>
        </Card>
        <div className="space-y-4">
          <Takeaway title="Limitations" points={[
            <>Synthetic data: conclusions are about the method, not a specific bank.</>,
            <>Alert dates are only known to the quarter, so day-level timing around the alert is approximate.</>,
            <>No investigator feedback loop yet: a production system would retrain each quarter on newly closed alerts.</>,
          ]} />
          <Card title="Built with">
            <div className="flex flex-wrap gap-1.5">{["Python", "polars", "scikit-learn", "XGBoost", "LightGBM", "imbalanced-learn", "SciPy", "hand-written HMM & AdaBoost", "React", "TypeScript", "Tailwind CSS", "Framer Motion", "Recharts", "Radix UI", "Vercel"].map((t) => <Badge key={t} tone="gray">{t}</Badge>)}</div>
            <DS><p className="mt-3 text-[11.5px] text-mute">Data: Jensen et al. (2023), “SynthAML: a synthetic data set to benchmark anti-money laundering methods”, Scientific Data 10, 661.</p></DS>
          </Card>
        </div>
      </div>

      <Card title="Glossary" explain="Every term used on the dashboard, in plain words and precisely." action={<input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search terms" className="h-9 rounded-xl border border-white/10 bg-ink-900 px-3 text-[12.5px] focus:outline-none" />}>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {Object.values(G).filter((t) => !q || `${t.title} ${t.plain}`.toLowerCase().includes(q.toLowerCase())).map((t) => (
            <div key={t.title} className="inset p-3.5"><div className="flex items-center gap-2 text-[13px] font-semibold"><Layers size={13} className="text-lime" />{t.title}</div><p className="mt-1 text-[12px] leading-relaxed text-soft">{t.plain}</p><DS><p className="mt-1.5 font-mono text-[10.5px] leading-relaxed text-mute">{t.tech}</p></DS></div>
          ))}
        </div>
      </Card>
      <p className="text-center text-[11.5px] text-mute">AdaptiveAML · SRM Institute of Science and Technology · {num(s.data.alerts)} alerts · {num(s.data.transactions)} transactions</p>
    </div>
  );
}
