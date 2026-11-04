"""Feature lab: do behavioural profiles and regime sequences make the risk models better?"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

en = ui.enrichment()
if not en:
    ui.setup("Feature Lab", "Model")
    st.info("Run stage 10 first:  python run_pipeline.py --stage 10")
    st.stop()
R = pd.DataFrame(en["rows"])
base = en["base_rate_test"]
variants = ["Baseline", "+ Profiles", "+ Sequences", "+ Both"]
vcol = {"Baseline": ui.MUTED, "+ Profiles": ui.VIOLET, "+ Sequences": ui.CYAN, "+ Both": ui.ORANGE}
best = R.loc[R.pr_auc.idxmax()]
helps = R[(R.variant != "Baseline") & (R.verdict == "adds signal")]

ui.setup("Feature Lab", "Model",
         f"The six risk models retrained with profile and sequence features added, on the same split and the same "
         f"{en['n_test']:,} test alerts. Each gain is a paired bootstrap difference against that model's own baseline.")

c = st.columns(4)
ui.kpi(c[0], "Model × feature-set combinations", "24", "6 models × 4 feature sets", ui.BLUE, count=24)
ui.kpi(c[1], "Combinations that add signal", f"{len(helps)}", "95% interval of the gain above zero", ui.GREEN if len(helps) else ui.RED, count=len(helps))
ui.kpi(c[2], "Best combination", f"{best.pr_auc:.3f}", f"{best.model}, {best.variant}", ui.ORANGE)
bl = R[(R.variant == "Baseline")].pr_auc.max()
ui.kpi(c[3], "Best baseline (stages 2-7 features)", f"{bl:.3f}", f"no skill {base:.3f}", ui.MUTED)
st.write("")

fig = go.Figure()
for v in variants:
    d = R[R.variant == v]
    fig.add_bar(x=d.model, y=d.pr_auc, name=v, marker_color=vcol[v],
                error_y=dict(type="data", symmetric=False, array=[ci[1] - p for ci, p in zip(d.pr_auc_ci, d.pr_auc)],
                             arrayminus=[p - ci[0] for ci, p in zip(d.pr_auc_ci, d.pr_auc)], color=ui.TEXT, thickness=1))
fig.add_hline(y=base, line=dict(color=ui.RED, dash="dash", width=1.5))
ui.fig_layout(fig, "PR-AUC on 2021 Q3-Q4 by model and feature set (95% bootstrap intervals)", 430, barmode="group",
              yaxis=dict(title="PR-AUC", range=[0, max(R.pr_auc) * 1.25]))
ui.show(fig)

a, b = st.columns([3, 2])
with a:
    D = R[R.variant != "Baseline"]
    fig = go.Figure()
    for v in variants[1:]:
        d = D[D.variant == v]
        fig.add_scatter(x=d.gain, y=d.model, mode="markers", name=v, marker=dict(size=12, color=vcol[v]),
                        error_x=dict(type="data", symmetric=False, array=[ci[1] - g for ci, g in zip(d.gain_ci, d.gain)],
                                     arrayminus=[g - ci[0] for ci, g in zip(d.gain_ci, d.gain)], color=vcol[v], thickness=1.2))
    fig.add_vline(x=0, line=dict(color=ui.TEXT, width=1))
    ui.fig_layout(fig, "Change in PR-AUC against each model's baseline", 400, xaxis=dict(title="Gain (paired bootstrap, 95% interval)"),
                  margin=dict(l=160, r=20, t=48, b=44))
    ui.show(fig)
with b:
    st.markdown("##### Scores without any classifier")
    S = pd.DataFrame(en["standalone"])
    fig = go.Figure(go.Bar(x=S.pr_auc, y=S.score, orientation="h", marker_color=[ui.VIOLET, ui.VIOLET, ui.CYAN, ui.CYAN, ui.CYAN],
                           text=[f"{v:.3f}" for v in S.pr_auc], textposition="inside", insidetextanchor="start",
                           error_x=dict(type="data", symmetric=False, array=[ci[1] - p for ci, p in zip(S.pr_auc_ci, S.pr_auc)],
                                        arrayminus=[p - ci[0] for ci, p in zip(S.pr_auc_ci, S.pr_auc)], color=ui.TEXT, thickness=1)))
    ui.ref_line(fig, base)
    ui.fig_layout(fig, "", 330, xaxis=dict(title="PR-AUC"), margin=dict(l=250, r=10, t=30, b=40))
    ui.show(fig)

T = R.assign(**{"95% interval": [f"{lo:.3f} to {hi:.3f}" for lo, hi in R.pr_auc_ci],
                "Gain interval": [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in R.gain_ci]})
st.dataframe(T[["model", "variant", "pr_auc", "95% interval", "recall_at_budget", "gain", "Gain interval", "verdict"]].rename(
    columns={"model": "Model", "variant": "Features", "pr_auc": "PR-AUC", "recall_at_budget": "Caught @ 20%", "gain": "Gain",
             "verdict": "Verdict"}), hide_index=True, width="stretch",
    column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f"), "Gain": st.column_config.NumberColumn(format="%+.3f"),
                   "Caught @ 20%": st.column_config.NumberColumn(format="percent")})
ui.note("<b>Reading this page.</b> Baseline rows reproduce the earlier results exactly. If an interval of the gain crosses zero, the "
        "added features made no measurable difference. Profiles and regimes were learned without the outcome, so they only help if "
        "the groups they find happen to differ in report rate - on this data they barely do.", ui.ORANGE)
