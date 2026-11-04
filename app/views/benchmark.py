"""Benchmark: the existing published approach vs AdaptiveAML on identical alerts."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

ev = ui.evaluation()
B = ev["benchmark"]
df = pd.DataFrame(B["rows"])
old, new = df[df.family.str.startswith("Existing")], df[df.family.str.startswith("Adaptive")]

ui.setup("Benchmark", "Model",
         "The dataset's own published recipe, re-implemented, against AdaptiveAML. Same alerts, same split: "
         f"trained on 2020, tested on all {B['n']:,} alerts of 2021.")

bo, bn = old.loc[old.pr_auc.idxmax()], new.loc[new.pr_auc.idxmax()]
c = st.columns(4)
ui.kpi(c[0], "Existing approach, best PR-AUC", f"{bo.pr_auc:.3f}", bo.model, ui.MUTED)
ui.kpi(c[1], "AdaptiveAML, best PR-AUC", f"{bn.pr_auc:.3f}", bn.model, ui.ORANGE)
ui.kpi(c[2], "Existing approach, best ROC-AUC", f"{old.roc_auc.max():.3f}", old.loc[old.roc_auc.idxmax()].model, ui.MUTED)
ui.kpi(c[3], "AdaptiveAML, best ROC-AUC", f"{new.roc_auc.max():.3f}", new.loc[new.roc_auc.idxmax()].model, ui.BLUE)
st.write("")


def bars(metric: str, title: str, ref: float):
    fig = go.Figure()
    for d, name, color in ((old, "Existing approach: 56 summary statistics", ui.rgba(ui.MUTED, 0.7)),
                           (new, "AdaptiveAML: profile + behaviour", ui.ORANGE if metric == "pr_auc" else ui.BLUE)):
        d = d.sort_values(metric)
        fig.add_bar(y=[m + ("  · existing" if name.startswith("Existing") else "") for m in d.model], x=d[metric],
                    orientation="h", name=name, marker_color=color, text=[f"{v:.3f}" for v in d[metric]], textposition="inside", insidetextanchor="start",
                    error_x=dict(type="data", symmetric=False, color=ui.TEXT, thickness=1,
                                 array=[ci[1] - v for ci, v in zip(d[f"{metric}_ci"], d[metric])],
                                 arrayminus=[v - ci[0] for ci, v in zip(d[f"{metric}_ci"], d[metric])]))
    ui.ref_line(fig, ref)
    ui.fig_layout(fig, title, 470, xaxis=dict(title=title.split(":")[0]), margin=dict(l=220, r=20, t=48, b=44))
    return fig


a, b = st.columns(2)
with a: ui.show(bars("pr_auc", "PR-AUC: ranking quality on the rare class", B["base_rate"]))
with b: ui.show(bars("roc_auc", "ROC-AUC: the only metric the published benchmark reports", 0.5))

gap_lo, gap_hi = bn.pr_auc_ci[0] - bo.pr_auc_ci[1], bn.pr_auc_ci[1] - bo.pr_auc_ci[0]
verdict = ("ahead of" if gap_lo > 0 else "behind" if gap_hi < 0 else "statistically level with")
ui.note(f"On ranking quality AdaptiveAML's best model is <b>{verdict}</b> the best existing-approach model "
        f"({bn.pr_auc:.3f} vs {bo.pr_auc:.3f}; the 95% intervals {'do not ' if verdict != 'statistically level with' else ''}overlap). "
        "What AdaptiveAML adds is listed below: a time-ordered protocol, calibrated scores, an operating point and per-alert reasons.",
        ui.ORANGE)

if B["published"]:
    st.markdown("##### Published figures")
    st.dataframe(pd.DataFrame([{"Model": k, "Published ROC-AUC": v} for k, v in B["published"].items()]),
                 hide_index=True, width="stretch")
    st.caption(B["published_source"])

st.markdown("##### What each approach delivers")
yes, no = "Yes", "No"
st.dataframe(pd.DataFrame([
    {"Capability": "Features named and documented one by one", "Existing benchmark": "Generic statistics", "AdaptiveAML": yes},
    {"Capability": "Behaviour-over-time features (bursts, dormancy, pass-through)", "Existing benchmark": "Not tested", "AdaptiveAML": "Tested: no measurable gain on this data"},
    {"Capability": "Separate validation period for tuning", "Existing benchmark": no, "AdaptiveAML": yes},
    {"Capability": "Precision-recall evaluation", "Existing benchmark": no, "AdaptiveAML": yes},
    {"Capability": "Confidence intervals and significance tests", "Existing benchmark": "Spread across runs", "AdaptiveAML": yes},
    {"Capability": "Calibrated risk score", "Existing benchmark": no, "AdaptiveAML": yes},
    {"Capability": "Operating point tied to analyst capacity", "Existing benchmark": no, "AdaptiveAML": yes},
    {"Capability": "Per-alert explanation", "Existing benchmark": no, "AdaptiveAML": yes},
]), hide_index=True, width="stretch")
ui.note("The existing approach is our re-implementation of the recipe described in the SynthAML paper (minimum, mean, median, "
        "maximum, standard deviation, count and sum of transaction size for every direction and channel), with library-default "
        "model settings. It is a like-for-like baseline on this data, not a copy of the paper's reported figures.")
