"""Activity baseline: what each client was expected to do, and how far they departed from it."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

rg = ui.regression()
tbl = pd.DataFrame(rg["table"])
naive = tbl.iloc[0]
best = tbl.iloc[1:].sort_values("rmse_valid").iloc[0]

ui.setup("Activity Baseline", "Model",
         "A per-client expectation of activity in the 30 days before the alert, learned from the 11 months before that. "
         "The departure from it is the burst signal used by the risk models.")

c = st.columns(4)
ui.kpi(c[0], "Best baseline model", best.model, "lowest validation error", ui.CYAN)
ui.kpi(c[1], "Variance explained (R²)", f"{best.r2_test:.2f}", "on held-out quarters, log scale", ui.BLUE)
ui.kpi(c[2], "Error vs naive rule", f"{100 * (best.rmse_test / naive.rmse_test - 1):+.1f}%", "naive: same daily rate as before", ui.GREEN)
b = rg["burst_by_class"]["valid"]
ui.kpi(c[3], "Burst score, reported vs dismissed", f"{b['z_report']:+.2f} / {b['z_dismiss']:+.2f}", "mean, validation quarters", ui.ORANGE)
st.write("")

tabs = st.tabs(["Model comparison", "Complexity and regularisation", "Expected vs actual"])
with tabs[0]:
    st.dataframe(pd.DataFrame([{"Model": r.model, "Penalty strength": r.alpha, "MAE (valid)": r.mae_valid, "RMSE (valid)": r.rmse_valid,
                                "R² (valid)": r.r2_valid, "RMSE (test)": r.rmse_test, "R² (test)": r.r2_test}
                               for r in tbl.itertuples()]), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in
                                ("MAE (valid)", "RMSE (valid)", "R² (valid)", "RMSE (test)", "R² (test)")})
    k = rg["ridge_coefs"][::-1]
    fig = go.Figure(go.Bar(x=[x["coef"] for x in k], y=[ui.label(x["feature"]) for x in k], orientation="h",
                           marker_color=[ui.CYAN if x["coef"] > 0 else ui.VIOLET for x in k]))
    ui.fig_layout(fig, "What predicts recent activity (regularised linear model)", 400, margin=dict(l=240, r=20, t=48, b=44))
    ui.show(fig)
    lasso = tbl[tbl.model == "Lasso"].iloc[0]
    ui.note(f"Target is log(1 + transactions in the last 30 days). Inputs come only from days 30 to 365, so the model cannot see "
            f"what it is predicting. The sparse model set {int(lasso.zero_coefs)} weights exactly to zero.")
with tabs[1]:
    a, b_ = st.columns(2)
    d = pd.DataFrame(rg["degree"])
    fig = go.Figure()
    fig.add_scatter(x=d.degree, y=d.rmse_train, mode="lines+markers", name="Training quarters", line=dict(color=ui.BLUE, width=2))
    fig.add_scatter(x=d.degree, y=d.rmse_valid, mode="lines+markers", name="Later quarters", line=dict(color=ui.ORANGE, width=2))
    ui.fig_layout(fig, "Model flexibility: error vs polynomial degree", 400, xaxis=dict(title="Polynomial degree", dtick=1),
                  yaxis=dict(title="RMSE", type="log"))
    with a: ui.show(fig)
    d = pd.DataFrame(rg["lambda"])
    fig = go.Figure()
    fig.add_scatter(x=d.alpha, y=d.rmse_train, mode="lines+markers", name="Training quarters", line=dict(color=ui.BLUE, width=2))
    fig.add_scatter(x=d.alpha, y=d.rmse_valid, mode="lines+markers", name="Later quarters", line=dict(color=ui.ORANGE, width=2))
    ui.fig_layout(fig, "Penalty strength: error vs λ (degree-2 model)", 400, xaxis=dict(title="λ", type="log"), yaxis=dict(title="RMSE"))
    with b_: ui.show(fig)
    ui.note("Left: a more flexible model always fits the training quarters better, but past a point it gets worse on later "
            f"quarters. Terms grow from {rg['degree'][0]['n_terms']} to {rg['degree'][-1]['n_terms']}. "
            "Right: the penalty pulls the flexible model back; too little over-fits, too much under-fits.")
with tabs[2]:
    s = rg["scatter"]
    act, exp, sd, y = (np.array(s[k], dtype=float) for k in ("actual", "expected", "sd", "y"))
    fig = go.Figure()
    lo, hi = float(min(act.min(), exp.min())), float(max(act.max(), exp.max()))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="As expected", line=dict(color=ui.MUTED, dash="dash", width=1))
    for lab, col, m in (("Dismissed", ui.DISMISS, y == 0), ("Reported", ui.REPORT, y == 1)):
        fig.add_scatter(x=exp[m], y=act[m], mode="markers", name=lab, marker=dict(color=col, size=5, opacity=0.55))
    ui.fig_layout(fig, "Actual vs expected recent activity (validation quarters, log scale)", 460,
                  xaxis=dict(title="Expected"), yaxis=dict(title="Actual"))
    a, b_ = st.columns([3, 2])
    with a: ui.show(fig)
    z = (act - exp) / sd
    fig = go.Figure()
    for lab, col, m in (("Dismissed", ui.DISMISS, y == 0), ("Reported", ui.REPORT, y == 1)):
        fig.add_histogram(x=np.clip(z[m], -5, 5), name=lab, marker_color=col, opacity=0.7, histnorm="probability density", nbinsx=40)
    ui.fig_layout(fig, "Burst score by outcome", 460, barmode="overlay", xaxis=dict(title="(actual - expected) / model uncertainty"))
    with b_: ui.show(fig)
    ui.note("Points above the line were busier than their own history predicted. Dividing by the model's uncertainty makes the "
            "signal comparable between a steady client and an erratic one.")
