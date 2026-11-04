"""Model reasoning: what the models learned, in terms an investigator can challenge."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

ex = ui.explain()
ui.setup("Model Reasoning", "Model", "Which behaviours drive risk scores overall. Per-alert reasons are on the Alert Detail page.")

_c = ex["coefs"]
_up = [c["label"].lower() for c in _c if c["coef"] > 0 and c["stable"]][:2]
_dn = [c["label"].lower() for c in _c if c["coef"] < 0 and c["stable"]][:2]
ui.takeaway([f"What pushes risk <b>up</b>: {', '.join(_up) if _up else 'no stable factor'}.",
             f"What pulls risk <b>down</b>: {', '.join(_dn) if _dn else 'no stable factor'}.",
             "Overall the models lean on the <b>size pattern of wire and card transactions</b> far more than on timing or activity bursts."])
a, b = st.columns(2)
with a:
    k = ex["coefs"][:15][::-1]
    fig = go.Figure(go.Bar(
        x=[x["coef"] for x in k], y=[x["label"] for x in k], orientation="h",
        marker_color=[(ui.ORANGE if x["coef"] > 0 else ui.BLUE) if x["stable"] else ui.rgba(ui.MUTED, 0.5) for x in k],
        error_x=dict(type="data", symmetric=False, color=ui.TEXT, thickness=1,
                     array=[x["hi"] - x["coef"] for x in k], arrayminus=[x["coef"] - x["lo"] for x in k])))
    fig.add_vline(x=0, line=dict(color=ui.TEXT, width=1))
    ui.fig_layout(fig, "Linear model weights (orange raises risk, blue lowers it)", 520,
                  xaxis=dict(title="Log-odds per standard deviation"), margin=dict(l=240, r=20, t=48, b=44))
    ui.show(fig)
with b:
    k = ex["permutation"][:15][::-1]
    fig = go.Figure(go.Bar(x=[x["importance"] for x in k], y=[x["label"] for x in k], orientation="h", marker_color=ui.VIOLET,
                           error_x=dict(type="data", array=[x["sd"] for x in k], color=ui.TEXT, thickness=1)))
    ui.fig_layout(fig, f"What the in-service model relies on ({ex['permutation_model']})", 520,
                  xaxis=dict(title="PR-AUC lost when scrambled"), margin=dict(l=240, r=20, t=48, b=44))
    ui.show(fig)
ui.note("Grey bars have an interval that crosses zero: the direction of that effect is not reliable. Correlated behaviours "
        "share weight between them, so read related bars as a group rather than as a strict ranking.")

a, b = st.columns([2, 3])
with a:
    k = ex["tree_importance"][::-1]
    fig = go.Figure(go.Bar(x=[x["importance"] for x in k], y=[x["label"] for x in k], orientation="h", marker_color=ui.GREEN))
    ui.fig_layout(fig, "Decision tree: share of splitting power", 440, margin=dict(l=240, r=20, t=48, b=44))
    ui.show(fig)
with b:
    st.markdown("##### Decision tree as readable rules (top levels)")
    st.code(ex["tree_rules"], language="text")
    st.caption("Values are in standardised units. class: 1 = reported, 0 = dismissed.")
