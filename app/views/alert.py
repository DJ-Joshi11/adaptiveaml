"""Alert detail: the evidence behind one score."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui

ev = ui.evaluation()
Q = ui.table("scored_alerts.parquet").sort_values("risk", ascending=False)
F = ui.table("alert_features.parquet").set_index("alert_id")
C = ui.table("contributions.parquet").set_index("alert_id")
E = ui.table("expected_activity.parquet").set_index("alert_id")
T = ui.table("alert_daily.parquet")
champ = next(r for r in ev["table"] if r["model"] == ev["champion"])

ui.setup("Alert Detail", "Operations", "Why this alert is ranked where it is.")

test = Q[Q.role == "test"]
s = st.columns([2, 5])
ids = test.alert_id.tolist()
want = st.session_state.get("alert_pick")
aid = s[0].selectbox("Alert", ids, index=ids.index(want) if want in ids else 0, format_func=lambda a: f"#{a}")
st.session_state["alert_pick"] = aid
row, f = Q.set_index("alert_id").loc[aid], F.loc[aid]
rank = int((test.risk > row.risk).sum()) + 1
tags = ui.pill(row.band, ui.BAND_COLORS[row.band])
if row.low_confidence:
    tags += " " + ui.pill("Low confidence: needs human judgement", ui.YELLOW)
s[1].markdown(f"<div style='padding-top:30px'>{tags}</div>", unsafe_allow_html=True)

contrib = C.loc[aid].drop("intercept")
up, down = contrib.sort_values(ascending=False), contrib.sort_values()
tr = F[F.role == "train"]
pct_of = lambda col: float((tr[col] < f[col]).mean())

c = st.columns(4)
ui.kpi(c[0], "Risk score", f"{row.risk:.2f}", f"rank {rank:,} of {len(test):,}", ui.BAND_COLORS[row.band])
ui.kpi(c[1], "Strongest reason for concern", ui.label(up.index[0]),
       f"higher than {ui.pct(pct_of(up.index[0]))} of past alerts" if up.iloc[0] > 0 else "no factor raises this score", ui.ORANGE)
ui.kpi(c[2], "Strongest reassuring factor", ui.label(down.index[0]),
       f"higher than {ui.pct(pct_of(down.index[0]))} of past alerts" if down.iloc[0] < 0 else "no factor lowers this score", ui.BLUE)
ui.kpi(c[3], "Transactions in the year", f"{int(f.h_n + f.r_n):,}", f"{int(f.r_n):,} in the last 30 days", ui.CYAN)
st.write("")
ui.takeaway([f"This alert is ranked <b>{rank:,} of {len(test):,}</b> with a risk score of <b>{row.risk:.2f}</b> "
             f"(about {row.risk:.0%} of alerts like it were reported).",
             f"Biggest reason for concern: <b>{ui.label(up.index[0]).lower()}</b>, higher than {pct_of(up.index[0]):.0%} of past alerts.",
             f"Biggest reassuring sign: <b>{ui.label(down.index[0]).lower()}</b>."], "This alert in one glance")

left, right = st.columns([2, 3])
with left:
    k = contrib.sort_values(key=abs, ascending=False).head(10)[::-1]
    fig = go.Figure(go.Bar(x=k.values, y=[ui.label(n) for n in k.index], orientation="h",
                           marker_color=[ui.ORANGE if v > 0 else ui.BLUE for v in k.values]))
    ui.fig_layout(fig, "Why this score: raises risk (orange) or lowers it (blue)", 400,
                  xaxis=dict(title="Contribution (log-odds)"), margin=dict(l=250, r=20, t=48, b=44))
    ui.show(fig)
with right:
    st.markdown("##### The evidence behind those factors")
    keys = contrib.sort_values(key=abs, ascending=False).head(8).index
    tbl = [{"Factor": ui.label(kx), "Effect": "Raises risk" if contrib[kx] > 0 else "Lowers risk",
            "This alert": float(f[kx]), "Typical (median)": float(tr[kx].median()), "Percentile": pct_of(kx)} for kx in keys]
    st.dataframe(tbl, hide_index=True, width="stretch", height=352, column_config={
        "This alert": st.column_config.NumberColumn(format="%.2f"), "Typical (median)": st.column_config.NumberColumn(format="%.2f"),
        "Percentile": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# ---- behavioural profile and regime sequence for this alert
if ui.has("clusters.parquet") and ui.has("sequence_features.parquet"):
    import pandas as pd
    st.markdown("##### Patterns")
    CL = ui.table("clusters.parquet").set_index("alert_id")
    SF = ui.table("sequence_features.parquet").set_index("alert_id")
    VP = ui.table("viterbi_paths.parquet").set_index("alert_id")
    clj, sqj = ui.clustering(), ui.sequences()
    profs = {p["cluster"]: p for p in clj["profiles"]["kmeans"]["train"]}
    c = st.columns(4)
    if aid in CL.index:
        k = int(CL.loc[aid, "kmeans"])
        ui.kpi(c[0], "Behavioural profile", f"P{k + 1}", profs[k]["name"][:70], ui.SERIES[k % 12])
        ui.kpi(c[1], "Report rate in this profile", f"{profs[k]['report_rate']:.0%}", "2020 alerts with the same profile", ui.ORANGE,
               ring=profs[k]["report_rate"])
    if aid in SF.index:
        sfr = SF.loc[aid]
        dom = int(np.argmax([sfr[f"hmm_occ_{j}"] for j in range(sqj["K"])]))
        ui.kpi(c[2], "Dominant regime", f"R{dom + 1}", sqj["states"][dom]["name"], ui.SERIES[dom % 12])
        pct = float((SF.hmm_llr < sfr.hmm_llr).mean())
        ui.kpi(c[3], "Sequence suspicion score", f"{sfr.hmm_llr:+.3f}", f"higher than {pct:.0%} of alerts", ui.VIOLET, ring=pct)
    if aid in VP.index:
        vp = VP.loc[aid]
        stv, days = np.asarray(vp.states), np.asarray(vp.days)
        nm = [ui.state_label(j, sqj["states"]) for j in range(sqj["K"])]
        fig = go.Figure()
        for j in range(sqj["K"]):
            m = stv == j
            if m.any():
                fig.add_scattergl(x=-days[m], y=[nm[j]] * int(m.sum()), mode="markers", showlegend=False,
                                  marker=dict(color=ui.SERIES[j % 12], size=7, symbol="line-ns-open", line=dict(width=2, color=ui.SERIES[j % 12])),
                                  hovertemplate="%{x:.1f} days before alert<extra></extra>")
        ui.fig_layout(fig, "Regime path across the year (most probable sequence of hidden regimes)", 250,
                      xaxis=dict(title="Days before alert", range=[-366, 1]), margin=dict(l=230, r=20, t=44, b=40),
                      yaxis=dict(categoryorder="array", categoryarray=nm[::-1]))
        ui.show(fig)

st.markdown("##### Activity context")
st.caption("Background for the investigator. How busy the client was against their own history did not turn out to "
           "separate reported from dismissed alerts in this data, so it carries little weight in the score.")
left, right = st.columns([3, 1])
with left:
    t = T[T.alert_id == aid].sort_values("day")
    fig = go.Figure()
    fig.add_vrect(x0=-30, x1=0.5, fillcolor=ui.rgba(ui.ORANGE, 0.10), line_width=0)
    fig.add_bar(x=-t.day, y=t.n - t.n_credit, name="Debits", marker_color=ui.rgba(ui.BLUE, 0.85))
    fig.add_bar(x=-t.day, y=t.n_credit, name="Credits", marker_color=ui.rgba(ui.GREEN, 0.85))
    ui.fig_layout(fig, "Daily transactions in the year before the alert (shaded: last 30 days)", 340, barmode="stack", bargap=0.05,
                  xaxis=dict(title="Days before alert", range=[-366, 1]), yaxis=dict(title="Transactions"))
    ui.show(fig)
exp, sd, z = E.loc[aid, "expected_log"], E.loc[aid, "expected_sd"], f.burst_z
ui.kpi(right, "Last 30 days vs expected", f"{int(f.r_n):,} vs {np.expm1(exp):,.0f}",
       f"likely range {max(np.expm1(exp - 1.96 * sd), 0):,.0f} to {np.expm1(exp + 1.96 * sd):,.0f} ({z:+.1f}σ)", ui.MUTED)

agree = ui.explain().get("rank_agreement_with_champion", 1)
same = ev["champion"] == "Logistic Regression"
ui.note("Factors are exact contributions: each one is the model weight multiplied by how unusual this alert is on that factor. "
        + ("They come from the same model that produced the risk score. " if same else
           f"They come from the transparent linear model, whose ranking agrees with the scoring model at {agree:.2f} rank correlation. ")
        + "Transaction sizes are relative (standardised), not currency amounts. The synthetic data piles transactions up at the two "
        "edges of the one-year window, so a spike on the last or first day is a property of the dataset, not of the client.")
