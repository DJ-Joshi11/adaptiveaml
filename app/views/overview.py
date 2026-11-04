"""Overview: the state of the alert queue and of the model, on one screen."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui

ev, dq = ui.evaluation(), ui.quality()
Q = ui.table("scored_alerts.parquet")
test = Q[Q.role == "test"]
champ = next(r for r in ev["table"] if r["model"] == ev["champion"])

ui.setup("Overview", "Operations", "Held-out alerts from quarters the model never saw, ranked by risk of being reported.")

c = st.columns(4)
by_q = [b[ev["champion"]] / b["base_rate"] for b in ev["by_quarter"]]
qcount = test.groupby("quarter").size().values
ui.kpi(c[0], "Alerts in queue", f"{len(test):,}", f"{', '.join(sorted(test.quarter.unique()))}", ui.BLUE, count=len(test),
       spark=[b["n"] for b in ev["by_quarter"]])
ui.kpi(c[1], "Flagged: review now", f"{int((test.band == 'Review now').sum()):,}",
       f"top {ui.pct(ev['budget'])} review budget", ui.ORANGE, count=int((test.band == 'Review now').sum()))
ui.kpi(c[2], "Reports caught in flagged set", ui.pct(champ["recall_at_budget"]),
       f"random review would catch {ui.pct(ev['budget'])}", ui.GREEN, count=round(100 * champ["recall_at_budget"]), suffix="%",
       ring=champ["recall_at_budget"])
ui.kpi(c[3], "Ranking quality (PR-AUC)", f"{champ['pr_auc']:.3f}",
       f"{champ['lift']:.1f}x the no-skill level of {ev['base_rate_test']:.3f}", ui.VIOLET, spark=by_q)
st.write("")
ui.takeaway([
    f"About <b>{ev['base_rate_test']:.0%}</b> of alerts end up reported. If analysts open alerts in random order, the first 20% of the queue "
    f"finds 20% of them; opened in AdaptiveAML's order, the same effort finds <b>{ui.pct(champ['recall_at_budget'])}</b>.",
    f"That is {champ['recall_at_budget'] / ev['budget']:.1f}× the reported cases for the same analyst time - useful, but not a replacement "
    "for review: most of the queue still needs a human eye.",
    "Every number here comes from alerts the model never saw while it was learning."])
ui.plain("Each alert gets a **risk score** between 0 and 1: the estimated chance it will be reported. Sorting by that score puts the "
         "likeliest cases first. The **workload curve** below shows how many of the truly reported alerts you find as you work down the "
         "sorted list; the dotted line is what random order would give.")

left, right = st.columns([3, 2])
with left:
    g = ev["curves"][ev["champion"]]["gain"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random review order",
                    line=dict(color=ui.MUTED, dash="dot", width=1.5))
    fig.add_scatter(x=g[0], y=g[1], mode="lines", name=f"AdaptiveAML ({ev['champion']})",
                    line=dict(color=ui.ORANGE, width=2.5), fill="tonexty", fillcolor=ui.rgba(ui.ORANGE, 0.10),
                    fillpattern=dict(shape="/", fgcolor=ui.rgba(ui.ORANGE, 0.35), size=6))
    fig.add_vline(x=ev["budget"], line=dict(color=ui.LINE, dash="dash"))
    fig.add_annotation(x=ev["budget"], y=champ["recall_at_budget"], text=f" {ui.pct(champ['recall_at_budget'])} caught",
                       showarrow=False, xanchor="left", font=dict(color=ui.TEXT))
    ui.fig_layout(fig, "Workload curve: reports caught vs share of queue reviewed", 380,
                  xaxis=dict(title="Share of queue reviewed", tickformat=".0%"),
                  yaxis=dict(title="Share of reported alerts caught", tickformat=".0%"))
    ui.show(fig)
with right:
    fig = go.Figure()
    for lab, col, mask in (("Dismissed", ui.DISMISS, test.y == 0), ("Reported", ui.REPORT, test.y == 1)):
        fig.add_histogram(x=test.risk[mask], name=lab, marker_color=col, opacity=0.75, nbinsx=40, histnorm="probability")
    fig.add_vline(x=champ["threshold"], line=dict(color=ui.TEXT, dash="dash", width=1))
    ui.fig_layout(fig, "Risk score by eventual outcome", 380, barmode="overlay",
                  xaxis=dict(title="Risk score"), yaxis=dict(title="Share of alerts", tickformat=".0%"))
    ui.show(fig)

left, right = st.columns([2, 3])
with left:
    bq = dq["by_quarter"]
    fig = go.Figure()
    fig.add_bar(x=[b["quarter"] for b in bq], y=[b["alerts"] for b in bq], name="Alerts", marker_color=ui.rgba(ui.BLUE, 0.55))
    fig.add_scatter(x=[b["quarter"] for b in bq], y=[b["report_rate"] for b in bq], name="Report rate", yaxis="y2",
                    mode="lines+markers", line=dict(color=ui.ORANGE, width=2))
    ui.fig_layout(fig, "Alert volume and report rate by quarter", 340,
                  yaxis=dict(title="Alerts"), yaxis2=dict(title="Report rate", overlaying="y", side="right",
                                                         tickformat=".0%", showgrid=False, rangemode="tozero"))
    ui.show(fig)
with right:
    st.markdown("##### Highest-risk alerts right now")
    top = test.sort_values("risk", ascending=False).head(8)[["alert_id", "quarter", "risk", "band", "low_confidence"]]
    st.dataframe(top, hide_index=True, width="stretch", height=318, column_config={
        "alert_id": "Alert", "quarter": "Quarter", "band": "Priority",
        "risk": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.2f"),
        "low_confidence": st.column_config.CheckboxColumn("Low confidence")})

ui.note(f"Model in service: <b>{ev['champion']}</b> ({ev.get('champion_reason', 'selected on validation quarters')}), evaluated once on "
        f"{len(test):,} later alerts ({ev['n_test_reported']:,} reported). "
        f"Up to <b>{ui.pct(champ['safe_deprioritised'])}</b> of the queue can be deprioritised while still retaining "
        f"{ui.pct(champ['safe_recall'])} of reported alerts.")

if ui.has("clustering.json") and ui.has("sequences.json"):
    cl, sq, en = ui.clustering(), ui.sequences(), ui.enrichment()
    st.markdown("##### Patterns behind the queue")
    a, b, c3 = st.columns(3)
    rates = [p["report_rate"] for p in cl["profiles"]["kmeans"]["test"]]
    ui.kpi(a, "Behavioural profiles", f"{cl['kmeans']['K']}", f"report rate {min(rates):.0%} to {max(rates):.0%} across profiles",
           ui.CYAN, count=cl["kmeans"]["K"])
    ui.kpi(b, "Transaction regimes", f"{sq['K']}", f"clients stay in a regime {np.mean(np.diag(np.array(sq['A']))):.1%} of steps",
           ui.PINK, count=sq["K"])
    if en:
        gains = [r for r in en["rows"] if r["variant"] != "Baseline" and r["verdict"] == "adds signal"]
        ui.kpi(c3, "Pattern features that improve ranking", f"{len(gains)} of 18", "of 18 model × feature-set tests (Feature Lab)",
               ui.LIME, count=len(gains))

