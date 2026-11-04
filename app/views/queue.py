"""Alert queue: the analyst's working list, ordered by risk."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui

ev = ui.evaluation()
Q = ui.table("scored_alerts.parquet")
models = [r["model"] for r in ev["table"]]

ui.setup("Alert Queue", "Operations", "Work the queue from the top. Adjust the review budget to see the trade-off.")

# attach behavioural profile and dominant regime when the pattern stages have run
prof_names, reg_names = {}, {}
if ui.has("clusters.parquet") and ui.has("sequence_features.parquet"):
    CL = ui.table("clusters.parquet")[["alert_id", "kmeans"]]
    SF = ui.table("sequence_features.parquet")
    sqj = ui.sequences()
    occ = [f"hmm_occ_{j}" for j in range(sqj["K"])]
    SF = SF.assign(regime=SF[occ].values.argmax(1))[["alert_id", "regime"]]
    Q = Q.merge(CL, on="alert_id", how="left").merge(SF, on="alert_id", how="left")
    Q["profile"] = Q.kmeans.map(lambda k: f"P{int(k) + 1}" if k == k else "")
    Q["regime_name"] = Q.regime.map(lambda k: f"R{int(k) + 1} · {sqj['states'][int(k)]['name']}" if k == k else "")

f = st.columns([2, 2, 2, 3])
quarters = f[0].multiselect("Quarter", sorted(Q[Q.role == "test"].quarter.unique()), default=sorted(Q[Q.role == "test"].quarter.unique()))
model = f[1].selectbox("Scoring model", models, index=models.index(ev["champion"]))
only_lc = f[2].selectbox("Confidence", ["All alerts", "Low confidence only"])
if "profile" in Q:
    profs = sorted(p for p in Q.profile.unique() if p)
    pick_prof = st.multiselect("Behavioural profile", profs, default=profs)
    Q = Q[Q.profile.isin(pick_prof) | (Q.profile == "")]
budget = f[3].slider("Review budget (share of queue)", 0.05, 1.0, float(ev["budget"]), 0.05, format="%.2f")

D = Q[Q.quarter.isin(quarters)].copy()
D["score"] = D[f"risk|{model}"]
D = D.sort_values("score", ascending=False).reset_index(drop=True)
D["rank"] = np.arange(1, len(D) + 1)
n_review = int(np.ceil(budget * len(D)))
D["action"] = np.where(D["rank"] <= n_review, "Review now", "Queue")
if only_lc != "All alerts":
    D = D[D.low_confidence]

full = Q[Q.quarter.isin(quarters)].assign(score=lambda d: d[f"risk|{model}"]).sort_values("score", ascending=False)
caught = full.y.head(n_review).sum()
total = max(int(full.y.sum()), 1)
c = st.columns(4)
ui.kpi(c[0], "Alerts to review", f"{n_review:,}", f"of {len(full):,} in the selected quarters", ui.ORANGE, count=n_review)
ui.kpi(c[1], "Reported alerts caught", ui.pct(caught / total), f"{int(caught):,} of {total:,} (back-test)", ui.GREEN,
       count=round(100 * caught / total), suffix="%", ring=caught / total)
ui.kpi(c[2], "Hit rate in reviewed set", ui.pct(caught / max(n_review, 1)),
       f"queue average {ui.pct(full.y.mean())}", ui.BLUE, count=round(100 * caught / max(n_review, 1)), suffix="%",
       ring=caught / max(n_review, 1))
ui.kpi(c[3], "Analyst effort saved", ui.pct(1 - budget), "alerts not opened at this budget", ui.VIOLET,
       count=round(100 * (1 - budget)), suffix="%", ring=1 - budget)
st.write("")
frac = np.arange(1, len(full) + 1) / len(full)
gain = np.cumsum(full.y.values) / total
fig = go.Figure()
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, dash="dot"), name="Random order")
fig.add_scatter(x=frac, y=gain, mode="lines", line=dict(color=ui.GREEN, width=2.5), name=model, fill="tonexty",
                fillcolor=ui.rgba(ui.GREEN, .08))
fig.add_scatter(x=[budget], y=[caught / total], mode="markers", marker=dict(size=14, color=ui.ORANGE, line=dict(color="#fff", width=2)),
                name="Your budget", hovertemplate="review %{x:.0%} → catch %{y:.0%}<extra></extra>")
ui.fig_layout(fig, "Where your review budget sits on the workload curve", 260, xaxis=dict(tickformat=".0%", title="Share reviewed"),
              yaxis=dict(tickformat=".0%", title="Reports caught"), margin=dict(l=50, r=20, t=44, b=40))
ui.show(fig)
st.write("")
ui.takeaway([f"At this budget an analyst opens <b>{n_review:,}</b> alerts and finds <b>{ui.pct(caught / total)}</b> of all reported cases; "
             f"about <b>{ui.pct(caught / max(n_review, 1))}</b> of the alerts they open turn out to be reported, against "
             f"{ui.pct(full.y.mean())} for the queue as a whole.",
             "Slide the budget up to catch more cases at the cost of more reviews - the orange dot shows where you are on that trade-off."],
            "What this budget means")

reveal = st.toggle("Reveal eventual outcome (back-test only)", value=False)
cols = ["rank", "alert_id", "quarter", "score", "action", "low_confidence"] + (["profile", "regime_name"] if "profile" in D else []) \
    + (["outcome"] if reveal else [])
D["outcome"] = np.where(D.y == 1, "Reported", "Dismissed")
st.dataframe(D[cols].head(500), hide_index=True, width="stretch", height=520, column_config={
    "rank": "Rank", "alert_id": "Alert", "quarter": "Quarter", "action": "Action", "outcome": "Outcome",
    "profile": "Profile", "regime_name": "Dominant regime",
    "score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f"),
    "low_confidence": st.column_config.CheckboxColumn("Low confidence")})
st.caption("Showing the top 500 rows. Open any alert ID on the Alert Detail page to see the evidence behind its score.")
