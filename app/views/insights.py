"""Data insights: what the data is, and what it says before any model is trained."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

dq, e = ui.quality(), ui.eda()
ui.setup("Data Insights", "Data", "Alerts raised by a bank's monitoring system, each linked to up to a year of the client's transactions. "
         "All analysis below uses the 2020 training quarters only.")

c = st.columns(4)
ui.kpi(c[0], "Transactions", f"{dq['n_transactions'] / 1e6:.1f}M", f"about {dq['txn_per_alert']['mean']:,.0f} per alert", ui.BLUE)
ui.kpi(c[1], "Investigated alerts", f"{dq['n_alerts']:,}", "one row per alert, many transactions each", ui.CYAN)
ui.kpi(c[2], "Reported to authorities", ui.pct(dq["report_rate"], 1), "the class the model must find", ui.ORANGE)
ui.kpi(c[3], "Quarters", f"{len(dq['by_quarter'])}", f"{dq['by_quarter'][0]['quarter']} to {dq['by_quarter'][-1]['quarter']}", ui.VIOLET)
st.write("")

tabs = st.tabs(["Structure and quality", "Class balance", "What separates the classes", "Evidence and belief", "Feature relationships", "Drift"])

with tabs[0]:
    ui.takeaway([f"Two files joined by alert ID: <b>{dq['n_alerts']:,} alerts</b>, each linked to about {dq['txn_per_alert']['mean']:,.0f} "
                 "transactions from the year before it was raised.",
                 "Every quality check passes: nothing missing, no duplicates, no orphan records.",
                 "Alerts are split <b>by time</b>: the models learn on 2020 and are judged on later 2021 quarters they have never seen."])
    a, b = st.columns(2)
    a.markdown("##### Two linked files")
    a.dataframe(pd.DataFrame([
        {"File": "Alerts", "Rows": f"{dq['n_alerts']:,}", "Fields": "Alert ID, date (quarter precision), outcome"},
        {"File": "Transactions", "Rows": f"{dq['n_transactions']:,}", "Fields": "Alert ID, timestamp, credit/debit, channel, relative size"}]),
        hide_index=True, width="stretch")
    a.markdown("##### Split by time")
    a.dataframe(pd.DataFrame([{"Period": "2020 Q1-Q4", "Role": "Training", "Alerts": dq["split"]["train"]},
                              {"Period": "2021 Q1-Q2", "Role": "Tuning, calibration, threshold", "Alerts": dq["split"]["valid"]},
                              {"Period": "2021 Q3-Q4", "Role": "Final evaluation", "Alerts": dq["split"]["test"]}]),
                hide_index=True, width="stretch")
    b.markdown("##### Quality checks")
    checks = [("Missing values in alerts", sum(dq["nulls_alerts"].values())), ("Missing values in transactions", sum(dq["nulls_transactions"].values())),
              ("Duplicate alert IDs", dq["duplicate_alert_ids"]), ("Alerts with no transactions", dq["alerts_without_transactions"]),
              ("Transactions with no matching alert", dq["transactions_without_alert"]), ("Alerts outside the eight quarters", dq["split"]["unused"])]
    b.dataframe(pd.DataFrame([{"Check": k, "Count": v, "Status": "Pass" if v == 0 else "Review"} for k, v in checks]),
                hide_index=True, width="stretch")
    tv = dq["type_values"]
    fig = go.Figure(go.Bar(x=[t["type"].title() for t in tv], y=[t["len"] for t in tv],
                           marker_color=[ui.BLUE, ui.CYAN, ui.VIOLET, ui.GREEN][:len(tv)]))
    ui.fig_layout(fig, "Transactions by channel", 300, yaxis=dict(title="Transactions"))
    with b: ui.show(fig)
    ui.note(f"Longest history seen: {dq['history_days_max']:.0f} days. Transaction size has mean {dq['size_mean']:.2f} and standard "
            f"deviation {dq['size_std']:.2f}: it is a standardised, relative amount, so it is never summed into money totals.")

with tabs[1]:
    ui.takeaway([f"Only about <b>{e['prior']:.0%}</b> of alerts end up reported - roughly 1 in {1 / e['prior']:.0f}.",
                 f"So a lazy model that dismisses everything is {ui.pct(e['accuracy_trap']['accuracy'])} \"accurate\" and useless. "
                 "That is why every score in this app is about <b>finding the reported ones</b>, not accuracy.",
                 "The share reported is steady from quarter to quarter, so no quarter is unusual."])
    a, b = st.columns(2)
    bal = e["balance"]
    fig = go.Figure(go.Bar(x=[x["quarter"] for x in bal], y=[x["rate"] for x in bal], marker_color=ui.ORANGE,
                           error_y=dict(type="data", symmetric=False, color=ui.TEXT, thickness=1.2,
                                        array=[x["hi"] - x["rate"] for x in bal], arrayminus=[x["rate"] - x["lo"] for x in bal])))
    ui.fig_layout(fig, "Report rate per training quarter (95% Wilson interval)", 380, yaxis=dict(title="Share reported", tickformat=".0%", rangemode="tozero"))
    with a: ui.show(fig)
    t = e["accuracy_trap"]
    fig = go.Figure(go.Bar(x=[t["accuracy"], t["recall"], t["pr_auc"], t["roc_auc"]], y=["Accuracy", "Recall", "PR-AUC", "ROC-AUC"],
                           orientation="h", marker_color=[ui.RED, ui.MUTED, ui.MUTED, ui.MUTED],
                           text=[f"{v:.3f}" for v in (t["accuracy"], t["recall"], t["pr_auc"], t["roc_auc"])], textposition="outside"))
    ui.fig_layout(fig, "A model that dismisses every alert", 380, xaxis=dict(range=[0, 1.1]), margin=dict(l=90, r=20, t=48, b=44))
    with b: ui.show(fig)
    ui.note(f"Dismissing everything scores {ui.pct(t['accuracy'], 1)} accuracy while catching no reported alert. Ranking quality on the "
            f"reported class (PR-AUC) is therefore the headline metric, and {e['prior']:.3f} is the level any model has to beat.")

with tabs[2]:
    mw = pd.DataFrame(e["mannwhitney"])
    k = mw.head(15)[::-1]
    top = mw.iloc[0]
    lim = max(0.25, float(mw.effect.abs().max()) * 1.4)
    ui.takeaway([f"The biggest difference is in <b>{top.label.lower()}</b>: "
                 f"{'higher' if top.effect > 0 else 'lower'} for reported alerts (median {top.median_report:.2f} vs {top.median_dismiss:.2f}).",
                 f"But even that difference is <b>small</b>: {abs(top.effect):.2f} on a scale where 0 means no difference and 1 means the groups "
                 "never overlap. No single behaviour separates reported from dismissed alerts.",
                 "That is why the models combine many weak clues instead of relying on one rule."])
    fig = go.Figure(go.Bar(x=k.effect, y=k.label, orientation="h", marker_color=[ui.ORANGE if v > 0 else ui.BLUE for v in k.effect]))
    fig.add_vline(x=0, line=dict(color=ui.TEXT, width=1))
    ui.fig_layout(fig, "Which behaviours differ between reported and dismissed alerts", 500,
                  xaxis=dict(title="Size of difference (blue = higher in dismissed)", range=[-lim, lim]),
                  margin=dict(l=250, r=20, t=48, b=44))
    a, b = st.columns([3, 2])
    with a: ui.show(fig)
    with b:
        feat = st.selectbox("Distribution", list(e["distributions"]), format_func=ui.label)
        d = e["distributions"][feat]
        mid = (np.array(d["bins"][:-1]) + np.array(d["bins"][1:])) / 2
        fig = go.Figure()
        fig.add_scatter(x=mid, y=d["dismiss"], name="Dismissed", mode="lines", line=dict(color=ui.DISMISS, width=2), fill="tozeroy", fillcolor=ui.rgba(ui.DISMISS, 0.18))
        fig.add_scatter(x=mid, y=d["report"], name="Reported", mode="lines", line=dict(color=ui.REPORT, width=2), fill="tozeroy", fillcolor=ui.rgba(ui.REPORT, 0.18))
        ov = float(np.minimum(d["report"], d["dismiss"]).sum() / max(np.sum(d["dismiss"]), 1e-12))
        ui.fig_layout(fig, ui.label(feat), 380, yaxis=dict(title="How common"))
        ui.show(fig)
        st.markdown(f"<div class='card' style='font-size:.9rem'>The two curves <b>overlap about {min(ov, 1):.0%}</b>. "
                    f"{'Knowing this value alone tells you almost nothing about the outcome.' if ov > .85 else 'This value helps a little to tell the groups apart.'}"
                    "</div>", unsafe_allow_html=True)
    sig = int((mw.p_adj < 0.05).sum())
    ui.note(f"{sig} of {len(mw)} behaviours differ significantly after correcting for multiple testing (Mann-Whitney U, "
            "Benjamini-Hochberg). With thousands of alerts almost anything is significant, so behaviours are ranked by effect size, not p-value.")
    ch = e["channel"]
    st.markdown("##### Dominant channel and outcome")
    st.dataframe(pd.DataFrame([{"Dominant channel": r["channel"].title(), "Alerts": r["alerts"], "Report rate": r["rate"]} for r in ch["table"]]),
                 hide_index=True, width="stretch", column_config={"Report rate": st.column_config.NumberColumn(format="percent")})
    st.caption(f"Chi-square test of independence: χ² = {ch['chi2']:.1f}, df = {ch['dof']}, p = {ch['p']:.2g}. "
               f"Strength of association (Cramér's V) = {ch['cramers_v']:.3f}.")

with tabs[3]:
    _b = pd.DataFrame(e["bayes"]).iloc[0]
    ui.takeaway([f"Before looking at anything, an alert has a <b>{e['prior']:.0%}</b> chance of being reported.",
                 f"Learning that an alert is in the top 10% for <b>{_b.label.lower()}</b> moves that to <b>{_b.posterior:.0%}</b> - "
                 "the strongest single update, and still a modest one.",
                 "This is Bayes' rule: start from a prior belief and update it with evidence. The Naive Bayes model does exactly this, "
                 "for all behaviours at once."])
    by = pd.DataFrame(e["bayes"]).head(12)[::-1]
    fig = go.Figure()
    for _, r in by.iterrows():
        fig.add_scatter(x=[e["prior"], r.posterior], y=[r.label, r.label], mode="lines", line=dict(color=ui.LINE, width=3), showlegend=False)
    fig.add_scatter(x=[e["prior"]] * len(by), y=by.label, mode="markers", name="Before: base rate", marker=dict(color=ui.MUTED, size=10))
    fig.add_scatter(x=by.posterior, y=by.label, mode="markers", name="After: behaviour in its top 10%", marker=dict(color=ui.ORANGE, size=12))
    ui.fig_layout(fig, "How much one piece of evidence moves the probability of a report", 480,
                  xaxis=dict(title="Probability the alert is reported", tickformat=".0%"), margin=dict(l=250, r=20, t=48, b=44))
    a, b = st.columns([3, 2])
    with a: ui.show(fig)
    with b:
        g = e["gaussian"]
        feat = st.selectbox("Bell-curve fit", list(g), format_func=ui.label)
        with st.expander("Fitted parameters"):
          st.dataframe(pd.DataFrame([{"Class": k.title(), "Mean": v["mu"], "Std dev": v["sigma"], "Skew": v["skew"],
                                    "Excess kurtosis": v["kurtosis"]} for k, v in g[feat].items()]),
                     hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("Mean", "Std dev", "Skew", "Excess kurtosis")})
        _g = g[feat]
        st.markdown(f"<div class='card' style='font-size:.88rem'>Reported and dismissed alerts are centred at {_g['report']['mu']:.2f} and "
                    f"{_g['dismiss']['mu']:.2f}: the two bell curves almost sit on top of each other.</div>", unsafe_allow_html=True)
        xs = np.linspace(min(v["mu"] - 3.5 * v["sigma"] for v in g[feat].values()), max(v["mu"] + 3.5 * v["sigma"] for v in g[feat].values()), 200)
        fig = go.Figure()
        for k, col in (("dismiss", ui.DISMISS), ("report", ui.REPORT)):
            v = g[feat][k]
            fig.add_scatter(x=xs, y=np.exp(-0.5 * ((xs - v["mu"]) / v["sigma"]) ** 2) / (v["sigma"] * np.sqrt(2 * np.pi)),
                            name=k.title() + "ed" if k == "report" else "Dismissed", mode="lines", line=dict(color=col, width=2))
        ui.fig_layout(fig, "Fitted normal curve per class", 300)
        ui.show(fig)
    ui.note("Left: the starting belief is the base rate; observing one extreme behaviour updates it. Right: skew and kurtosis far "
            "from zero mean the bell-curve assumption is strained for that behaviour, which is where the simplest probabilistic model will struggle.")

with tabs[4]:
    _p = e["corr_top"][0]
    ui.takeaway([f"Many behaviours move together - for example <b>{ui.label(_p['a']).lower()}</b> and <b>{ui.label(_p['b']).lower()}</b> "
                 f"(correlation {_p['rho']:+.2f}). Such pairs carry the same information twice.",
                 "Models that assume every clue is independent (Naive Bayes) count those clues twice and become over-confident; "
                 "the regularised models are built to handle it.",
                 f"Even the best straight-line combination of all behaviours separates the two groups only weakly "
                 f"(separation score {e['fisher']['criterion']:.2f}; well-separated groups score above 1)."])
    a, b = st.columns([3, 2])
    cm = e["corr"]
    fig = go.Figure(go.Heatmap(z=cm["matrix"], x=[ui.label(f) for f in cm["features"]], y=[ui.label(f) for f in cm["features"]],
                               zmin=-1, zmax=1, colorscale=[[0, ui.BLUE], [0.5, "#14161B"], [1, ui.ORANGE]], colorbar=dict(thickness=10)))
    ui.fig_layout(fig, "Rank correlation between behaviours", 620, margin=dict(l=230, r=10, t=48, b=200))
    fig.update_xaxes(tickangle=60, tickfont=dict(size=9)); fig.update_yaxes(tickfont=dict(size=9))
    with a: ui.show(fig)
    with b:
        st.markdown("##### Most strongly related pairs")
        st.dataframe(pd.DataFrame([{"Behaviour A": ui.label(r["a"]), "Behaviour B": ui.label(r["b"]), "Correlation": r["rho"]} for r in e["corr_top"]]),
                     hide_index=True, width="stretch", column_config={"Correlation": st.column_config.NumberColumn(format="%+.2f")})
        fz = e["fisher"]
        mid = (np.array(fz["bins"][:-1]) + np.array(fz["bins"][1:])) / 2
        fig = go.Figure()
        fig.add_scatter(x=mid, y=fz["dismiss"], name="Dismissed", mode="lines", line=dict(color=ui.DISMISS, width=2), fill="tozeroy", fillcolor=ui.rgba(ui.DISMISS, 0.18))
        fig.add_scatter(x=mid, y=fz["report"], name="Reported", mode="lines", line=dict(color=ui.REPORT, width=2), fill="tozeroy", fillcolor=ui.rgba(ui.REPORT, 0.18))
        ui.fig_layout(fig, "Best single straight-line view of the two classes", 300, xaxis=dict(title="Discriminant score"))
        ui.show(fig)
        st.caption(f"Separation score {fz['criterion']:.2f} (distance between class means squared, over the sum of class variances).")
    ui.note("Strongly related behaviours carry overlapping information. That is why the linear models are regularised, and why a model "
            "that treats every behaviour as independent evidence tends to be over-confident.")

with tabs[5]:
    dr = pd.DataFrame(e["drift"])
    _n = int((dr.p_adj < 0.05).sum())
    ui.takeaway([f"<b>{_n} of {len(dr)}</b> behaviours look measurably different in early 2021 than in 2020 (red bars).",
                 f"The largest shift is in <b>{dr.iloc[0].label.lower()}</b>. Shifts like this are why the models are always "
                 "judged on later quarters, never on the year they learned from."])
    k = dr.head(15)[::-1]
    fig = go.Figure(go.Bar(x=k.stat, y=k.label, orientation="h", marker_color=[ui.RED if p < 0.05 else ui.rgba(ui.MUTED, 0.6) for p in k.p_adj]))
    ui.fig_layout(fig, "Shift in each behaviour between 2020 and early 2021", 500,
                  xaxis=dict(title="Kolmogorov-Smirnov distance (0 = identical distributions)"), margin=dict(l=250, r=20, t=48, b=44))
    ui.show(fig)
    ui.note(f"{int((dr.p_adj < 0.05).sum())} of {len(dr)} behaviours have shifted significantly (red). Shifts in the inputs are an early "
            "warning that a model trained on last year may age; the per-quarter results on the Model Comparison page show whether it actually does.")
