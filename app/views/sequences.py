"""Sequence analysis: hidden behavioural regimes in each alert's transaction history."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from adaptiveaml.s03_eda import wilson

sq = ui.sequences()
if not sq:
    ui.setup("Sequence Analysis", "Patterns")
    st.info("Run stage 9 first:  python run_pipeline.py --stage 9")
    st.stop()
K, states = sq["K"], sq["states"]
A = np.array(sq["A"])
SF = ui.table("sequence_features.parquet")
VP = ui.table("viterbi_paths.parquet")
names = [ui.state_label(k, states) for k in range(K)]
short = [f"R{k + 1}" for k in range(K)]
col = [ui.SERIES[k % 12] for k in range(K)]
best = max(sq["standalone"], key=lambda r: r["pr_auc"])
CH, SZ, GP = ["cash", "card", "wire", "international"], ["small", "medium", "large"], ["within the hour", "on the same day", "a day or more apart"]

# ---- regime risk: report rate of alerts whose history is MOSTLY in each regime (2020 and 2021)
occ_cols = [f"hmm_occ_{k}" for k in range(K)]
SF = SF.assign(dominant=SF[occ_cols].values.argmax(1))
risk = []
for k in range(K):
    for period, roles in (("2020", ["train"]), ("2021", ["valid", "test"])):
        d = SF[SF.role.isin(roles) & (SF.dominant == k)]
        lo, hi = wilson(int(d.y.sum()), len(d))
        risk.append({"k": k, "period": period, "rate": float(d.y.mean()) if len(d) else np.nan, "lo": lo, "hi": hi, "n": len(d)})
risk = pd.DataFrame(risk)
r21 = risk[risk.period == "2021"].set_index("k")
r20 = risk[risk.period == "2020"].set_index("k")
overall21 = float(SF[SF.role != "train"].y.mean())
hi_k, lo_k = int(r21.rate.idxmax()), int(r21.rate.idxmin())
consistent = (r20.rate.idxmax() == hi_k) and (r20.rate.idxmin() == lo_k)


def describe(k: int) -> str:
    """One sentence a non-specialist can read: what a client in this regime typically does."""
    m = states[k]["marginals"]
    c, s, g = int(np.argmax(m["channel"])), int(np.argmax(m["size"])), int(np.argmax(m["gap"]))
    return (f"Mostly <b>{CH[c]}</b> transactions ({m['channel'][c]:.0%}), usually <b>{SZ[s]}</b> amounts ({m['size'][s]:.0%}), "
            f"typically <b>{GP[g]}</b> ({m['gap'][g]:.0%}). {m['direction'][0]:.0%} of them are money coming in (credits).")


ui.setup("Sequence Analysis", "Patterns",
         "Each alert's year of transactions, read in order. The model splits that year into a few long behavioural phases "
         "(\"regimes\") and checks whether some phases are riskier than others.")

c = st.columns(4)
ui.kpi(c[0], "Transactions read in order", f"{sq['n_transactions'] / 1e6:.1f}M", f"{sq['n_sequences']:,} alert histories", ui.BLUE)
ui.kpi(c[1], "Behavioural regimes", f"{K}", "recurring phases found in the data", ui.VIOLET, count=K)
ui.kpi(c[2], "Riskiest regime", short[hi_k], f"{r21.rate[hi_k]:.0%} reported vs {overall21:.0%} overall (2021)", ui.ORANGE, ring=float(r21.rate[hi_k]))
ui.kpi(c[3], "Sequence score on its own", f"{best['pr_auc']:.3f}", f"ranking quality; random guessing = {sq['base_rate_test']:.3f}", ui.GREEN)
st.write("")

ui.takeaway([
    f"Every client's year breaks into a few <b>long phases</b>: once in a regime, a client usually stays for "
    f"{min(s['expected_duration'] for s in states):.0f}-{max(s['expected_duration'] for s in states):.0f} transactions before switching.",
    f"Alerts that spend most of their year in <b>{names[hi_k]}</b> were reported <b>{r21.rate[hi_k]:.0%}</b> of the time in 2021 "
    f"({r20.rate[hi_k]:.0%} in 2020), against <b>{r21.rate[lo_k]:.0%}</b> for <b>{names[lo_k]}</b>"
    + (" - the same ordering in both years, so it is a stable pattern." if consistent else "."),
    f"Read on its own, a client's sequence ranks alerts better than chance ({best['pr_auc']:.3f} vs {sq['base_rate_test']:.3f}), "
    "but the main risk model already captures the same information (see Feature Lab)."])
ui.plain(
    "Think of a client's year like the weather. You never see \"the season\" directly, you only see each day's weather. "
    "A hidden Markov model works the same way: it only sees transactions (cash or card, big or small, rushed or spread out) and infers "
    "the hidden **regime** - the kind of phase the client is in - that most likely produced them.\n\n"
    "- **Regime**: a typical pattern of transactions, like *many small card payments on the same day*.\n"
    "- **Transition**: the chance of moving from one regime to another at the next transaction.\n"
    "- **Viterbi path**: the single most likely sequence of regimes for one client, shown on *One client's year*.\n"
    "- **Suspicion score**: how much better a sequence fits the pattern of *reported* alerts than of *dismissed* ones.")

tabs = st.tabs(["Regimes explained", "Which regimes are riskier", "How clients move", "One client's year", "Suspicion score",
                "Model fit (technical)"])

# ---------------------------------------------------------------- regimes explained
with tabs[0]:
    pick = st.radio("Choose a regime", short, horizontal=True, format_func=lambda s: f"{s} · {states[int(s[1:]) - 1]['name']}")
    k = int(pick[1:]) - 1
    s = states[k]
    a, b = st.columns([2, 3])
    with a:
        st.markdown(
            f"<div class='card' style='border-top:3px solid {col[k]}'>"
            f"<div style='font-size:1.15rem;font-weight:700;margin-bottom:6px'>{names[k]}</div>"
            f"<div style='color:#DCE0E8;font-size:.95rem;line-height:1.55'>{describe(k)}</div>"
            f"<div style='display:flex;gap:10px;margin-top:14px'>"
            f"<div style='flex:1;background:#151A24;border-radius:10px;padding:10px'><div style='color:{ui.MUTED};font-size:.75rem'>Typical stay</div>"
            f"<div style='font-size:1.3rem;font-weight:700'>{s['expected_duration']:.0f}</div><div style='color:{ui.MUTED};font-size:.72rem'>transactions in a row</div></div>"
            f"<div style='flex:1;background:#151A24;border-radius:10px;padding:10px'><div style='color:{ui.MUTED};font-size:.75rem'>Reported when dominant</div>"
            f"<div style='font-size:1.3rem;font-weight:700;color:{ui.ORANGE if r21.rate[k] > overall21 else ui.BLUE}'>{r21.rate[k]:.0%}</div>"
            f"<div style='color:{ui.MUTED};font-size:.72rem'>vs {overall21:.0%} overall (2021, {int(r21.n[k]):,} alerts)</div></div></div>"
            f"<div style='color:{ui.MUTED};font-size:.8rem;margin-top:12px'>Most typical single transaction: {s['top'][0]['symbol']}</div></div>",
            unsafe_allow_html=True)
    with b:
        m, lift = s["marginals"], s["lifts"]
        cats = [("Channel", ["Cash", "Card", "Wire", "International"], "channel"), ("Amount", ["Small", "Medium", "Large"], "size"),
                ("Timing", ["< 1 hour apart", "Same day", "≥ 1 day apart"], "gap")]
        fig = go.Figure()
        xs, this, typical = [], [], []
        for grp, labels, key in cats:
            for i, lab in enumerate(labels):
                xs.append(f"{grp}: {lab}")
                this.append(m[key][i])
                typical.append(m[key][i] / max(lift[key][i], 1e-9))
        fig.add_bar(y=xs, x=typical, orientation="h", name="All transactions", marker_color=ui.rgba(ui.MUTED, .45))
        fig.add_bar(y=xs, x=this, orientation="h", name=pick, marker_color=col[k])
        ui.fig_layout(fig, f"{pick} compared with all transactions", 380, barmode="group",
                      xaxis=dict(tickformat=".0%", title="Share of transactions"),
                      yaxis=dict(autorange="reversed"), margin=dict(l=160, r=20, t=48, b=44))
        ui.show(fig)
    st.markdown("##### All regimes at a glance")
    st.dataframe(pd.DataFrame([{"Regime": names[j], "Typical stay (transactions)": round(states[j]["expected_duration"]),
                                "Reported when dominant, 2020": r20.rate[j], "Reported when dominant, 2021": r21.rate[j],
                                "Alerts (2021)": int(r21.n[j])} for j in range(K)]),
                 hide_index=True, width="stretch", column_config={
                     "Reported when dominant, 2020": st.column_config.NumberColumn(format="percent"),
                     "Reported when dominant, 2021": st.column_config.NumberColumn(format="percent")})

# ---------------------------------------------------------------- risk by regime
with tabs[1]:
    ui.takeaway([f"<b>{names[hi_k]}</b> is the riskiest phase: {r21.rate[hi_k]:.0%} of alerts dominated by it were reported in 2021.",
                 f"<b>{names[lo_k]}</b> is the calmest: {r21.rate[lo_k]:.0%}.",
                 "Bars whose interval does not cross the dashed overall line are reliably different from average."],
                "What the chart shows")
    fig = go.Figure()
    for period, colr in (("2020", ui.rgba(ui.BLUE, .75)), ("2021", ui.ORANGE)):
        d = risk[risk.period == period]
        fig.add_bar(x=[f"{short[int(k)]} · {states[int(k)]['name']}" for k in d.k], y=d.rate, name=f"{period} alerts", marker_color=colr,
                    error_y=dict(type="data", symmetric=False, array=d.hi - d.rate, arrayminus=d.rate - d.lo, color=ui.TEXT, thickness=1.2),
                    customdata=d.n, hovertemplate="%{x}<br>%{y:.1%} reported<br>%{customdata:,} alerts<extra></extra>")
    fig.add_hline(y=overall21, line=dict(color=ui.RED, dash="dash", width=1.5))
    ui.fig_layout(fig, "Share reported, by the regime that dominates an alert's year (95% intervals)", 460, barmode="group",
                  yaxis=dict(tickformat=".0%", rangemode="tozero", title="Reported"), margin=dict(l=50, r=20, t=70, b=90))
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
    ui.show(fig)
    ui.note("\"Dominates\" means the regime the client spent most of the year in. The 2020 bars come from the data the model was "
            "trained on; the 2021 bars are new alerts, so a pattern visible in both is not a fluke of one year.", ui.ORANGE)

# ---------------------------------------------------------------- transitions
with tabs[2]:
    stay_avg = float(np.mean(np.diag(A)))
    off = A - np.diag(np.diag(A))
    ui.takeaway([f"Clients almost never jump around: on average they stay in the same regime <b>{stay_avg:.1%}</b> of the time from "
                 "one transaction to the next.",
                 "When they do move, they tend to follow a few preferred routes (the thicker arrows) rather than switching at random."])
    if True:
        ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, K, endpoint=False)
        xy = np.c_[np.cos(ang), np.sin(ang)] * 1.0
        scale = off.max() or 1
        fig = go.Figure()
        for j in range(K):
            for kk in range(K):
                if j != kk and off[j, kk] > 0.05 * scale:
                    fig.add_annotation(x=xy[kk, 0] * .78 + xy[j, 0] * .22, y=xy[kk, 1] * .78 + xy[j, 1] * .22,
                                       ax=xy[j, 0] * .78 + xy[kk, 0] * .22, ay=xy[j, 1] * .78 + xy[kk, 1] * .22,
                                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1,
                                       arrowwidth=1.5 + 6 * off[j, kk] / scale, arrowcolor=ui.rgba(col[j], .8))
        fig.add_scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=short, textfont=dict(color="#fff", size=14),
                        marker=dict(size=46, color=col, line=dict(color="#fff", width=1.5)),
                        hovertext=[f"{names[k]}<br>stays {A[k, k]:.1%}" for k in range(K)], hoverinfo="text", showlegend=False)
        # demo path: an alert whose year passes through several regimes (a path that stays put shows nothing)
        n_sw = VP.states.map(lambda v: int((np.diff(np.asarray(v)) != 0).sum()))
        demo_ids = VP.alert_id[(n_sw >= 3) & (n_sw <= 12)].tolist() or VP.alert_id.tolist()
        demo = st.selectbox("Path to play", demo_ids[:200], format_func=lambda a_: f"Alert #{a_}", key="demo_path")
        pick_row = VP.set_index("alert_id").loc[demo]
        pick_row = pick_row.copy()
        pick_row["alert_id"] = demo
        runs = [int(pick_row.states[0])] + [int(s_) for s_, p_ in zip(pick_row.states[1:], pick_row.states[:-1]) if s_ != p_]
        runs = (runs if len(runs) > 1 else runs * 2)[:40]
        fig.add_scatter(x=[xy[runs[0], 0]], y=[xy[runs[0], 1]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=64, color="rgba(0,0,0,0)", line=dict(width=4, color="#FFFFFF")))
        fig.frames = [go.Frame(data=[go.Scatter(x=[xy[r, 0]], y=[xy[r, 1]])], traces=[1], name=str(i)) for i, r in enumerate(runs)]
        ui.fig_layout(fig, "Routes between regimes (thicker arrow = more common)", 460, margin=dict(l=10, r=10, t=48, b=10),
                      xaxis=dict(visible=False, range=[-2.4, 2.4]), yaxis=dict(visible=False, range=[-1.3, 1.3]),
                      updatemenus=[dict(type="buttons", x=0.0, y=0.0, xanchor="left", yanchor="bottom", bgcolor="#2A3350",
                                        bordercolor=ui.BLUE, font=dict(color="#FFFFFF", size=12), showactive=False,
                                        buttons=[dict(label="▶ Play a real client's path", method="animate",
                                                      args=[None, dict(frame=dict(duration=800, redraw=True), transition=dict(duration=300),
                                                                       fromcurrent=True, mode="immediate")])])])
        ui.show(fig)
        st.caption(f"Press Play: the white ring follows alert #{int(demo)} through its year: "
                   + " → ".join(short[r] for r in runs[:12]) + (" …" if len(runs) > 12 else ""))
    a, b = st.columns([2, 3])
    with a:
        frm = st.selectbox("If a client is in…", short, format_func=lambda s: names[int(s[1:]) - 1])
        j = int(frm[1:]) - 1
        leave = off[j] / max(off[j].sum(), 1e-12)
        st.markdown(f"<div class='card'>They stay put <b>{A[j, j]:.1%}</b> of the time - on average about "
                    f"<b>{1 / max(1 - A[j, j], 1e-9):.0f}</b> transactions before moving on.</div>", unsafe_allow_html=True)
    with b:
        order = [i for i in np.argsort(-leave) if i != j]
        fig = go.Figure(go.Bar(x=[leave[i] for i in order], y=[names[i] for i in order], orientation="h",
                               marker_color=[col[i] for i in order], text=[f"{leave[i]:.0%}" for i in order], textposition="outside"))
        ui.fig_layout(fig, "…and when they leave, they go to", 300, xaxis=dict(tickformat=".0%", range=[0, 1.15]),
                      yaxis=dict(autorange="reversed"), margin=dict(l=200, r=20, t=48, b=30))
        ui.show(fig)
    with st.expander("Full transition matrix (technical)"):
        hm = go.Figure(go.Heatmap(z=A, x=short, y=short, zmin=0, zmax=1, colorscale=[[0, "#10131A"], [0.02, "#2A3350"], [0.5, ui.VIOLET], [1, ui.ORANGE]],
                                  text=np.round(A, 3), texttemplate="%{text}", textfont=dict(color="#FFFFFF"), colorbar=dict(thickness=10)))
        ui.fig_layout(hm, "p(next regime | current regime)", 420, xaxis=dict(title="next"), yaxis=dict(title="current", autorange="reversed"))
        ui.show(hm)

# ---------------------------------------------------------------- one client's year
with tabs[3]:
    Q = ui.table("scored_alerts.parquet").set_index("alert_id")
    ids = [a_ for a_ in Q[Q.role == "test"].sort_values("risk", ascending=False).index if a_ in set(VP.alert_id)]
    default = st.session_state.get("alert_pick")
    sel = st.selectbox("Alert (highest risk first)", ids, index=ids.index(default) if default in ids else 0, format_func=lambda a_: f"#{a_}")
    row = VP.set_index("alert_id").loc[sel]
    stv, days = np.asarray(row.states), np.asarray(row.days)
    f_ = SF.set_index("alert_id").loc[sel]
    # narrate the path: merge runs into phases, ignoring flickers shorter than 2% of the history
    seg, start = [], 0
    for i in range(1, len(stv) + 1):
        if i == len(stv) or stv[i] != stv[start]:
            seg.append([int(stv[start]), start, i - 1])
            start = i
    minlen = max(3, int(0.02 * len(stv)))
    phases = []
    for s_ in seg:
        if phases and (s_[0] == phases[-1][0] or s_[2] - s_[1] + 1 < minlen):
            phases[-1][2] = s_[2]
        else:
            phases.append(list(s_))
    lines = [f"<b>{days[p[1]]:.0f} to {days[p[2]]:.0f} days before the alert</b>: {names[p[0]]}" for p in phases[:6]]
    pct = float((SF.hmm_llr < f_.hmm_llr).mean())
    ui.takeaway(lines + [f"Its sequence fits the reported pattern better than <b>{pct:.0%}</b> of alerts; "
                         f"the last 30 days look {'very different from' if f_.hmm_regime_shift > .5 else 'similar to'} the rest of the year "
                         f"({f_.hmm_regime_shift:.0%} change)."], f"Alert #{sel}: the story of its year")
    fig = go.Figure()
    for k in range(K):
        m_ = stv == k
        if m_.any():
            fig.add_scattergl(x=-days[m_], y=[names[k]] * int(m_.sum()), mode="markers", name=names[k],
                              marker=dict(color=col[k], size=8, symbol="line-ns-open", line=dict(width=2, color=col[k])),
                              hovertemplate="%{x:.0f} days before alert<extra></extra>")
    fig.add_vrect(x0=-30, x1=0, fillcolor=ui.rgba(ui.ORANGE, .08), line_width=0)
    ui.fig_layout(fig, "Most likely regime at each transaction (shaded: last 30 days)", 340,
                  xaxis=dict(title="Days before alert", range=[-366, 2]), yaxis=dict(categoryorder="array", categoryarray=names[::-1]),
                  margin=dict(l=230, r=20, t=48, b=44), showlegend=False)
    ui.show(fig)

# ---------------------------------------------------------------- suspicion score
with tabs[4]:
    ui.takeaway(["The score asks: <i>does this year of transactions look more like the reported alerts we saw in 2020, or the dismissed ones?</i>",
                 f"Reported alerts (orange) sit slightly to the right, but the two shapes overlap heavily: on its own the score is "
                 f"<b>{best['pr_auc']:.3f}</b> against <b>{sq['base_rate_test']:.3f}</b> for random guessing - a weak but real signal."])
    hst = sq["llr_hist"]
    mid = (np.array(hst["bins"][:-1]) + np.array(hst["bins"][1:])) / 2
    fig = go.Figure()
    fig.add_scatter(x=mid, y=hst["dismiss"], name="Dismissed", mode="lines", line=dict(color=ui.DISMISS, width=2.5), fill="tozeroy",
                    fillcolor=ui.rgba(ui.DISMISS, .2))
    fig.add_scatter(x=mid, y=hst["report"], name="Reported", mode="lines", line=dict(color=ui.REPORT, width=2.5), fill="tozeroy",
                    fillcolor=ui.rgba(ui.REPORT, .2))
    fig.add_vline(x=0, line=dict(color=ui.TEXT, dash="dot", width=1))
    fig.add_annotation(x=0, y=1.02, yref="paper", text="← looks dismissed   ·   looks reported →", showarrow=False, font=dict(color=ui.MUTED))
    ui.fig_layout(fig, "Suspicion score on new (2021 Q3-Q4) alerts", 380, xaxis=dict(title="Suspicion score"), yaxis=dict(title="How common"))
    ui.show(fig)
    with st.expander("Every sequence score compared (technical)"):
        st.dataframe(pd.DataFrame([{"Score": r["score"], "PR-AUC": r["pr_auc"], "ROC-AUC": r["roc_auc"]} for r in sq["standalone"]]),
                     hide_index=True, width="stretch", column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                                                     "ROC-AUC": st.column_config.NumberColumn(format="%.3f")})
        st.caption("The score is log p(sequence | reported-type model) − log p(sequence | dismissed-type model), per transaction. "
                   "Training alerts are scored by models fitted on the other half of 2020, so the score never sees its own label.")

# ---------------------------------------------------------------- model fit
with tabs[5]:
    ui.takeaway([f"{K} regimes were chosen because adding a sixth no longer improved the fit on new alerts.",
                 "Training improves the fit at every step and then levels off, which is what the algorithm guarantees."])
    a, b = st.columns(2)
    ks = pd.DataFrame(sq["k_selection"])
    fig = go.Figure()
    fig.add_scatter(x=ks.K, y=ks.val_ll_per_txn, mode="lines+markers", name="New alerts (early 2021)", line=dict(color=ui.ORANGE, width=2.5))
    fig.add_scatter(x=ks.K, y=ks.train_ll_per_txn, mode="lines+markers", name="Training alerts (2020)", line=dict(color=ui.BLUE, width=2.5))
    fig.add_vline(x=K, line=dict(color=ui.GREEN, dash="dash"))
    ui.fig_layout(fig, "How well each number of regimes fits (higher is better)", 360, xaxis=dict(title="Number of regimes", dtick=1))
    with a: ui.show(fig)
    fig = go.Figure()
    fig.add_scatter(y=sq["history"], mode="lines+markers", name="Behaviour model", line=dict(color=ui.VIOLET, width=2.5))
    for k_, c_ in (("report", ui.REPORT), ("dismiss", ui.DISMISS)):
        fig.add_scatter(y=sq["class_history"][k_], mode="lines", name=f"{k_.title()}-type model", line=dict(color=c_, width=2))
    ui.fig_layout(fig, "Training progress (Baum-Welch / EM iterations)", 360, xaxis=dict(title="Iteration"))
    with b: ui.show(fig)
    with st.expander("Model-selection table, numerical safeguards and related models"):
        st.dataframe(ks.rename(columns={"K": "Regimes", "val_ll_per_txn": "Fit on new alerts", "bic": "BIC (lower is better)",
                                        "train_ll_per_txn": "Fit on training alerts"}), hide_index=True, width="stretch")
        st.markdown("- **Scaling factors.** A year of transactions multiplies hundreds of probabilities, which would round to zero; "
                    "the forward pass rescales at every step and adds the logs of the scaling factors.\n"
                    "- **Viterbi** works with log-probabilities, so it needs no scaling.\n"
                    "- **Linear dynamical systems** replace the discrete regime with a continuous hidden state (Kalman filter); not used "
                    "here because the transaction types are categories, not measurements.")
