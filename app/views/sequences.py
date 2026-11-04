"""Sequence analysis: hidden behavioural regimes in each alert's transaction history."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

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
col = [ui.SERIES[k % 12] for k in range(K)]
best = max(sq["standalone"], key=lambda r: r["pr_auc"])

ui.setup("Sequence Analysis", "Patterns",
         f"Each alert's year of transactions read in order. A hidden Markov model finds {K} recurring behavioural regimes and how "
         "clients move between them. Learned on 2020 alerts only.")

c = st.columns(4)
ui.kpi(c[0], "Transactions modelled", f"{sq['n_transactions'] / 1e6:.1f}M", f"{sq['n_sequences']:,} alert sequences", ui.BLUE)
ui.kpi(c[1], "Behavioural regimes", f"{K}", "chosen by BIC and held-out fit", ui.VIOLET, count=K)
stay = float(np.mean(np.diag(A)))
ui.kpi(c[2], "Chance of staying in a regime", f"{stay:.1%}", "per transaction (mean of the diagonal)", ui.GREEN, ring=stay)
ui.kpi(c[3], "Best sequence score alone", f"{best['pr_auc']:.3f}", f"PR-AUC; no skill {sq['base_rate_test']:.3f}", ui.ORANGE)
st.write("")

tabs = st.tabs(["Regimes", "Transitions", "Alert timeline", "Suspicion scores", "Model fit"])

# ---------------------------------------------------------------- regimes
with tabs[0]:
    cols = st.columns(min(K, 3))
    for k, s in enumerate(states):
        m = s["marginals"]
        def bars(vals, labels, c_):
            return "".join(f"<div style='display:flex;align-items:center;gap:8px;font-size:.76rem;margin:2px 0'><span style='width:88px;color:#C9CFDB'>{l}</span>"
                           f"<div style='flex:1;background:{ui.LINE};border-radius:4px;height:7px'><div style='width:{v * 100:.0f}%;"
                           f"background:{c_};height:7px;border-radius:4px'></div></div><span style='width:34px;text-align:right'>{v:.0%}</span></div>"
                           for v, l in zip(vals, labels))
        delta = s["occupancy_report"] - s["occupancy_dismiss"]
        with cols[k % len(cols)]:
            st.markdown(f"<div class='card' style='border-top:3px solid {col[k]};margin-bottom:12px'>"
                        f"<b style='font-size:1.02rem'>{names[k]}</b>"
                        f"<div style='color:{ui.MUTED};font-size:.78rem;margin:4px 0 8px'>stays for about {s['expected_duration']:.0f} "
                        f"transactions · {s['occupancy_report']:.0%} of reported vs {s['occupancy_dismiss']:.0%} of dismissed histories "
                        f"({'+' if delta >= 0 else ''}{delta * 100:.1f} pts)</div>"
                        + bars(m["channel"], ["Cash", "Card", "Wire", "International"], col[k])
                        + bars(m["size"], ["Small", "Medium", "Large"], ui.rgba(col[k], .7))
                        + bars(m["gap"], ["< 1 hour", "Same day", "≥ 1 day"], ui.rgba(col[k], .5))
                        + f"<div style='font-size:.74rem;color:{ui.MUTED};margin-top:8px'>Most typical: {s['top'][0]['symbol']}</div></div>",
                        unsafe_allow_html=True)
    occ_r = [s["occupancy_report"] for s in states]
    occ_d = [s["occupancy_dismiss"] for s in states]
    fig = go.Figure()
    short = [f"R{k + 1}" for k in range(K)]
    fig.add_bar(x=short, y=occ_d, name="Dismissed alerts", marker_color=ui.DISMISS, hovertext=names)
    fig.add_bar(x=short, y=occ_r, name="Reported alerts", marker_color=ui.REPORT, hovertext=names)
    ui.fig_layout(fig, "Share of time spent in each regime (2020 alerts)", 340, barmode="group", yaxis=dict(tickformat=".0%"))
    ui.show(fig)
    ui.note("Each regime is a probability distribution over 72 kinds of transaction (direction × channel × size band × time since the "
            "previous one). Names come from what a regime over-represents compared with all transactions. Reported and dismissed "
            "histories spend similar time in each regime, which is why the regimes add little to the risk score.", ui.VIOLET)

# ---------------------------------------------------------------- transitions
with tabs[1]:
    a, b = st.columns([3, 2])
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, K, endpoint=False)
    xy = np.c_[np.cos(ang), np.sin(ang)]
    off = A - np.diag(np.diag(A))
    scale = off.max() or 1
    def base_fig():
        f = go.Figure()
        for j in range(K):
            for k in range(K):
                if j != k and off[j, k] > 0.02 * scale:
                    f.add_annotation(x=xy[k, 0] * .82 + xy[j, 0] * .18, y=xy[k, 1] * .82 + xy[j, 1] * .18,
                                     ax=xy[j, 0] * .82 + xy[k, 0] * .18, ay=xy[j, 1] * .82 + xy[k, 1] * .18,
                                     xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                                     arrowsize=1, arrowwidth=1 + 6 * off[j, k] / scale, arrowcolor=ui.rgba(col[j], .75))
        return f
    fig = base_fig()
    fig.add_scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[f"R{k + 1}" for k in range(K)], textfont=dict(color="#fff", size=13),
                    marker=dict(size=[40 + 40 * A[k, k] for k in range(K)], color=col, line=dict(color="#fff", width=1.5)),
                    hovertext=[f"{names[k]}<br>stay {A[k, k]:.1%}" for k in range(K)], hoverinfo="text", showlegend=False)
    # animation: a token walks the most probable regime path of a high-risk alert
    pick = VP.iloc[0]
    runs = [int(pick.states[0])] + [int(s) for s, p in zip(pick.states[1:], pick.states[:-1]) if s != p]
    runs = runs[:40] if len(runs) > 1 else runs * 2
    fig.add_scatter(x=[xy[runs[0], 0]], y=[xy[runs[0], 1]], mode="markers", marker=dict(size=22, color="#FFFFFF", symbol="circle-open",
                    line=dict(width=4, color="#FFFFFF")), showlegend=False, hoverinfo="skip")
    fig.frames = [go.Frame(data=[go.Scatter(x=xy[:, 0], y=xy[:, 1]), go.Scatter(x=[xy[r, 0]], y=[xy[r, 1]])], traces=[0, 1], name=str(i))
                  for i, r in enumerate(runs)]
    ui.fig_layout(fig, "Regime transitions (arrow = chance of switching, bubble = chance of staying)", 520,
                  xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x"),
                  updatemenus=[dict(type="buttons", x=0.02, y=0.02, xanchor="left", yanchor="bottom", bgcolor="#2A3350", bordercolor=ui.BLUE,
                                    font=dict(color="#FFFFFF", size=12), showactive=False, pad=dict(l=8, r=8, t=4, b=4), buttons=[dict(label="▶ Play a client's path", method="animate",
                                    args=[None, dict(frame=dict(duration=700, redraw=True), transition=dict(duration=350), fromcurrent=True)])])])
    with a: ui.show(fig)
    hm = go.Figure(go.Heatmap(z=A, x=[f"R{k + 1}" for k in range(K)], y=[f"R{k + 1}" for k in range(K)], zmin=0, zmax=1,
                              colorscale=[[0, "#10131A"], [0.02, "#2A3350"], [0.5, ui.VIOLET], [1, ui.ORANGE]],
                              text=np.round(A, 3), texttemplate="%{text}", colorbar=dict(thickness=10)))
    ui.fig_layout(hm, "Transition matrix p(next regime | current regime)", 520, xaxis=dict(title="next"), yaxis=dict(title="current", autorange="reversed"))
    with b: ui.show(hm)
    ui.note("The diagonal is close to 1: once a client is in a regime they usually stay for dozens to hundreds of transactions. "
            "Regimes behave like long phases of a client's life rather than minute-to-minute switching. "
            f"Play shows the regime path of alert #{int(pick.alert_id)}.")

# ---------------------------------------------------------------- alert timeline
with tabs[2]:
    Q = ui.table("scored_alerts.parquet").set_index("alert_id")
    ids = [a for a in Q[Q.role == "test"].sort_values("risk", ascending=False).index if a in set(VP.alert_id)]
    default = st.session_state.get("alert_pick")
    sel = st.selectbox("Alert (highest risk first)", ids, index=ids.index(default) if default in ids else 0, format_func=lambda a: f"#{a}")
    row = VP.set_index("alert_id").loc[sel]
    stv, days, sym = np.asarray(row.states), np.asarray(row.days), np.asarray(row.symbols)
    f_ = SF.set_index("alert_id").loc[sel]
    c = st.columns(4)
    ui.kpi(c[0], "Transactions in sequence", f"{len(stv):,}", "", ui.BLUE, count=len(stv))
    ui.kpi(c[1], "Regime switches", f"{int((np.diff(stv) != 0).sum())}", f"{f_.hmm_switch_rate:.2f} per 100 transactions", ui.VIOLET)
    pct = float((SF.hmm_llr < f_.hmm_llr).mean())
    ui.kpi(c[2], "Sequence suspicion score", f"{f_.hmm_llr:+.3f}", f"higher than {pct:.0%} of alerts", ui.ORANGE, ring=pct)
    ui.kpi(c[3], "Regime change in last 30 days", f"{f_.hmm_regime_shift:.0%}", "0% = same mix as before", ui.GREEN, ring=float(f_.hmm_regime_shift))
    st.write("")
    fig = go.Figure()
    for k in range(K):
        m = stv == k
        if m.any():
            fig.add_scattergl(x=-days[m], y=[names[k]] * int(m.sum()), mode="markers", name=names[k],
                              marker=dict(color=col[k], size=7, symbol="line-ns-open", line=dict(width=2, color=col[k])),
                              customdata=[ui_sym for ui_sym in sym[m]], hovertemplate="%{x:.1f} days before alert<extra></extra>")
    fig.add_scatter(x=-days, y=[names[s] for s in stv], mode="lines", line=dict(color=ui.rgba("#FFFFFF", .25), width=1, shape="hv"),
                    hoverinfo="skip", showlegend=False)
    fig.add_vrect(x0=-30, x1=0, fillcolor=ui.rgba(ui.ORANGE, .08), line_width=0)
    ui.fig_layout(fig, "Most probable regime path (Viterbi) across the year before the alert", 380,
                  xaxis=dict(title="Days before alert"), yaxis=dict(categoryorder="array", categoryarray=names[::-1]),
                  margin=dict(l=230, r=20, t=48, b=44), showlegend=False)
    ui.show(fig, explore=True)
    share = np.bincount(stv, minlength=K) / len(stv)
    fig = go.Figure(go.Bar(x=share, y=names, orientation="h", marker_color=col, text=[f"{v:.0%}" for v in share], textposition="outside"))
    ui.fig_layout(fig, "Time spent in each regime by this client", 260, xaxis=dict(tickformat=".0%", range=[0, 1.1]),
                  margin=dict(l=230, r=20, t=48, b=30), yaxis=dict(categoryorder="array", categoryarray=names[::-1]))
    ui.show(fig)

# ---------------------------------------------------------------- suspicion scores
with tabs[3]:
    a, b = st.columns([3, 2])
    hst = sq["llr_hist"]
    mid = (np.array(hst["bins"][:-1]) + np.array(hst["bins"][1:])) / 2
    fig = go.Figure()
    fig.add_scatter(x=mid, y=hst["dismiss"], name="Dismissed", mode="lines", line=dict(color=ui.DISMISS, width=2.5), fill="tozeroy",
                    fillcolor=ui.rgba(ui.DISMISS, .2))
    fig.add_scatter(x=mid, y=hst["report"], name="Reported", mode="lines", line=dict(color=ui.REPORT, width=2.5), fill="tozeroy",
                    fillcolor=ui.rgba(ui.REPORT, .2))
    ui.fig_layout(fig, "Sequence suspicion score on 2021 Q3-Q4 alerts", 400,
                  xaxis=dict(title="log p(sequence | reported-type HMM) − log p(sequence | dismissed-type HMM), per transaction"),
                  yaxis=dict(title="Density"))
    with a: ui.show(fig)
    with b:
        st.markdown("##### Each sequence score on its own")
        st.dataframe(pd.DataFrame([{"Score": r["score"], "PR-AUC": r["pr_auc"], "ROC-AUC": r["roc_auc"]} for r in sq["standalone"]]),
                     hide_index=True, width="stretch", column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                                                     "ROC-AUC": st.column_config.NumberColumn(format="%.3f")})
        st.caption(f"No skill: PR-AUC {sq['base_rate_test']:.3f}, ROC-AUC 0.500.")
    ui.note("Two HMMs are trained, one on reported and one on dismissed 2020 histories. A sequence that the reported-type model "
            "explains better scores higher. This is a generative classifier over sequences. Training alerts are scored by models "
            "fitted on the other half of 2020, so the score never sees its own label.", ui.ORANGE)

# ---------------------------------------------------------------- model fit
with tabs[4]:
    a, b = st.columns(2)
    ks = pd.DataFrame(sq["k_selection"])
    fig = go.Figure()
    fig.add_scatter(x=ks.K, y=ks.val_ll_per_txn, mode="lines+markers", name="Held-out (early 2021)", line=dict(color=ui.ORANGE, width=2.5))
    fig.add_scatter(x=ks.K, y=ks.train_ll_per_txn, mode="lines+markers", name="Training (2020)", line=dict(color=ui.BLUE, width=2.5))
    fig.add_vline(x=K, line=dict(color=ui.GREEN, dash="dash"))
    ui.fig_layout(fig, "Number of regimes: log-likelihood per transaction", 380, xaxis=dict(title="Hidden states", dtick=1))
    with a: ui.show(fig)
    fig = go.Figure()
    fig.add_scatter(y=sq["history"], mode="lines+markers", name="Behaviour model", line=dict(color=ui.VIOLET, width=2.5))
    for k_, c_ in (("report", ui.REPORT), ("dismiss", ui.DISMISS)):
        fig.add_scatter(y=sq["class_history"][k_], mode="lines", name=f"{k_.title()}-type model", line=dict(color=c_, width=2))
    ui.fig_layout(fig, "Baum-Welch (EM): log-likelihood per transaction after each iteration", 380, xaxis=dict(title="Iteration"))
    with b: ui.show(fig)
    st.dataframe(ks.rename(columns={"K": "Hidden states", "val_ll_per_txn": "Held-out log-lik / txn", "bic": "BIC",
                                    "train_ll_per_txn": "Training log-lik / txn"}), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        ui.note("<b>Why the numbers do not underflow.</b> A year of transactions multiplies hundreds of probabilities, far below the "
                "smallest number a computer can store. The forward pass rescales its values at every step and keeps the scaling "
                "factors; the log-likelihood is the sum of their logs. Viterbi works in log space, so it needs no scaling.", ui.GREEN)
    with b:
        ui.note("<b>Beyond discrete regimes.</b> A linear dynamical system replaces the discrete hidden state with a continuous one "
                "(for example a client's underlying activity level) and Gaussian noise; its forward pass is the Kalman filter and "
                "particle filters handle non-linear versions. It was not used here because channel and direction are categorical.",
                ui.CYAN)
