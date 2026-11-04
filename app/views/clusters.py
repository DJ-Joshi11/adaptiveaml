"""Behavioural profiles: which alerts behave alike, and does any group concentrate the reported ones?"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

cl = ui.clustering()
if not cl:
    ui.setup("Behavioural Profiles", "Patterns")
    st.info("Run stage 8 first:  python run_pipeline.py --stage 8")
    st.stop()
C = ui.table("clusters.parquet")
ev = cl["evaluation"]
K, G = cl["kmeans"]["K"], cl["gmm"]["K"]
prof_tr, prof_te = cl["profiles"]["kmeans"]["train"], cl["profiles"]["kmeans"]["test"]
test = C[C.role == "test"]
base = float(test.y.mean())

ui.setup("Behavioural Profiles", "Patterns",
         f"Alerts grouped by how they behave, without looking at the outcome. Learned on 2020 alerts and applied unchanged to "
         f"{len(test):,} alerts from 2021 Q3-Q4.")

rates = [p["report_rate"] for p in prof_te]
c = st.columns(4)
ui.kpi(c[0], "Profiles found", f"{K}", "K-Means, chosen by silhouette", ui.VIOLET, count=K)
ui.kpi(c[1], "Dimensions kept", f"{cl['pca']['d80']}", f"of {cl['n_features']} features, 80% of variance", ui.CYAN,
       count=cl["pca"]["d80"], spark=np.cumsum(cl["pca"]["ratio"][:20]))
ui.kpi(c[2], "Report rate across profiles", f"{min(rates):.0%} to {max(rates):.0%}", f"overall {base:.0%} on the test quarters", ui.ORANGE,
       ring=max(rates))
ui.kpi(c[3], "Separation (silhouette)", f"{ev['K-Means']['test']['silhouette']:.2f}", "-1 to 1; above 0.5 would be strong", ui.BLUE)
st.write("")

_lo = min(prof_te, key=lambda p: p["report_rate"])
_hi = max(prof_te, key=lambda p: p["report_rate"])
ui.takeaway([f"Alerts fall into <b>{K} broad behavioural profiles</b>, found without ever looking at whether they were reported.",
             f"The profiles are real but blurry: they overlap a lot (separation {ev['K-Means']['test']['silhouette']:.2f} on a scale "
             "where 0.5+ would be clear-cut groups).",
             f"They barely differ in risk: from <b>{_lo['report_rate']:.0%}</b> (P{_lo['cluster'] + 1}) to <b>{_hi['report_rate']:.0%}</b> "
             f"(P{_hi['cluster'] + 1}) reported, around {base:.0%} overall. Useful to <i>describe</i> clients, not to decide who to escalate."])
ui.plain("**Clustering** groups alerts that look alike, like sorting customers into segments, without being told which alerts were "
         "reported. Because 84 behaviours are too many to compare directly, **PCA** first squeezes them into a few summary directions "
         "that keep most of the information. The map shows every alert on the two most important of those directions.")
tabs = st.tabs(["Profile map", "Profiles", "How many profiles?", "Latent dimensions", "Hierarchy", "Quality"])

# ---------------------------------------------------------------- map
with tabs[0]:
    f = st.columns([2, 2, 2, 3])
    colour = f[0].selectbox("Colour by", ["Profile (K-Means)", "Profile (Gaussian mixture)", "Outcome", "Assignment confidence"])
    dims = f[1].radio("View", ["2-D", "3-D"], horizontal=True)
    period = f[2].selectbox("Alerts", ["2021 Q3-Q4 (test)", "2020 (training)"])
    n_show = f[3].slider("Points shown", 500, 6000, 3000, 500)
    D = (C[C.role == "test"] if period.startswith("2021") else C[C.role == "train"]).sample(frac=1, random_state=1).head(n_show)
    names_km = {p["cluster"]: p["name"] for p in prof_tr}
    slot = ui.skeleton(560)
    fig = go.Figure()
    hover = ("Alert %{customdata[0]}<br>%{customdata[1]}<br>Outcome: %{customdata[2]}<br>"
             "Mixture confidence %{customdata[3]:.0%}<extra></extra>")
    def cd(d):
        return np.stack([d.alert_id, d.kmeans.map(lambda k: f"P{k + 1} · {names_km.get(k, '')}"),
                         np.where(d.y == 1, "Reported", "Dismissed"), d.gmm_conf], 1)
    groups = []
    if colour.startswith("Profile (K"):
        groups = [(f"P{k + 1}", D[D.kmeans == k], ui.SERIES[k % 12]) for k in sorted(D.kmeans.unique())]
    elif colour.startswith("Profile (G"):
        groups = [(f"M{k + 1}", D[D.gmm == k], ui.SERIES[k % 12]) for k in sorted(D.gmm.unique())]
    elif colour == "Outcome":
        groups = [("Dismissed", D[D.y == 0], ui.DISMISS), ("Reported", D[D.y == 1], ui.REPORT)]
    for name, d, col in groups:
        kw = dict(mode="markers", name=name, customdata=cd(d), hovertemplate=hover,
                  marker=dict(color=col, size=4 if dims == "3-D" else 6, opacity=0.75, line=dict(width=0)))
        fig.add_trace(go.Scatter3d(x=d.pc1, y=d.pc2, z=d.pc3, **kw) if dims == "3-D" else go.Scattergl(x=d.pc1, y=d.pc2, **kw))
    if colour == "Assignment confidence":
        kw = dict(mode="markers", customdata=cd(D), hovertemplate=hover, name="alerts",
                  marker=dict(color=D.gmm_conf, colorscale=[[0, ui.RED], [0.5, ui.YELLOW], [1, ui.GREEN]], size=5 if dims == "3-D" else 6,
                              opacity=0.8, colorbar=dict(title="confidence", thickness=10)))
        fig.add_trace(go.Scatter3d(x=D.pc1, y=D.pc2, z=D.pc3, **kw) if dims == "3-D" else go.Scattergl(x=D.pc1, y=D.pc2, **kw))
    if dims == "3-D":
        ui.fig_layout(fig, "Alerts on the first three principal components (drag to rotate)", 560, margin=dict(l=0, r=0, t=48, b=0),
                      scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", bgcolor="rgba(0,0,0,0)",
                                 xaxis=dict(gridcolor=ui.LINE), yaxis=dict(gridcolor=ui.LINE), zaxis=dict(gridcolor=ui.LINE)))
        ui.show(fig, explore=True, slot=slot)
        sel = None
    else:
        ui.fig_layout(fig, "Alerts on the first two principal components (box or lasso to select)", 560,
                      xaxis=dict(title=f"PC1 ({cl['pca']['ratio'][0]:.0%} of variance)"),
                      yaxis=dict(title=f"PC2 ({cl['pca']['ratio'][1]:.0%} of variance)"), dragmode="lasso")
        sel = ui.show(fig, explore=True, slot=slot, on_select="rerun", selection_mode=["points", "box", "lasso"], key="pca_map")
    picked = []
    if sel and getattr(sel, "selection", None) and sel.selection.get("points"):
        picked = [int(pt["customdata"][0]) for pt in sel.selection["points"] if pt.get("customdata")]
    if picked:
        P = C[C.alert_id.isin(picked)]
        a, b, cc = st.columns(3)
        ui.kpi(a, "Alerts selected", f"{len(P):,}", "", ui.VIOLET, count=len(P))
        ui.kpi(b, "Report rate in selection", f"{P.y.mean():.0%}", f"vs {base:.0%} overall", ui.ORANGE, ring=P.y.mean())
        ui.kpi(cc, "Most common profile", f"P{int(P.kmeans.mode()[0]) + 1}", names_km.get(int(P.kmeans.mode()[0]), "")[:60], ui.CYAN)
        st.write("")
        pick = st.selectbox("Open an alert from the selection", P.alert_id.tolist(), format_func=lambda x: f"#{x}")
        if st.button("Open in Alert Detail", type="primary"):
            st.session_state["alert_pick"] = int(pick)
            st.switch_page("views/alert.py")
    else:
        st.caption("Tip: switch to 2-D and draw a lasso around any region to see its report rate and open its alerts.")
    ui.note("The map is a projection for the eye only; profiles are found in the full "
            f"{cl['pca']['d80']}-dimensional reduced space. Overlapping colours mean the profiles are not sharply separated.")

# ---------------------------------------------------------------- profiles
with tabs[1]:
    ui.takeaway([f"<b>P{_lo['cluster'] + 1}</b> ({_lo['report_rate']:.0%} reported): {_lo['name']}.",
                 f"<b>P{_hi['cluster'] + 1}</b> ({_hi['report_rate']:.0%} reported): {_hi['name']}.",
                 "Diamonds (2020) sit close to the bars (2021): the profiles mean the same thing a year later."], "The profiles in words")
    tr_map = {p["cluster"]: p for p in prof_tr}
    fig = go.Figure()
    for p in prof_te:
        k = p["cluster"]
        fig.add_bar(x=[f"P{k + 1}"], y=[p["report_rate"]], marker_color=ui.SERIES[k % 12], showlegend=False,
                    error_y=dict(type="data", symmetric=False, array=[p["hi"] - p["report_rate"]],
                                 arrayminus=[p["report_rate"] - p["lo"]], color=ui.TEXT, thickness=1.2),
                    hovertemplate=f"P{k + 1}: %{{y:.1%}} reported<br>{p['size']:,} test alerts<extra></extra>")
        fig.add_scatter(x=[f"P{k + 1}"], y=[tr_map[k]["report_rate"]], mode="markers", showlegend=k == prof_te[0]["cluster"],
                        name="2020 rate", marker=dict(symbol="diamond", size=11, color="#FFFFFF", line=dict(color=ui.BG, width=1)))
    fig.add_hline(y=base, line=dict(color=ui.RED, dash="dash", width=1.5))
    fig.add_annotation(x=1, xref="paper", y=base, text="overall rate", showarrow=False, yshift=10, xanchor="right", font=dict(color=ui.RED))
    ui.fig_layout(fig, "Report rate per profile on 2021 Q3-Q4 (bars, 95% interval) and in 2020 (diamonds)", 380,
                  yaxis=dict(title="Share reported", tickformat=".0%", rangemode="tozero"))
    ui.show(fig)
    cols = st.columns(min(K, 3))
    for i, p in enumerate(prof_te):
        k = p["cluster"]
        with cols[i % len(cols)]:
            col = ui.SERIES[k % 12]
            rate_pill = ui.pill(f"{p['report_rate']:.0%} reported", ui.ORANGE)
            sig = "".join(f"<div style='display:flex;justify-content:space-between;font-size:.8rem;margin:2px 0'>"
                          f"<span style='color:#C9CFDB'>{s['label']}</span><b style='color:{ui.ORANGE if s['z'] > 0 else ui.BLUE}'>"
                          f"{s['z']:+.2f}σ</b></div>" for s in p["signature"][:5])
            st.markdown(f"<div class='card' style='border-top:3px solid {col};margin-bottom:12px'>"
                        f"<div style='display:flex;justify-content:space-between'><b style='font-size:1.05rem'>Profile P{k + 1}</b>"
                        f"{rate_pill}</div>"
                        f"<div style='color:{ui.MUTED};font-size:.8rem;margin:4px 0 10px'>{p['size']:,} test alerts · {p['share']:.0%} of queue</div>"
                        f"{sig}</div>", unsafe_allow_html=True)
    ui.note("σ values show how far the profile's average sits from all training alerts on that feature. Profiles describe <b>who</b> "
            "the alerts look like; the report-rate bars show whether that helps to decide <b>which</b> to escalate.", ui.VIOLET)

# ---------------------------------------------------------------- choosing K
with tabs[2]:
    ui.takeaway([f"Three numbers were checked to pick the number of profiles; the clearest split is <b>{K}</b> groups.",
                 "K-Means puts each alert in exactly one group; the Gaussian mixture gives each alert a probability for every group - "
                 f"and is usually very sure ({C.gmm_conf.median():.0%} for the typical alert).",
                 "The mixture keeps preferring more groups the more we allow, a sign there is no small set of natural groups."])
    a, b = st.columns(2)
    el = pd.DataFrame(cl["kmeans"]["elbow"])
    fig = go.Figure()
    fig.add_scatter(x=el.k, y=el.inertia, mode="lines+markers", name="Within-cluster sum of squares", line=dict(color=ui.BLUE, width=2.5))
    fig.add_scatter(x=el.k, y=el.silhouette, mode="lines+markers", name="Silhouette", yaxis="y2", line=dict(color=ui.ORANGE, width=2.5))
    fig.add_vline(x=K, line=dict(color=ui.GREEN, dash="dash"))
    ui.fig_layout(fig, "K-Means: elbow and silhouette", 400, xaxis=dict(title="Number of profiles K", dtick=1),
                  yaxis=dict(title="Inertia"), yaxis2=dict(title="Silhouette", overlaying="y", side="right", showgrid=False))
    with a: ui.show(fig)
    bi = pd.DataFrame(cl["gmm"]["bic"])
    fig = go.Figure()
    fig.add_scatter(x=bi.k, y=bi.bic, mode="lines+markers", name="BIC", line=dict(color=ui.VIOLET, width=2.5))
    fig.add_scatter(x=bi.k, y=bi.aic, mode="lines+markers", name="AIC", line=dict(color=ui.CYAN, width=2, dash="dot"))
    fig.add_vline(x=G, line=dict(color=ui.GREEN, dash="dash"))
    ui.fig_layout(fig, "Gaussian mixture: information criteria (lower is better)", 400, xaxis=dict(title="Components", dtick=1))
    with b: ui.show(fig)
    a, b = st.columns([3, 2])
    em = cl["gmm"]["em_curve"]
    fig = go.Figure(go.Scatter(x=list(range(1, len(em) + 1)), y=em, mode="lines+markers", line=dict(color=ui.GREEN, width=2.5),
                               marker=dict(size=5)))
    ui.fig_layout(fig, "EM for the Gaussian mixture: log-likelihood after each E-step / M-step cycle", 360,
                  xaxis=dict(title="EM cycle"), yaxis=dict(title="Average log-likelihood"))
    with a: ui.show(fig)
    with b:
        st.markdown("##### Hard vs soft assignment")
        st.markdown(f"- K-Means gives every alert exactly one profile.\n- The Gaussian mixture gives a probability for each of its "
                    f"{G} components; the median alert's top component has **{C.gmm_conf.median():.0%}** probability.\n"
                    f"- Agreement between the two partitions (adjusted Rand): **{ev['kmeans_vs_gmm_ari']:.2f}**.")
        fig = go.Figure(go.Histogram(x=C.gmm_conf, nbinsx=30, marker_color=ui.GREEN, opacity=.85))
        ui.fig_layout(fig, "How sure is the mixture?", 220, xaxis=dict(title="Top component probability", tickformat=".0%"),
                      margin=dict(l=40, r=10, t=40, b=40))
        ui.show(fig)
    ui.note(f"K-Means silhouette peaks at K = {K}; the BIC of the mixture keeps falling up to the largest K tried ({G}), "
            "which usually means the data has no small number of natural, well-separated groups.")

# ---------------------------------------------------------------- latent dimensions
with tabs[3]:
    ui.takeaway([f"<b>{cl['pca']['d80']}</b> summary directions keep 80% of the information in the 84 behaviours - many behaviours "
                 "repeat each other.",
                 f"The single most important direction (PC1, {cl['pca']['ratio'][0]:.0%}) is essentially <b>how large the transactions are</b>."])
    a, b = st.columns(2)
    r = np.array(cl["pca"]["ratio"])
    fig = go.Figure()
    fig.add_bar(x=np.arange(1, len(r) + 1), y=r, name="Each component", marker_color=ui.rgba(ui.BLUE, .85))
    fig.add_scatter(x=np.arange(1, len(r) + 1), y=np.cumsum(r), name="Cumulative", yaxis="y2", mode="lines+markers",
                    line=dict(color=ui.ORANGE, width=2.5))
    fig.add_vline(x=cl["pca"]["d80"], line=dict(color=ui.GREEN, dash="dash"))
    ui.fig_layout(fig, "Scree plot: variance explained by each principal component", 400, xaxis=dict(title="Component"),
                  margin=dict(l=55, r=55, t=48, b=44),
                  yaxis=dict(title="Share of variance", tickformat=".0%"),
                  yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.02], showgrid=False))
    with a: ui.show(fig)
    ds = pd.DataFrame(cl["dim_selection"])
    fig = go.Figure()
    fig.add_scatter(x=ds.k, y=ds.ppca, mode="lines+markers", name="Probabilistic PCA", line=dict(color=ui.BLUE, width=2.5))
    fig.add_scatter(x=ds.k, y=ds.fa, mode="lines+markers", name="Factor analysis", line=dict(color=ui.VIOLET, width=2.5))
    ui.fig_layout(fig, "Held-out log-likelihood vs number of latent dimensions", 400,
                  xaxis=dict(title="Latent dimensions"), yaxis=dict(title="Held-out log-likelihood per alert"))
    with b: ui.show(fig)
    st.markdown(ui.pill("Eigen-decomposition of the covariance and the SVD give identical variances" if cl["pca"]["svd_matches_eigen"]
                        else "Eigen and SVD results differ", ui.GREEN), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("##### What the first principal components capture")
        for ld in cl["pca"]["loadings"][:3]:
            st.markdown(f"**PC{ld['component']}** ({ld['variance']:.0%}): " + ", ".join(f"{t['label']} ({t['weight']:+.2f})" for t in ld["top"][:4]))
    with b:
        fa = cl["fa"]
        st.markdown(f"##### {fa['n_factors']} latent factors (varimax rotated)")
        for f_ in fa["factors"][:4]:
            st.markdown(f"**Factor {f_['factor']}**: " + ", ".join(f"{t['label']} ({t['loading']:+.2f})" for t in f_["top"][:3]))
        st.caption(f"Log-likelihood before rotation {fa['loglik_unrotated']:.4f}, after {fa['loglik_rotated']:.4f}: rotating the "
                   "factors changes the loadings but not the fit, which is why factor loadings are not unique.")

# ---------------------------------------------------------------- hierarchy
with tabs[4]:
    ui.takeaway(["A family tree of alerts: similar alerts merge first, and the height of a join shows how different the merged groups are.",
                 "How you measure \"different\" matters: the simplest rules (single and average link) lump almost everyone into one group; "
                 "Ward's rule gives balanced groups that agree with K-Means."])
    h = cl["hierarchical"]
    dg = h["dendrogram"]
    fig = go.Figure()
    for xs, ys in zip(dg["icoord"], dg["dcoord"]):
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=ui.VIOLET, width=1.6), hoverinfo="skip", showlegend=False)
    fig.update_xaxes(tickvals=[5 + 10 * i for i in range(len(dg["leaves"]))], ticktext=dg["leaves"], tickangle=-60, tickfont=dict(size=9))
    ui.fig_layout(fig, f"Dendrogram (Ward linkage, {h['sample']:,} training alerts, last 30 merges)", 420,
                  yaxis=dict(title="Merge distance"), xaxis=dict(title="Groups (size in brackets)"))
    ui.show(fig)
    H = pd.DataFrame(h["table"])
    st.dataframe(H.rename(columns={"metric": "Dissimilarity", "linkage": "Linkage", "cophenetic": "Cophenetic correlation",
                                   "silhouette": "Silhouette", "largest_cluster_share": "Largest group"}),
                 hide_index=True, width="stretch", column_config={
                     "Cophenetic correlation": st.column_config.NumberColumn(format="%.2f"),
                     "Silhouette": st.column_config.NumberColumn(format="%.2f"),
                     "Largest group": st.column_config.NumberColumn(format="percent")})
    ui.note("Single and average linkage score a high silhouette only because they put almost every alert in one group and leave a few "
            "outliers on their own (the chaining effect). Ward linkage, which merges by the increase in within-group variance, gives "
            "balanced groups and agrees with K-Means. Cophenetic correlation measures how faithfully each tree keeps the original distances.",
            ui.YELLOW)

# ---------------------------------------------------------------- quality
with tabs[5]:
    ui.takeaway(["Measured on 2021 alerts, the groups are compact but overlapping, and almost unrelated to the outcome "
                 f"(mutual information {ev['K-Means']['test']['nmi']:.3f}, where 0 = unrelated).",
                 "Purity looks high (0.82) only because 82% of all alerts are dismissed - any grouping scores that."])
    rows = []
    for name, v in ev.items():
        if not isinstance(v, dict):
            continue
        for period, q in v.items():
            rows.append({"Method": name, "Alerts": {"train": "2020", "test": "2021 Q3-Q4"}[period], **q})
    Q = pd.DataFrame(rows).rename(columns={"silhouette": "Silhouette ↑", "davies_bouldin": "Davies-Bouldin ↓", "purity": "Purity",
                                           "nmi": "Mutual information with outcome", "ari": "Adjusted Rand with outcome"})
    st.dataframe(Q, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in Q.columns if c not in ("Method", "Alerts")})
    a, b = st.columns(2)
    with a:
        ui.note(f"<b>Purity {ev['K-Means']['test']['purity']:.3f} vs {ev['baseline_purity']:.3f} for a single all-dismiss group.</b> "
                "With one class at 82%, purity cannot fall below 0.82, so it says nothing here. Mutual information and adjusted Rand "
                "close to zero say the same thing properly: the profiles are almost independent of the outcome.", ui.ORANGE)
    with b:
        ui.note("Silhouette compares each alert's distance to its own profile with its distance to the nearest other profile. "
                "Davies-Bouldin compares profile spread to the distance between profiles. Both describe the shape of the groups; "
                "only the outcome-based measures say whether the groups matter for escalation.")
