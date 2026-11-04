"""Model comparison: six classifiers under one protocol, on quarters none of them saw."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

ev, cl = ui.evaluation(), ui.classification()
rows, curves, base = ev["table"], ev["curves"], ev["base_rate_test"]
col = lambda m: ui.MODEL_COLORS.get(m, ui.MUTED)

ui.setup("Model Comparison", "Model",
         f"Trained on 2020, tuned and calibrated on early 2021, evaluated once on {ev['n_test']:,} later alerts. "
         f"A random ranking scores PR-AUC {base:.3f}.")

_tied = [x["model"] for x in rows if x.get("tied_with_champion")]
ui.takeaway([f"Six different ways of learning from the same alerts were tried. The best, <b>{ev['champion']}</b>, ranks alerts "
             f"<b>{rows[0]['lift']:.1f}×</b> better than random guessing.",
             (f"{', '.join(_tied)} are statistically tied with it: on this data the choice of method matters less than the features."
              if _tied else "The leader is clearly ahead of the others."),
             "Simple, fully explainable models are within touching distance of the best one."])
ui.plain("**PR-AUC** is a single score for how well a model puts reported alerts near the top of the list: 1.0 would be perfect, and "
         f"random guessing scores about {base:.2f}. The thin whiskers show the uncertainty: if two whiskers overlap, those two models "
         "cannot honestly be called different.")
tabs = st.tabs(["Leaderboard", "Curves", "Significance", "Stability over time", "Ablations", "Tuning"])


with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        r = rows[::-1]
        fig = go.Figure(go.Bar(
            x=[x["pr_auc"] for x in r], y=[x["model"] for x in r], orientation="h",
            marker_color=[col(x["model"]) for x in r], text=[f"{x['pr_auc']:.3f}" for x in r], textposition="inside", insidetextanchor="start",
            error_x=dict(type="data", symmetric=False, color=ui.TEXT, thickness=1.2,
                         array=[x["pr_auc_ci"][1] - x["pr_auc"] for x in r],
                         arrayminus=[x["pr_auc"] - x["pr_auc_ci"][0] for x in r])))
        ui.ref_line(fig, base)
        ui.fig_layout(fig, "PR-AUC on held-out quarters, with 95% bootstrap intervals", 400,
                      xaxis=dict(title="PR-AUC (average precision)"), margin=dict(l=160, r=20, t=48, b=44))
        ui.show(fig)
    with right:
        fig = go.Figure(go.Bar(
            x=[x["recall_at_budget"] for x in r], y=[x["model"] for x in r], orientation="h",
            marker_color=[ui.rgba(col(x["model"]), 0.8) for x in r], text=[ui.pct(x["recall_at_budget"]) for x in r],
            textposition="inside", insidetextanchor="start"))
        ui.ref_line(fig, ev["budget"], "random review")
        ui.fig_layout(fig, f"Reports caught when reviewing the top {ui.pct(ev['budget'])}", 400,
                      xaxis=dict(title="Share of reported alerts caught", tickformat=".0%", range=[0, 1]),
                      margin=dict(l=160, r=20, t=48, b=44))
        ui.show(fig)
    df = pd.DataFrame([{
        "Model": x["model"], "PR-AUC": x["pr_auc"], "95% interval": f"{x['pr_auc_ci'][0]:.3f} to {x['pr_auc_ci'][1]:.3f}",
        "Lift vs no skill": x["lift"], "ROC-AUC": x["roc_auc"], "Caught @ budget": x["recall_at_budget"],
        "Hit rate @ budget": x["precision_at_budget"], "Brier": x["brier"],
        "Status": "In service" if x["model"] == ev["champion"] else ("Tied with in-service" if x["tied_with_champion"] else "")}
        for x in rows] + [{"Model": x["model"], "PR-AUC": x["pr_auc"], "ROC-AUC": x["roc_auc"],
                           "Caught @ budget": x["recall_at_budget"], "Status": "Reference"} for x in ev["references"]])
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "PR-AUC": st.column_config.NumberColumn(format="%.3f"), "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Lift vs no skill": st.column_config.NumberColumn(format="%.2fx"), "Brier": st.column_config.NumberColumn(format="%.3f"),
        "Caught @ budget": st.column_config.NumberColumn(format="percent"),
        "Hit rate @ budget": st.column_config.NumberColumn(format="percent")})
    ui.note("Accuracy is deliberately absent. Dismissing every alert would be about "
            f"{ui.pct(1 - base)} accurate and catch nothing. Brier score measures calibration error (lower is better).")

with tabs[1]:
    pick = st.multiselect("Models", [x["model"] for x in rows], default=[x["model"] for x in rows])
    a, b = st.columns(2)
    fig = go.Figure()
    for m in pick:
        fig.add_scatter(x=curves[m]["pr"][0], y=curves[m]["pr"][1], mode="lines", name=m, line=dict(color=col(m), width=2))
        fig.add_scatter(x=[curves[m]["op_point"][0]], y=[curves[m]["op_point"][1]], mode="markers", showlegend=False,
                        marker=dict(color=col(m), size=10, line=dict(color=ui.TEXT, width=1)))
    fig.add_hline(y=base, line=dict(color=ui.RED, dash="dash", width=1))
    ui.fig_layout(fig, "Precision-recall (dots mark the review-budget operating point)", 420,
                  xaxis=dict(title="Recall: share of reported alerts caught"), yaxis=dict(title="Precision: hit rate", range=[0, 1.02]))
    with a: ui.show(fig)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, line=dict(color=ui.RED, dash="dash", width=1))
    for m in pick:
        fig.add_scatter(x=curves[m]["roc"][0], y=curves[m]["roc"][1], mode="lines", name=m, line=dict(color=col(m), width=2))
    ui.fig_layout(fig, "ROC", 420, xaxis=dict(title="False positive rate"), yaxis=dict(title="True positive rate"))
    with b: ui.show(fig)
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, line=dict(color=ui.RED, dash="dash", width=1))
    for m in pick:
        fig.add_scatter(x=curves[m]["gain"][0], y=curves[m]["gain"][1], mode="lines", name=m, line=dict(color=col(m), width=2))
    ui.fig_layout(fig, "Workload: reports caught vs queue reviewed", 420,
                  xaxis=dict(title="Share of queue reviewed", tickformat=".0%"), yaxis=dict(title="Reports caught", tickformat=".0%"))
    with a: ui.show(fig)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=ui.MUTED, dash="dash", width=1))
    for m in pick:
        rel = curves[m]["reliability"]
        fig.add_scatter(x=[p["pred"] for p in rel], y=[p["obs"] for p in rel], mode="lines+markers", name=m,
                        line=dict(color=col(m), width=2))
    ui.fig_layout(fig, "Calibration: does a score of 0.3 mean 30% get reported?", 420,
                  xaxis=dict(title="Risk score (after calibration)"), yaxis=dict(title="Observed report rate"))
    with b: ui.show(fig)

with tabs[2]:
    others = [x for x in rows if x["model"] != ev["champion"]]
    fig = go.Figure(go.Scatter(
        x=[x["diff_vs_champion"] for x in others], y=[x["model"] for x in others], mode="markers",
        marker=dict(size=12, color=[ui.MUTED if x["tied_with_champion"] else col(x["model"]) for x in others]),
        error_x=dict(type="data", symmetric=False, color=ui.TEXT, thickness=1.2,
                     array=[x["diff_ci"][1] - x["diff_vs_champion"] for x in others],
                     arrayminus=[x["diff_vs_champion"] - x["diff_ci"][0] for x in others])))
    fig.add_vline(x=0, line=dict(color=ui.TEXT, width=1))
    ui.fig_layout(fig, f"PR-AUC difference vs {ev['champion']} (paired bootstrap, 95% interval)", 340,
                  xaxis=dict(title="Difference in PR-AUC"), margin=dict(l=170, r=20, t=48, b=44))
    ui.show(fig)
    st.dataframe(pd.DataFrame([{
        "Model": x["model"], "Difference": x["diff_vs_champion"],
        "95% interval": f"{x['diff_ci'][0]:+.3f} to {x['diff_ci'][1]:+.3f}",
        "Verdict": "Statistically tied" if x["tied_with_champion"] else ("Worse" if x["diff_vs_champion"] < 0 else "Better"),
        "McNemar p (at budget)": x["mcnemar_p_vs_champion"]} for x in others]),
        hide_index=True, width="stretch", column_config={
            "Difference": st.column_config.NumberColumn(format="%+.3f"),
            "McNemar p (at budget)": st.column_config.NumberColumn(format="%.4f")})
    ui.note("If an interval crosses zero the two models cannot be separated on this data. Both tests use the same held-out "
            "alerts: the bootstrap compares ranking quality, McNemar compares the flag decisions at the review budget.")

with tabs[3]:
    bq = ev["by_quarter"]
    fig = go.Figure()
    for x in rows:
        fig.add_scatter(x=[b["quarter"] for b in bq], y=[b[x["model"]] / b["base_rate"] for b in bq],
                        mode="lines+markers", name=x["model"], line=dict(color=col(x["model"]), width=2))
    fig.add_hline(y=1, line=dict(color=ui.RED, dash="dash", width=1), annotation_text="no skill", annotation_font_color=ui.RED)
    ui.fig_layout(fig, "Lift over each quarter's own base rate", 400, yaxis=dict(title="PR-AUC / base rate"))
    ui.show(fig)
    st.dataframe(pd.DataFrame([{"Quarter": b["quarter"], "Used for": {"valid": "Tuning and calibration", "test": "Final evaluation"}[b["role"]],
                                "Alerts": b["n"], "Report rate": b["base_rate"], **{x["model"]: b[x["model"]] for x in rows}} for b in bq]),
                 hide_index=True, width="stretch", column_config={"Report rate": st.column_config.NumberColumn(format="percent")})
    ui.note("PR-AUC depends on how many alerts are reported in a quarter, so quarters are compared on lift. "
            "A downward slope away from the training year is the signature of behavioural drift.")

with tabs[4]:
    a, b = st.columns(2)
    ab = ev["ablation_burst"]
    fig = go.Figure()
    fig.add_bar(x=[x["model"] for x in ab], y=[x["without"] for x in ab], name="Without burst signal", marker_color=ui.rgba(ui.MUTED, 0.6))
    fig.add_bar(x=[x["model"] for x in ab], y=[x["with"] for x in ab], name="With burst signal", marker_color=ui.ORANGE)
    ui.fig_layout(fig, "Does the activity-burst signal add ranking quality?", 400, barmode="group", yaxis=dict(title="PR-AUC"))
    with a:
        ui.show(fig)
        st.dataframe(pd.DataFrame([{"Model": x["model"], "Gain": x["gain"],
                                    "95% interval": f"{x['gain_ci'][0]:+.3f} to {x['gain_ci'][1]:+.3f}",
                                    "Verdict": "Helps" if x["significant"] else "Hurts" if x["gain_ci"][1] < 0 else "No measurable effect"} for x in ab]),
                     hide_index=True, width="stretch", column_config={"Gain": st.column_config.NumberColumn(format="%+.3f")})
    im = pd.DataFrame(ev["ablation_imbalance"])
    arm_names = {"none": "No handling", "weights": "Class weights", "smote": "SMOTE oversampling"}
    fig = go.Figure()
    for arm, c_ in (("none", ui.MUTED), ("weights", ui.BLUE), ("smote", ui.VIOLET)):
        d = im[im.arm == arm]
        fig.add_bar(x=d.model, y=d.pr_auc, name=arm_names[arm], marker_color=c_)
    ui.fig_layout(fig, "Imbalance strategy compared on the same alerts", 400, barmode="group", yaxis=dict(title="PR-AUC"))
    with b:
        ui.show(fig)
        st.dataframe(im.assign(arm=im.arm.map(arm_names)).pivot(index="model", columns="arm", values="recall_at_budget").reset_index()
                     .rename(columns={"model": "Caught @ budget"}), hide_index=True, width="stretch")
    af = ev.get("ablation_features", [])
    if af:
        fig = go.Figure()
        for key, name_, c_ in (("behaviour_only", "Behavioural features only", ui.BLUE),
                               ("profile_only", "Channel-direction profile only", ui.CYAN), ("all", "Both together", ui.ORANGE)):
            fig.add_bar(x=[x["model"] for x in af], y=[x[key] for x in af], name=name_, marker_color=c_)
        fig.add_hline(y=base, line=dict(color=ui.RED, dash="dash", width=1), annotation_text="no skill", annotation_font_color=ui.RED)
        ui.fig_layout(fig, "Which feature family carries the signal?", 400, barmode="group", yaxis=dict(title="PR-AUC"))
        ui.show(fig)
        st.dataframe(pd.DataFrame([{"Model": x["model"], "Behavioural only": x["behaviour_only"], "Profile only": x["profile_only"],
                                    "Both": x["all"], "Behavioural adds on top": x["behaviour_adds"],
                                    "95% interval": f"{x['behaviour_adds_ci'][0]:+.3f} to {x['behaviour_adds_ci'][1]:+.3f}",
                                    "Verdict": "Adds signal" if x["significant"] else "Hurts" if x["behaviour_adds_ci"][1] < 0 else "No measurable effect"}
                                   for x in af]), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("Behavioural only", "Profile only", "Both")}
                     | {"Behavioural adds on top": st.column_config.NumberColumn(format="%+.3f")})
    ui.note("Each experiment changes one thing and keep everything else fixed. The burst signal is partly redundant for "
            "linear models, which already see the recent and historical counts it is built from.")

with tabs[5]:
    m = st.selectbox("Model", list(cl["models"]))
    info = cl["models"][m]
    c = st.columns(4)
    ui.kpi(c[0], "Selected settings", ", ".join(f"{k}={v}" for k, v in info["best_params"].items()), "chosen by forward-chaining", col(m))
    ui.kpi(c[1], "Forward-chaining PR-AUC", f"{info['cv_pr_auc']:.3f}", "mean over later-quarter folds", ui.BLUE)
    ui.kpi(c[2], "Imbalance handling", info["default_imbalance"], "", ui.VIOLET)
    extra = (f"{info['n_support_vectors']:,}" if "n_support_vectors" in info else f"{info['leaves']}" if "leaves" in info else f"{info['fit_seconds']}s")
    ui.kpi(c[3], "Support vectors" if "n_support_vectors" in info else "Leaves" if "leaves" in info else "Search time", extra,
           f"{ui.pct(info['support_vector_share'])} of training alerts" if "support_vector_share" in info else "", ui.GREEN)
    st.write("")
    g = pd.DataFrame([{**x["params"], "PR-AUC": x["cv_pr_auc"]} for x in info["grid"]]).sort_values("PR-AUC", ascending=False)
    st.dataframe(g, hide_index=True, width="stretch", column_config={"PR-AUC": st.column_config.NumberColumn(format="%.4f")})
    ui.note("Each setting is fitted on earlier 2020 quarters and scored on the next one (Q1 then Q2, Q1-Q2 then Q3, Q1-Q3 then Q4). "
            "Shuffled cross-validation is never used: it would let the model learn from the future.")
    lin = next((x for x in ev["references"] if "linear kernel" in x["model"]), None)
    rbf = next(x for x in rows if x["model"] == "SVM (RBF)")
    lr_, nb, lda = (next(x for x in rows if x["model"] == n) for n in ("Logistic Regression", "Naive Bayes", "LDA"))
    a, b = st.columns(2)
    if lin:
        a.markdown("##### Is the non-linear boundary worth it?")
        a.dataframe(pd.DataFrame([{"Boundary": "Linear kernel", "PR-AUC": lin["pr_auc"]}, {"Boundary": "RBF kernel", "PR-AUC": rbf["pr_auc"]}]),
                    hide_index=True, width="stretch", column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f")})
    b.markdown("##### Modelling the classes vs modelling the boundary")
    b.dataframe(pd.DataFrame([{"Approach": "Models each class, independent features", "Model": "Naive Bayes", "PR-AUC": nb["pr_auc"], "Brier": nb["brier"]},
                              {"Approach": "Models each class, shared covariance", "Model": "LDA", "PR-AUC": lda["pr_auc"], "Brier": lda["brier"]},
                              {"Approach": "Models the boundary directly", "Model": "Logistic Regression", "PR-AUC": lr_["pr_auc"], "Brier": lr_["brier"]}]),
                hide_index=True, width="stretch", column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                                                          "Brier": st.column_config.NumberColumn(format="%.3f")})
