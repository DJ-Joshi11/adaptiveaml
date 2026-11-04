"""What's new: the landing page for this release - what was added, the headline results, and a guided tour."""
import numpy as np
import streamlit as st

import ui

cl, sq, en, ev = ui.clustering(), ui.sequences(), ui.enrichment(), ui.evaluation()

ui.setup("What's new in AdaptiveAML", "Release 2.0 · Patterns",
         "The first release scored each alert on its own. This release looks across alerts and along time: which alerts behave "
         "alike, which behavioural regimes a client moves through, and whether either makes the risk score better.")

if not (cl and sq and en):
    st.info("The pattern stages have not been run yet:  python run_pipeline.py --stage 8 9 10")
    st.stop()

champ = next(r for r in ev["table"] if r["model"] == ev["champion"])
rates = [p["report_rate"] for p in cl["profiles"]["kmeans"]["test"]]
best_seq = max(sq["standalone"], key=lambda r: r["pr_auc"])
gains = [r for r in en["rows"] if r["variant"] != "Baseline" and r["verdict"] == "adds signal"]

c = st.columns(4)
ui.kpi(c[0], "Behavioural profiles", f"{cl['kmeans']['K']}", f"report rate {min(rates):.0%} to {max(rates):.0%} across profiles",
       ui.CYAN, count=cl["kmeans"]["K"])
ui.kpi(c[1], "Transaction regimes", f"{sq['K']}", f"{sq['n_transactions'] / 1e6:.1f}M transactions read in order", ui.PINK, count=sq["K"])
ui.kpi(c[2], "Best sequence score alone", f"{best_seq['pr_auc']:.3f}", f"PR-AUC; random {sq['base_rate_test']:.3f}", ui.ORANGE)
ui.kpi(c[3], "Risk model after enrichment", f"{champ['pr_auc']:.3f}", f"{len(gains)} of 18 combinations improved it", ui.VIOLET)
st.write("")

st.markdown("##### Take the tour")
tour = [
    ("views/overview.py", ":material/dashboard:", "1 · Where we left off", "The queue, the in-service model and the 35% catch rate at a 20% review budget."),
    ("views/clusters.py", ":material/bubble_chart:", "2 · Behavioural profiles",
     "PCA and factor analysis compress 84 features; K-Means, a Gaussian mixture and hierarchical clustering group the alerts."),
    ("views/sequences.py", ":material/timeline:", "3 · Sequence analysis",
     "A hidden Markov model reads each year of transactions in order: regimes, transitions, Viterbi paths, suspicion scores."),
    ("views/enrich.py", ":material/science:", "4 · Feature lab",
     "The six risk models retrained with the new features, compared with paired bootstrap intervals."),
    ("views/alert.py", ":material/person_search:", "5 · One alert, end to end",
     "Score, reasons, behavioural profile and regime path for a single alert."),
]
cols = st.columns(len(tour))
for col, (page, icon, title, text) in zip(cols, tour):
    with col:
        st.markdown(f"<div class='card' style='min-height:150px'><b style='font-size:.98rem'>{title}</b>"
                    f"<div style='color:{ui.MUTED};font-size:.82rem;margin-top:6px'>{text}</div></div>", unsafe_allow_html=True)
        st.page_link(page, label="Open", icon=icon)
st.write("")

a, b = st.columns(2)
with a:
    st.markdown("##### What we found")
    st.markdown(
        f"- **Profiles exist but barely differ in outcome.** Report rates run from {min(rates):.0%} to {max(rates):.0%} against "
        f"{sq['base_rate_test']:.0%} overall; silhouette "
        f"{cl['evaluation']['K-Means']['test']['silhouette']:.2f}.\n"
        f"- **Regimes are long phases.** Clients stay in the same regime {np.mean(np.diag(np.array(sq['A']))):.1%} of the time from one "
        "transaction to the next.\n"
        f"- **Sequences carry some signal on their own** (PR-AUC {best_seq['pr_auc']:.3f} vs {sq['base_rate_test']:.3f} random), "
        "but it overlaps with what the existing features already capture.\n"
        f"- **No combination improved the risk model** beyond its interval. The in-service model stays at PR-AUC {champ['pr_auc']:.3f}.")
with b:
    st.markdown("##### How it was kept fair")
    st.markdown("- Every new model is fitted on 2020 alerts only and applied unchanged to 2021.\n"
                "- The outcome is never used to form profiles or regimes; it is used only to evaluate them.\n"
                "- Outcome-specific sequence models score their own training alerts by cross-fitting.\n"
                "- The comparison reruns the original six models and reproduces their original scores before adding anything.")
