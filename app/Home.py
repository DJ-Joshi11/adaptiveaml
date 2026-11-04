"""AdaptiveAML console - entry point.

    streamlit run app/Home.py
"""
import streamlit as st

st.set_page_config(page_title="AdaptiveAML", page_icon=":material/shield:", layout="wide")

from ui import require_artifacts  # noqa: E402

require_artifacts()

pages = {
    "Start": [
        st.Page("views/whatsnew.py", title="What's New", icon=":material/auto_awesome:", default=True),
    ],
    "Operations": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:"),
        st.Page("views/queue.py", title="Alert Queue", icon=":material/notifications_active:"),
        st.Page("views/alert.py", title="Alert Detail", icon=":material/person_search:"),
    ],
    "Model": [
        st.Page("views/models.py", title="Model Comparison", icon=":material/leaderboard:"),
        st.Page("views/enrich.py", title="Feature Lab", icon=":material/science:"),
        st.Page("views/benchmark.py", title="Benchmark", icon=":material/compare_arrows:"),
        st.Page("views/reasoning.py", title="Model Reasoning", icon=":material/psychology:"),
        st.Page("views/baseline.py", title="Activity Baseline", icon=":material/show_chart:"),
    ],
    "Patterns": [
        st.Page("views/clusters.py", title="Behavioural Profiles", icon=":material/bubble_chart:"),
        st.Page("views/sequences.py", title="Sequence Analysis", icon=":material/timeline:"),
    ],
    "Data": [
        st.Page("views/insights.py", title="Data Insights", icon=":material/insights:"),
    ],
}
with st.sidebar:
    st.markdown('<div class="brand"><div class="dot"></div><div><b>AdaptiveAML</b>'
                '<span>Alert triage console</span></div></div>', unsafe_allow_html=True)
st.navigation(pages).run()
