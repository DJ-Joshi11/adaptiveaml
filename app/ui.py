"""Shared look-and-feel and data access for the AdaptiveAML console.

The app never trains or computes a model. It only reads the files written by the pipeline
into artifacts/ - which is what keeps it fast and what will let a real API replace it later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptiveaml.common import REGISTRY, label  # noqa: E402

ART = ROOT / "artifacts"

# ---- palette: near-black surfaces, high-contrast text, one colour per meaning
BG, CARD, LINE, TEXT, MUTED = "#07080B", "#10131A", "#232836", "#F2F4F8", "#9AA3B5"
BLUE, ORANGE, GREEN, VIOLET, YELLOW, RED, CYAN = "#6C8CFF", "#FF7A45", "#3DDC97", "#B58CFF", "#F5C451", "#FF5C7A", "#4FD1E0"
PINK, LIME = "#FF6FB5", "#B6E35A"
REPORT, DISMISS = ORANGE, BLUE                       # reported alerts are always orange, dismissed blue
MODEL_COLORS = {"Logistic Regression": BLUE, "Naive Bayes": YELLOW, "LDA": CYAN,
                "K-Nearest Neighbours": VIOLET, "Decision Tree": GREEN, "SVM (RBF)": ORANGE}
BAND_COLORS = {"Review now": ORANGE, "Standard": BLUE, "Low priority": MUTED}
SERIES = [BLUE, ORANGE, GREEN, VIOLET, YELLOW, CYAN, PINK, LIME, RED, "#8FA1C7", "#E39B5B", "#58B5A8"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@property --n {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes rise {{ from {{ opacity:0; transform: translateY(10px); }} to {{ opacity:1; transform:none; }} }}
@keyframes count {{ from {{ --n: 0; }} to {{ --n: var(--to); }} }}
@keyframes ring {{ from {{ stroke-dashoffset: var(--c); }} }}
@keyframes draw {{ from {{ stroke-dashoffset: 400; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes shimmer {{ 0% {{ background-position: -400px 0; }} 100% {{ background-position: 400px 0; }} }}
@keyframes glow {{ 0%,100% {{ opacity:.55; }} 50% {{ opacity:1; }} }}
html, body, [class*="css"], .stMarkdown, .stDataFrame {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(108,140,255,.10), transparent 60%),
                       radial-gradient(900px 500px at -10% 110%, rgba(255,122,69,.07), transparent 60%), {BG}; }}
.block-container {{ padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1440px; animation: rise .45s ease-out both; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0D1017,#090B10); border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] a {{ border-radius: 10px; transition: background .2s, transform .2s; }}
section[data-testid="stSidebar"] a:hover {{ transform: translateX(3px); background: rgba(108,140,255,.10); }}
section[data-testid="stSidebar"] a[aria-current="page"] {{ background: linear-gradient(90deg, rgba(108,140,255,.25), rgba(181,140,255,.08));
        box-shadow: inset 2px 0 0 {BLUE}; }}
div[data-testid="stSidebarContent"] {{ display:flex; flex-direction:column; }}
div[data-testid="stSidebarUserContent"] {{ order:-1; padding-bottom:0; }}
div[data-testid="stSidebarHeader"] {{ display:none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1 {{ font-weight: 700; letter-spacing: -0.025em; font-size: 1.85rem !important; color:{TEXT};
      background: linear-gradient(90deg,#FFFFFF, #C9D3FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
h2, h3, h4, h5 {{ font-weight: 600; letter-spacing: -0.01em; color:{TEXT}; }}
p, li, label {{ color:#DCE0E8; }}
.brand {{ display:flex; align-items:center; gap:10px; padding: 4px 4px 14px 4px; }}
.brand .dot {{ width:28px; height:28px; border-radius:9px; background: linear-gradient(135deg,{BLUE},{VIOLET} 60%,{ORANGE});
              box-shadow: 0 0 18px rgba(108,140,255,.55); animation: glow 3s ease-in-out infinite; }}
.brand b {{ font-size: 1.05rem; letter-spacing:-0.01em; color:{TEXT}; }} .brand span {{ color:{MUTED}; font-size:.72rem; display:block; }}
.eyebrow {{ color:{BLUE}; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; margin-bottom:2px; font-weight:600; }}
.sub {{ color:{MUTED}; font-size:.93rem; margin-top:-6px; margin-bottom: 16px; }}
.kpi {{ background: linear-gradient(155deg, var(--tint) 0%, {CARD} 55%); border:1px solid {LINE}; border-radius:16px;
        padding:16px 18px; height:100%; position:relative; overflow:hidden; animation: rise .5s ease-out both;
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }}
.kpi:hover {{ transform: translateY(-3px); border-color: var(--edge); box-shadow: 0 10px 30px -12px var(--edge); }}
.kpi .l {{ color:{MUTED}; font-size:.8rem; font-weight:500; padding-right:64px; }}
.kpi .v {{ font-size:1.95rem; font-weight:700; letter-spacing:-0.025em; line-height:1.25; color:#FFFFFF; }}
.kpi .s {{ color:{MUTED}; font-size:.76rem; }}
.kpi .bar {{ position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px; box-shadow: 0 0 12px var(--edge); }}
.kpi .spark {{ position:absolute; right:12px; top:14px; opacity:.8; width:60px; pointer-events:none; }}
.kpi .spark polyline {{ stroke-dasharray: 400; animation: draw 1.4s ease-out both; }}
.kpi .ringwrap {{ position:absolute; right:14px; top:14px; }}
.kpi .ringwrap circle.fg {{ animation: ring 1.3s cubic-bezier(.2,.8,.2,1) both; }}
.cnt {{ animation: count 1.2s cubic-bezier(.2,.8,.2,1) forwards; counter-reset: n var(--n); }}
.cnt::after {{ content: counter(n); }}
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:16px 18px; animation: rise .5s ease-out both;
         transition: border-color .25s, box-shadow .25s; }}
.card:hover {{ border-color:#34405A; box-shadow: 0 8px 26px -14px rgba(108,140,255,.6); }}
.note {{ background: linear-gradient(90deg, rgba(108,140,255,.08), {CARD} 40%); border:1px solid {LINE}; border-left:3px solid {BLUE};
         border-radius:12px; padding:10px 14px; color:#D5DAE4; font-size:.88rem; margin: 6px 0 14px 0; animation: rise .5s ease-out both; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:600; border:1px solid; }}
.warn {{ background:#2A1B10; border:1px solid #6B3A16; color:#FFB58A; border-radius:10px; padding:9px 14px; font-size:.86rem; margin-bottom:14px; }}
.take {{ background: linear-gradient(120deg, rgba(61,220,151,.10), {CARD} 45%); border:1px solid rgba(61,220,151,.35);
         border-radius:16px; padding:14px 18px 10px; margin: 4px 0 16px; animation: rise .5s ease-out both; }}
.take .t {{ color:{GREEN}; font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; margin-bottom:6px; }}
.take ul {{ margin:0; padding-left:18px; }} .take li {{ color:#E6EAF1; font-size:.92rem; margin:3px 0; line-height:1.45; }}
.take b {{ color:#FFFFFF; }}
.skel {{ border-radius:16px; border:1px solid {LINE}; background: linear-gradient(90deg,#10131A 0%,#1A1F2B 50%,#10131A 100%);
         background-size: 800px 100%; animation: shimmer 1.2s linear infinite; }}
div[data-testid="stPlotlyChart"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:8px 8px 0 8px;
        animation: rise .55s ease-out both; transition: border-color .25s, box-shadow .25s; }}
div[data-testid="stPlotlyChart"]:hover {{ border-color:#34405A; box-shadow: 0 10px 30px -16px rgba(108,140,255,.55); }}
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px; overflow:hidden; animation: rise .55s ease-out both; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; color:{MUTED}; border-radius:10px 10px 0 0; transition: color .2s, background .2s; }}
.stTabs [data-baseweb="tab"]:hover {{ color:{TEXT}; background: rgba(108,140,255,.08); }}
.stTabs [aria-selected="true"] {{ color:{TEXT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg,{BLUE},{VIOLET}); height:3px; }}
.stTabs [data-baseweb="tab-panel"] {{ animation: rise .4s ease-out both; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ border-radius:10px; transition: border-color .2s, box-shadow .2s; }}
div[data-baseweb="select"] > div:hover {{ border-color:{BLUE}; box-shadow: 0 0 0 3px rgba(108,140,255,.15); }}
div[data-testid="stSlider"] div[role="slider"] {{ box-shadow: 0 0 0 6px rgba(108,140,255,.18); transition: box-shadow .2s, transform .15s; }}
div[data-testid="stSlider"] div[role="slider"]:hover {{ transform: scale(1.15); box-shadow: 0 0 0 9px rgba(108,140,255,.22); }}
code, pre {{ font-family:'JetBrains Mono', monospace !important; font-size:.8rem !important; }}
</style>
"""


def setup(title: str, eyebrow: str, sub: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if evaluation().get("data_source") == "fixture":
        st.markdown('<div class="warn"><b>Test fixture loaded.</b> These numbers come from a small generated stand-in, '
                    'not from SynthAML. Run <code>python run_pipeline.py</code> on the real files before presenting.</div>',
                    unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if sub:
        st.markdown(f'<div class="sub">{sub}</div>', unsafe_allow_html=True)


def _spark(values, color: str) -> str:
    v = [float(x) for x in values]
    lo, hi = min(v), max(v)
    pts = " ".join(f"{i * 90 / max(len(v) - 1, 1):.1f},{26 - (x - lo) / ((hi - lo) or 1) * 22:.1f}" for i, x in enumerate(v))
    return (f'<svg class="spark" width="92" height="28" viewBox="0 0 92 28"><polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _ring(frac: float, color: str) -> str:
    r, c = 17, 2 * 3.14159 * 17
    off = c * (1 - max(0.0, min(1.0, frac)))
    return (f'<div class="ringwrap"><svg width="44" height="44" viewBox="0 0 44 44"><circle cx="22" cy="22" r="{r}" fill="none" '
            f'stroke="{LINE}" stroke-width="5"/><circle class="fg" cx="22" cy="22" r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="5" stroke-linecap="round" stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" '
            f'style="--c:{c:.1f}" transform="rotate(-90 22 22)"/></svg></div>')


def kpi(col, label_: str, value: str, sub: str = "", color: str = BLUE, count: int | None = None,
        suffix: str = "", spark=None, ring: float | None = None) -> None:
    """A metric card. `count` animates the number up from zero; `spark` adds a sparkline;
    `ring` (0..1) adds an animated progress ring."""
    shown = f'<span class="cnt" style="--to:{int(count)}"></span>{suffix}' if count is not None else value
    extra = (_spark(spark, color) if spark is not None and len(spark) > 1 else "") + (_ring(ring, color) if ring is not None else "")
    vstyle = ' style="font-size:1.2rem;line-height:1.35;padding:4px 0"' if count is None and len(value) > 14 else ""
    col.markdown(f'<div class="kpi" style="--tint:{rgba(color, .13)};--edge:{rgba(color, .55)}"><div class="bar" style="background:{color}"></div>'
                 f'<div class="l">{label_}</div><div class="v"{vstyle}>{shown}</div><div class="s">{sub}</div>{extra}</div>',
                 unsafe_allow_html=True)


def takeaway(points: list[str], title: str = "What this tells us") -> None:
    """The plain-language conclusion of a section, shown before the detail."""
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(f'<div class="take"><div class="t">✦ {title}</div><ul>{items}</ul></div>', unsafe_allow_html=True)


def plain(text: str, title: str = "New to this? The idea in plain words") -> None:
    """A collapsed explanation for readers who do not know the method."""
    with st.expander(title, icon=":material/lightbulb:"):
        st.markdown(text)


def skeleton(height: int = 360):
    """Shimmering placeholder shown while a heavy chart is being built; returns the slot to fill."""
    slot = st.empty()
    slot.markdown(f'<div class="skel" style="height:{height}px"></div>', unsafe_allow_html=True)
    return slot


def note(text: str, color: str = BLUE) -> None:
    st.markdown(f'<div class="note" style="border-left-color:{color}">{text}</div>', unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="color:{color};border-color:{color}55;background:{color}14">{text}</span>'


def fig_layout(fig: go.Figure, title: str = "", height: int = 360, **kw) -> go.Figure:
    margin = kw.pop("margin", dict(l=50, r=20, t=48 if title else 20, b=44))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01, xanchor="left"),
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12),
        margin=margin,
        legend=dict(orientation="h", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color="#C9CDD6")),
        hoverlabel=dict(bgcolor="#161A24", bordercolor="#34405A", font=dict(color=TEXT, family="Inter, sans-serif")),
        transition=dict(duration=400, easing="cubic-in-out"), **kw)
    fig.update_xaxes(gridcolor="#171B25", zerolinecolor=LINE, linecolor=LINE, tickcolor=LINE, tickfont=dict(color="#B4BCCB"))
    fig.update_yaxes(gridcolor="#171B25", zerolinecolor=LINE, linecolor=LINE, tickcolor=LINE, tickfont=dict(color="#B4BCCB"))
    return fig


def show(fig: go.Figure, explore: bool = False, slot=None, **kw):
    """Render a chart. `explore=True` keeps the zoom / pan / lasso toolbar for charts meant to be explored."""
    if explore:
        cfg = {"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["toImage"]}
    else:
        # read-only charts: hover works, but dragging no longer zooms into a box
        cfg = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}
        fig.update_layout(dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    target = slot if slot is not None else st
    return target.plotly_chart(fig, width="stretch", config=cfg, **kw)


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ---- artifact loaders (cached: the files only change when the pipeline is re-run)
@st.cache_data
def _json(name: str) -> dict:
    path = ART / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluation() -> dict: return _json("evaluation.json")
def eda() -> dict: return _json("eda.json")
def regression() -> dict: return _json("regression.json")
def classification() -> dict: return _json("classification.json")
def explain() -> dict: return _json("explain.json")
def quality() -> dict: return _json("data_quality.json")
def clustering() -> dict: return _json("clustering.json")
def sequences() -> dict: return _json("sequences.json")
def enrichment() -> dict: return _json("enrichment.json")


def has(name: str) -> bool:
    return (ART / name).exists()


def state_label(k: int, states: list[dict]) -> str:
    return f"R{k + 1} · {states[k]['name']}" if k < len(states) else f"R{k + 1}"


def cluster_label(k: int, profiles: list[dict]) -> str:
    p = next((x for x in profiles if x["cluster"] == k), None)
    return f"P{k + 1}" + (f" · {p['name']}" if p and p["name"] else "")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    path = ART / name if (ART / name).exists() else ROOT / "data" / "processed" / name
    return pd.read_parquet(path)


def require_artifacts() -> None:
    if not (ART / "evaluation.json").exists():
        st.markdown(CSS, unsafe_allow_html=True)
        st.title("AdaptiveAML")
        st.error("No results found. Run the pipeline first:  python run_pipeline.py")
        st.stop()


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def ref_line(fig: go.Figure, x: float, text: str = "no skill", color: str = RED) -> None:
    """Vertical reference line with its label above the plot area (never on top of the bars)."""
    fig.add_vline(x=x, line=dict(color=color, dash="dash", width=1.5))
    fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom", text=text, showarrow=False, font=dict(color=color, size=11))
