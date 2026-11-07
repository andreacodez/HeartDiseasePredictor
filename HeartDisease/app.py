import streamlit as st
import streamlit.components.v1 as components
import textwrap
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from xgboost import XGBClassifier


# ============================================================
# Streamlit Configuration
# ============================================================
st.set_page_config(
    page_title="Heart Disease Status Predictor",
    page_icon="❤️",
    layout="wide"
)

MODEL_DIR = Path(__file__).parent / "models"

# ============================================================
# Dashboard Theme (Visualization Analysis tab only)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* ---------------------------------------------------------- */
/* Model Used selector — match the dark EDA "View Information" */
/* button/card treatment instead of the default white input.  */
/* ---------------------------------------------------------- */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #1e2a47 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-radius: 10px !important;
    min-height: 46px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    background: #243252 !important;
    border-color: rgba(148, 163, 184, 0.38) !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #f8fafc !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #e2e8f0 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="popover"] {
    background: #1e2a47 !important;
}

ul[role="listbox"] {
    background: #1e2a47 !important;
}

ul[role="listbox"] li {
    color: #e2e8f0 !important;
}

ul[role="listbox"] li:hover {
    background: #2a395b !important;
}

/* ---------------------------------------------------------- */
/* Group "Recommended Ranges" info popover — smaller box       */
/* ---------------------------------------------------------- */
div[data-testid="stPopoverBody"] {
    max-width: 260px !important;
    width: 260px !important;
    padding: 10px 12px !important;
}
div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] strong,
div[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] {
    font-size: 0.8rem !important;
    line-height: 1.3 !important;
}
div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --hd-bg: #05060f;
    --hd-bg-2: #0a0e1f;
    --hd-surface: rgba(35, 42, 74, 0.62);
    --hd-surface-solid: #1a2040;
    --hd-border: rgba(148, 163, 184, 0.14);
    --hd-border-strong: rgba(148, 163, 184, 0.24);

    --hd-accent: #ec4899;         /* magenta / pink */
    --hd-accent-2: #22d3ee;       /* cyan */
    --hd-accent-3: #a78bfa;       /* violet */
    --hd-accent-bright: #f9a8d4;
    --hd-accent-dark: #9d174d;
    --hd-accent-light: rgba(236, 72, 153, 0.14);
    --hd-gradient: linear-gradient(120deg, #ec4899 0%, #a855f7 50%, #22d3ee 100%);

    --hd-risk-high: #fb7185;
    --hd-risk-moderate: #fbbf24;
    --hd-risk-low: #34d399;

    --hd-text: #eef2f9;
    --hd-text-dim: #94a3b8;
    --hd-text-faint: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---------------------------------------------------------- */
/* Global app background — dark, holographic-medical mood     */
/* ---------------------------------------------------------- */
.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(236, 72, 153, 0.10) 0%, transparent 42%),
        radial-gradient(circle at 88% 12%, rgba(34, 211, 238, 0.10) 0%, transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(168, 85, 247, 0.08) 0%, transparent 50%),
        #1e2a47;
    color: var(--hd-text);
}
[data-testid="stHeader"] {
    background: rgba(5, 6, 15, 0.0);
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    font-family: 'Sora', 'Inter', sans-serif;
    color: #f8fafc;
    letter-spacing: -0.01em;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stMarkdownContainer"] {
    color: var(--hd-text);
}
[data-testid="stCaptionContainer"] {
    color: var(--hd-text-dim) !important;
}
hr {
    border-color: var(--hd-border) !important;
}

/* Cards for every Plotly chart */
div[data-testid="stPlotlyChart"] {
    background: var(--hd-surface);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 18px;
    border: 1px solid var(--hd-border);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), 0 0 0 1px rgba(236,72,153,0.03) inset;
    padding: 16px 12px 8px 12px;
}

/* Cards for native st.metric widgets */
div[data-testid="stMetric"] {
    background: var(--hd-surface);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 18px;
    border: 1px solid var(--hd-border);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    padding: 16px 18px;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"] {
    color: var(--hd-text-dim) !important;
}
div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* Bordered st.container()s used as chart/section cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid var(--hd-border) !important;
    background: var(--hd-surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.30);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(236, 72, 153, 0.35) !important;
    box-shadow: 0 12px 32px rgba(0,0,0,0.4), 0 0 24px rgba(236,72,153,0.10);
    transform: translateY(-2px);
}

/* Make every bordered container the same height as its tallest
   sibling within a row of columns, so card rows line up evenly. */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* ---------------------------------------------------------- */
/* EDA "Explore the Data" navigation cards                  */
/* Keep the three cards identical in height and push their    */
/* action buttons to the bottom for a clean, aligned row.     */
/* ---------------------------------------------------------- */
[class*="st-key-eda_nav_card_"] {
    min-height: 365px !important;
    height: 365px !important;
    box-sizing: border-box !important;
}

[class*="st-key-eda_nav_card_"] > div {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}

[class*="st-key-eda_nav_card_"] div[data-testid="stVerticalBlock"] {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Visible dark button instead of Streamlit's white default. */
[class*="st-key-eda_nav_card_"] div[data-testid="stButton"] button {
    margin-top: auto !important;
    min-height: 48px !important;
    background: rgba(30, 41, 70, 0.95) !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22) !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
}

[class*="st-key-eda_nav_card_"] div[data-testid="stButton"] button p,
[class*="st-key-eda_nav_card_"] div[data-testid="stButton"] button span {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
}

[class*="st-key-eda_nav_card_"] div[data-testid="stButton"] button:hover {
    background: rgba(236, 72, 153, 0.16) !important;
    color: #f9a8d4 !important;
    -webkit-text-fill-color: #f9a8d4 !important;
    border-color: rgba(236, 72, 153, 0.48) !important;
    box-shadow: 0 8px 24px rgba(236, 72, 153, 0.16) !important;
    transform: translateY(-1px);
}

[class*="st-key-eda_nav_card_"] div[data-testid="stButton"] button:hover p,
[class*="st-key-eda_nav_card_"] div[data-testid="stButton"] button:hover span {
    color: #f9a8d4 !important;
    -webkit-text-fill-color: #f9a8d4 !important;
}

/* Pill-style tabs */
div[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--hd-border) !important;
}
button[data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
    color: var(--hd-text-dim);
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--hd-accent-light);
    color: var(--hd-accent-bright);
}
div[data-baseweb="tab-highlight"] {
    background: var(--hd-gradient) !important;
    height: 3px !important;
}

/* Custom KPI cards */
.hd-kpi-card {
    background: var(--hd-surface);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 18px;
    border: 1px solid var(--hd-border);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    padding: 16px 18px;
    height: 100%;
}
.hd-kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--hd-text-dim);
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 6px;
}
.hd-kpi-value {
    font-family: 'Sora', sans-serif;
    font-size: 28px;
    font-weight: 750;
    color: #f8fafc;
    line-height: 1.1;
}
.hd-kpi-value.accent {
    background: var(--hd-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hd-kpi-icon {
    flex: 0 0 auto;
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.hd-kpi-sub {
    font-size: 12px;
    color: var(--hd-text-faint);
    margin-top: 2px;
}

/* Dashboard header banner */
.hd-dashboard-header {
    background: #A84139;
    padding: 24px 28px;
    border-radius: 18px;
    margin-bottom: 18px;
    box-shadow: 0 10px 34px rgba(168, 85, 247, 0.28), 0 0 60px rgba(34,211,238,0.10);
    position: relative;
    overflow: hidden;
}
.hd-dashboard-header::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 85% 20%, rgba(255,255,255,0.22) 0%, transparent 45%);
    pointer-events: none;
}
.hd-dashboard-header h2 {
    font-family: 'Sora', sans-serif;
    color: #FFFFFF;
    margin: 0;
    font-size: 25px;
}
.hd-dashboard-header p {
    color: rgba(255,255,255,0.88);
    margin: 4px 0 0 0;
    font-size: 13px;
}

/* Header icon badge (glass square inside the banner) */
.hd-header-row {
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    z-index: 1;
}
.hd-header-icon {
    flex: 0 0 auto;
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(6px);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e2a47;
    border-right: 1px solid var(--hd-border);
}
.hd-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 2px 10px 2px;
}
.hd-sidebar-brand-icon {
    font-size: 22px;
    filter: drop-shadow(0 0 8px rgba(236,72,153,0.6));
}
.hd-sidebar-brand-text {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 15px;
    background: var(--hd-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    text-align: center;
    justify-content: center;
    border-radius: 10px;
    border: 1px solid transparent;
    font-weight: 500;
}

/* Horizontal navigation bar — full content width */
.hd-nav-bar {
    width: 100%;
    box-sizing: border-box;
    background: rgba(20, 30, 52, 0.92);
    border: 1px solid var(--hd-border);
    border-radius: 14px;
    padding: 6px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28);
}

/* Sidebar navigation — vertical and left aligned */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    min-height: 46px;
    padding: 8px 12px !important;
    font-size: 13px;
    line-height: 1.2;
    white-space: nowrap;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 10px;
}

/* Streamlit's button content is a flex container. Force the icon + label
   to start from exactly the same left position on every sidebar item. */
section[data-testid="stSidebar"] div[data-testid="stButton"] button > div {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button > div > div {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button [data-testid="stMarkdownContainer"] p {
    text-align: left !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    color: var(--hd-text-dim) !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.04) !important;
    color: var(--hd-text) !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: var(--hd-accent-light) !important;
    color: var(--hd-accent-bright) !important;
    border: 1px solid rgba(236,72,153,0.35) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
    background: transparent;
    color: var(--hd-text-dim);
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.04);
    color: var(--hd-text);
    border-color: var(--hd-border);
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: var(--hd-accent-light);
    color: var(--hd-accent-bright);
    border: 1px solid rgba(236,72,153,0.35);
    box-shadow: 0 0 18px rgba(236,72,153,0.12);
}
.hd-sidebar-about {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--hd-border);
    border-radius: 14px;
    padding: 14px 16px;
}
.hd-sidebar-about-title {
    font-weight: 700;
    font-size: 13px;
    color: var(--hd-text);
    margin-bottom: 6px;
}
.hd-sidebar-about-body {
    font-size: 12.5px;
    color: var(--hd-text-dim);
    line-height: 1.5;
}

/* Risk Scorer factor-field cards */
.hd-field-head {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 6px;
}
.hd-field-icon {
    flex: 0 0 auto;
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}
.hd-field-title {
    font-weight: 650;
    font-size: 14px;
    color: var(--hd-text);
}
.hd-field-desc {
    font-size: 11.5px;
    color: #ffffff;
    line-height: 1.3;
}

/* Project Functions cards (Background Overview page) — force every
   card in the 5-column row to the same height regardless of how much
   its title/description text wraps. */
[class*="st-key-obj_card_"] {
    min-height: 150px !important;
    height: 150px !important;
    box-sizing: border-box !important;
}
[class*="st-key-obj_card_"] > div {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}
[class*="st-key-obj_card_"] div[data-testid="stVerticalBlock"] {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}


/* Risk factor checklist card (right rail, next to patient figure) */
.hd-checklist-card {
border-radius: 18px;
    border: 1px solid var(--hd-border);
    background: var(--hd-surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.32);
    padding: 14px 14px 12px 14px;
    height: 100%;
}
.hd-checklist-header {
    position:absolute;
    left:0;
    top:72px;
    width:100%;
    box-sizing:border-box;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 13.5px;
    color: var(--hd-accent-bright);
    margin-bottom: 0;
    letter-spacing: .01em;
}
.hd-checklist-row {
    display: flex;
    align-items: center;
    gap: 10px;
    border: 1px solid var(--hd-border);
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 8px 10px;
    margin-bottom: 8px;
}
.hd-checklist-icon {
    flex: 0 0 auto;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    background: rgba(255,255,255,0.05);
}
.hd-checklist-text {
    flex: 1 1 auto;
    min-width: 0;
}
.hd-checklist-title {
    font-weight: 650;
    font-size: 12px;
    color: var(--hd-text);
    line-height: 1.25;
}
.hd-checklist-desc {
    font-size: 10px;
    color: var(--hd-text-faint);
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hd-checklist-dot {
    flex: 0 0 auto;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    box-sizing: border-box;
}
.hd-checklist-dot.pending {
    border: 2px solid var(--hd-accent);
    background: transparent;
}
.hd-checklist-dot.healthy {
    border: 2px solid #34d399;
    background: rgba(52, 211, 153, 0.30);
    box-shadow: 0 0 10px rgba(52, 211, 153, 0.55);
}
.hd-checklist-dot.warning {
    border: 2px solid #fbbf24;
    background: rgba(251, 191, 36, 0.30);
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.55);
}
.hd-checklist-dot.danger {
    border: 2px solid #fb7185;
    background: rgba(251, 113, 133, 0.30);
    box-shadow: 0 0 10px rgba(251, 113, 133, 0.55);
}
.hd-checklist-legend {
    display: flex;
    gap: 16px;
    justify-content: center;
    margin-top: 4px;
    font-size: 10.5px;
    color: var(--hd-text-dim);
}
.hd-checklist-legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.hd-checklist-legend .hd-checklist-dot {
    position: static;
}


/* Risk Factors Status is now embedded inside the holographic patient card. */
.hd-checklist-inside {
    position:absolute;
    left:16px;
    top:0;
    width:42%;
    max-width:245px;
    height:100%;
    z-index:7;
    transform:none;
    box-sizing:border-box;
    padding-top:30px;
}
/* Rows are positioned as a % of the SAME box the callout lines and body
   hotspots use (.hd-holo-card, height:600px) so a row's vertical center
   lands exactly on its matching line/dot, no matter how tall the card is.
   (top:X% - 25px) centers the 50px-tall row on the X% mark, since
   percentage offsets for absolutely-positioned elements are always
   resolved against the containing block's full height, ignoring this
   element's own padding-top above. */
.hd-checklist-inside .hd-checklist-header {
    position:absolute;
    left:0;
    top:calc(21% - 55px);
    margin:0;
}
.hd-checklist-inside .hd-checklist-row {
    position:absolute;
    left:0;
    width:100%;
    height:50px;
    min-height:50px;
    margin:0;
    padding:7px 8px;
    gap:7px;
    box-sizing:border-box;
}
.hd-checklist-inside .hd-checklist-row:nth-of-type(1) { top:calc(21% - 25px); }
.hd-checklist-inside .hd-checklist-row:nth-of-type(2) { top:calc(31% - 25px); }
.hd-checklist-inside .hd-checklist-row:nth-of-type(3) { top:calc(41% - 25px); }
.hd-checklist-inside .hd-checklist-row:nth-of-type(4) { top:calc(51% - 25px); }
.hd-checklist-inside .hd-checklist-row:nth-of-type(5) { top:calc(61% - 25px); }
.hd-checklist-inside .hd-checklist-icon {
    width:26px;
    height:26px;
    font-size:13px;
}
.hd-checklist-inside .hd-checklist-title {
    font-size:10px;
}
.hd-checklist-inside .hd-checklist-desc {
    font-size:8px;
}
.hd-checklist-inside .hd-checklist-legend {
    position:absolute;
    left:0;
    width:100%;
    top:calc(51% + 32px);
    font-size:8px;
    display:grid;
    grid-template-columns: auto auto;
    column-gap:14px;
    row-gap:6px;
    justify-content:start;
    justify-items:start;
}

/* Prediction summary panel */
.hd-risk-badge {
    border-radius: 14px;
    padding: 14px 16px;
    margin: 10px 0;
    border: 1px solid var(--hd-border);
}
.hd-risk-badge.high { background: rgba(251, 113, 133, 0.10); border-color: rgba(251,113,133,0.30); }
.hd-risk-badge.moderate { background: rgba(251, 191, 36, 0.10); border-color: rgba(251,191,36,0.30); }
.hd-risk-badge.low { background: rgba(52, 211, 153, 0.10); border-color: rgba(52,211,153,0.30); }
.hd-risk-badge-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: var(--hd-text-dim);
    margin-bottom: 4px;
}
.hd-risk-badge-value {
    font-family: 'Sora', sans-serif;
    font-size: 20px;
    font-weight: 750;
}
.hd-risk-badge-value.high { color: var(--hd-risk-high); }
.hd-risk-badge-value.moderate { color: var(--hd-risk-moderate); }
.hd-risk-badge-value.low { color: var(--hd-risk-low); }
.hd-info-box {
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.5;
    margin: 10px 0;
    border: 1px solid var(--hd-border);
}
.hd-info-box.interpretation { background: rgba(251, 191, 36, 0.08); color: #fde68a; }
.hd-info-box.whatsnext { background: rgba(52, 211, 153, 0.08); color: #a7f3d0; }
.hd-info-box.whatsnext ul { margin: 6px 0 0 18px; padding: 0; }

/* Bot-style chat message for risk interpretation & recommendations */
.hd-bot-wrap {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin: 18px 0 10px 0;
}
.hd-bot-avatar {
    flex-shrink: 0;
    width: 68px;
    height: 68px;
    border-radius: 50%;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at 50% 40%, #16223d 0%, #0a1226 70%, #060a16 100%);
    box-shadow:
        0 0 0 1px rgba(96,165,250,0.20),
        0 0 22px rgba(59,130,246,0.35),
        0 0 44px rgba(139,92,246,0.20),
        inset 0 0 14px rgba(0,0,0,0.55);
}
.hd-bot-avatar::before {
    content: "";
    position: absolute;
    inset: -3px;
    border-radius: 50%;
    padding: 2px;
    background: conic-gradient(
        from 200deg,
        #22d3ee 0%,
        #60a5fa 30%,
        #a78bfa 55%,
        #60a5fa 75%,
        #22d3ee 100%
    );
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.9;
}
.hd-bot-avatar img {
    width: 74%;
    height: 74%;
    object-fit: contain;
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 0 6px rgba(96,165,250,0.35));
}
.hd-bot-bubbles {
    display: flex;
    flex-direction: column;
    gap: 12px;
    flex: 1;
    min-width: 0;
}
.hd-bot-bubble {
    position: relative;
    background: linear-gradient(155deg, rgba(30,41,68,0.85) 0%, rgba(14,20,38,0.92) 100%);
    color: var(--hd-text);
    border-radius: 18px;
    padding: 15px 19px;
    font-size: 13.5px;
    line-height: 1.65;
    box-shadow:
        0 8px 24px rgba(3,7,18,0.45),
        0 0 24px rgba(59,130,246,0.10);
}
.hd-bot-bubble::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 18px;
    padding: 1.4px;
    background: linear-gradient(120deg, #22d3ee 0%, #60a5fa 45%, #a78bfa 100%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.55;
    pointer-events: none;
}
.hd-bot-bubble b { font-weight: 750; }
.hd-bot-bubble ul {
    list-style: none;
    margin: 8px 0 0 0;
    padding: 0;
}
.hd-bot-bubble li {
    margin: 6px 0;
    display: flex;
    gap: 8px;
    align-items: flex-start;
}

/* Section titles inside the two-column grouped form */
.hd-section-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 15px;
    color: var(--hd-accent-bright);
    margin: 18px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--hd-accent-light);
}

/* Horizontal pill-style radio (used elsewhere for multi-option groups) */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    flex-direction: row;
    gap: 8px;
}
div[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--hd-border);
    border-radius: 10px;
    padding: 4px 16px;
    margin: 0;
}

/* Risk-aware holographic patient figure */
.hd-body-figure-title { font-family:'Sora',sans-serif; font-weight:750; font-size:16px; color:#f8fafc; margin:4px 0 10px; }
.hd-body-status { font-size:12px; font-weight:700; margin-left:8px; }
.hd-holo-card { position:relative; height:600px; min-height:600px; overflow:visible; margin-bottom:28px; border-radius:24px; border:1px solid rgba(34,211,238,.24); background:radial-gradient(circle at 50% 42%,rgba(34,211,238,.09),transparent 38%),radial-gradient(circle at 50% 80%,rgba(167,139,250,.10),transparent 32%),rgba(3,10,28,.62); box-shadow:0 18px 45px rgba(0,0,0,.42),inset 0 0 50px rgba(34,211,238,.04); }
.hd-holo-grid { position:absolute; inset:0; opacity:.30; background-image:linear-gradient(rgba(34,211,238,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.07) 1px,transparent 1px); background-size:32px 32px; mask-image:linear-gradient(to bottom,transparent,black 16%,black 88%,transparent); }
/* Confined to the body-figure side of the card only — starts right at
   the checklist card's edge (same boundary the callout lines use) so it
   doesn't sweep over the four risk-factor containers on the left. */
.hd-scanline { position:absolute; left:min(calc(16px + 42%), 261px); right:8%; top:15%; height:2px; background:linear-gradient(90deg,transparent,rgba(34,211,238,.7),rgba(236,72,153,.6),transparent); animation:hd-scan 3.8s ease-in-out infinite; z-index:4; }
@keyframes hd-scan { 0%,100%{transform:translateY(0);opacity:.08} 50%{transform:translateY(400px);opacity:.8} }
.hd-holo-figure { position:absolute; width:64%; max-width:300px; height:auto; left:70%; top:50%; transform:translate(-50%,-50%); filter:drop-shadow(0 0 22px rgba(34,211,238,.34)); z-index:2; }
.hd-holo-outline { fill:url(#bodyFill); stroke:url(#bodyStroke); stroke-width:2.1; vector-effect:non-scaling-stroke; }
.hd-heart-glow { animation:hd-heart-glow 1.2s ease-in-out infinite; transform-origin:180px 240px; }
@keyframes hd-heart-glow { 0%,100%{opacity:.76;filter:url(#glowPink)} 50%{opacity:1;filter:url(#glowPinkStrong)} }
.hd-hud-ring { fill:none; stroke:rgba(34,211,238,.22); stroke-width:1; }
.hd-hud-ring.dash { stroke:rgba(34,211,238,.48); stroke-dasharray:3 8; animation:hd-rotate 14s linear infinite; transform-origin:180px 315px; }
@keyframes hd-rotate { to{transform:rotate(360deg)} }
.hd-hotspot { position:absolute; transform:translate(-50%,-50%); z-index:10; pointer-events:auto; }
/* Spans exactly from the checklist card's right edge (left:16px + 42%
   width, matching .hd-checklist-inside) to the body hotspot dot
   (left:70%, matching .hd-hotspot), so the line always touches both
   dots regardless of the card's actual rendered width. */
.hd-callout-line { position:absolute; left:min(calc(16px + 42%), 261px); width:calc(70% - min(calc(16px + 42%), 261px)); height:1px; background:linear-gradient(90deg,rgba(34,211,238,.75),rgba(34,211,238,.4),rgba(34,211,238,.85)); transform-origin:left center; z-index:6; pointer-events:none; opacity:.85; }
.hd-callout-line::before { content:""; position:absolute; left:-3px; top:-3px; width:6px; height:6px; border-radius:50%; background:#22d3ee; box-shadow:0 0 8px rgba(34,211,238,.8); }
.hd-callout-line::after { content:""; position:absolute; right:-3px; top:-3px; width:6px; height:6px; border-radius:50%; background:#22d3ee; box-shadow:0 0 8px rgba(34,211,238,.8); }
.hd-callout-line.line-1 { top:31%; }
.hd-callout-line.line-2 { top:41%; }
.hd-callout-line.line-3 { top:51%; }
.hd-hotspot-dot { display:block; width:12px; height:12px; border-radius:50%; background:#fb7185; border:2px solid #070b19; box-shadow:0 0 0 4px rgba(251,113,133,.20),0 0 20px rgba(251,113,133,.80); cursor:pointer; animation:hd-marker 1.7s ease-in-out infinite; }
.hd-hotspot-dot.danger { background:#fb7185; box-shadow:0 0 0 4px rgba(251,113,133,.20),0 0 20px rgba(251,113,133,.80); animation:hd-marker 1.7s ease-in-out infinite; }
.hd-hotspot-dot.warning { background:#fbbf24; box-shadow:0 0 0 4px rgba(251,191,36,.20),0 0 20px rgba(251,191,36,.80); animation:hd-marker-yellow 1.7s ease-in-out infinite; }
.hd-hotspot-dot.healthy { background:#34d399; box-shadow:0 0 0 4px rgba(52,211,153,.20),0 0 20px rgba(52,211,153,.80); animation:hd-marker-green 1.7s ease-in-out infinite; }
@keyframes hd-marker { 0%,100%{box-shadow:0 0 0 3px rgba(251,113,133,.18),0 0 12px rgba(251,113,133,.55)} 50%{box-shadow:0 0 0 8px rgba(251,113,133,.06),0 0 28px rgba(251,113,133,.95)} }
@keyframes hd-marker-yellow { 0%,100%{box-shadow:0 0 0 3px rgba(251,191,36,.18),0 0 12px rgba(251,191,36,.55)} 50%{box-shadow:0 0 0 8px rgba(251,191,36,.06),0 0 28px rgba(251,191,36,.95)} }
@keyframes hd-marker-green { 0%,100%{box-shadow:0 0 0 3px rgba(52,211,153,.18),0 0 12px rgba(52,211,153,.55)} 50%{box-shadow:0 0 0 8px rgba(52,211,153,.06),0 0 28px rgba(52,211,153,.95)} }
.hd-tooltip { visibility:hidden; opacity:0; position:absolute; left:22px; top:-10px; min-width:205px; max-width:245px; background:rgba(10,14,31,.97); color:#eef2f9; border:1px solid rgba(34,211,238,.28); font-size:10.5px; line-height:1.45; padding:9px 11px; border-radius:11px; box-shadow:0 14px 34px rgba(0,0,0,.55),0 0 20px rgba(34,211,238,.08); transition:opacity .16s ease; z-index:20; }
.hd-hotspot:hover .hd-tooltip { visibility:visible; opacity:1; }
.hd-body-hint { position:absolute; bottom:12px; left:0; right:0; text-align:center; font-size:10.5px; color:#64748b; z-index:9; }
.hd-holo-label { position:absolute; top:15px; left:18px; font-size:10px; letter-spacing:.13em; text-transform:uppercase; color:#38bdf8; opacity:.82; z-index:5; }
.hd-holo-status { position:absolute; top:15px; right:18px; font-size:10px; color:#94a3b8; z-index:5; }
</style>
""", unsafe_allow_html=True)


def kpi_card(
    label,
    value,
    subtitle=None,
    icon=None,
    badge_bg="rgba(236,72,153,0.14)",
    badge_color="#f9a8d4",
    accent=False,
):
    """Render a Power-BI-style KPI card using native Streamlit elements."""
    with st.container(border=True):
        icon_col, content_col = st.columns([0.20, 0.80], gap="small")

        with icon_col:
            if icon:
                st.markdown(f"### {icon}")

        with content_col:
            st.caption(label)
            if accent:
                st.markdown(
                    f"**<span style='background:var(--hd-gradient);"
                    f"-webkit-background-clip:text;background-clip:text;"
                    f"-webkit-text-fill-color:transparent;'>{value}</span>**",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"### {value}")

            if subtitle:
                st.caption(subtitle)



# ============================================================
# Shared Chart Theme
# ============================================================
CHART_TEMPLATE = "plotly_dark"

COLOR_NO = "#4C72B0"        # No heart disease (blue — matches the notebook)
COLOR_YES = "#DD8452"       # Heart disease (orange — matches the notebook)
COLOR_ACCENT = "#4C72B0"    # Neutral / general accent — numerical charts
COLOR_DIST_BLUE = "#4C72B0"  # Blue used for numerical & categorical distribution bars

# Diverging blue -> white -> orange scale, built from the same two notebook
# colors above, used for any chart that needs to show a signed magnitude
# (e.g. positive vs negative correlation) while staying on-palette.
COLOR_SCALE_BLUE_ORANGE_DIVERGING = [
    [0.0, "#2E4E7E"],
    [0.25, COLOR_NO],
    [0.5, "#F5F0EA"],
    [0.75, COLOR_YES],
    [1.0, "#A85D2E"],
]

# Exact Matplotlib/Seaborn coolwarm scale used by the notebook.
# The notebook uses: sns.heatmap(..., cmap='coolwarm').
COLOR_SCALE_NOTEBOOK_COOLWARM = [
    [0.000000, "#3B4CC0"],
    [0.003922, "#3C4EC2"],
    [0.007843, "#3D50C3"],
    [0.011765, "#3E51C5"],
    [0.015686, "#3F53C6"],
    [0.019608, "#4055C8"],
    [0.023529, "#4257C9"],
    [0.027451, "#4358CB"],
    [0.031373, "#445ACC"],
    [0.035294, "#455CCE"],
    [0.039216, "#465ECF"],
    [0.043137, "#485FD1"],
    [0.047059, "#4961D2"],
    [0.050980, "#4A63D3"],
    [0.054902, "#4B64D5"],
    [0.058824, "#4C66D6"],
    [0.062745, "#4E68D8"],
    [0.066667, "#4F69D9"],
    [0.070588, "#506BDA"],
    [0.074510, "#516DDB"],
    [0.078431, "#536EDD"],
    [0.082353, "#5470DE"],
    [0.086275, "#5572DF"],
    [0.090196, "#5673E0"],
    [0.094118, "#5875E1"],
    [0.098039, "#5977E3"],
    [0.101961, "#5A78E4"],
    [0.105882, "#5B7AE5"],
    [0.109804, "#5D7CE6"],
    [0.113725, "#5E7DE7"],
    [0.117647, "#5F7FE8"],
    [0.121569, "#6180E9"],
    [0.125490, "#6282EA"],
    [0.129412, "#6384EB"],
    [0.133333, "#6485EC"],
    [0.137255, "#6687ED"],
    [0.141176, "#6788EE"],
    [0.145098, "#688AEF"],
    [0.149020, "#6A8BEF"],
    [0.152941, "#6B8DF0"],
    [0.156863, "#6C8FF1"],
    [0.160784, "#6E90F2"],
    [0.164706, "#6F92F3"],
    [0.168627, "#7093F3"],
    [0.172549, "#7295F4"],
    [0.176471, "#7396F5"],
    [0.180392, "#7597F6"],
    [0.184314, "#7699F6"],
    [0.188235, "#779AF7"],
    [0.192157, "#799CF8"],
    [0.196078, "#7A9DF8"],
    [0.200000, "#7B9FF9"],
    [0.203922, "#7DA0F9"],
    [0.207843, "#7EA1FA"],
    [0.211765, "#80A3FA"],
    [0.215686, "#81A4FB"],
    [0.219608, "#82A6FB"],
    [0.223529, "#84A7FC"],
    [0.227451, "#85A8FC"],
    [0.231373, "#86A9FC"],
    [0.235294, "#88ABFD"],
    [0.239216, "#89ACFD"],
    [0.243137, "#8BADFD"],
    [0.247059, "#8CAFFE"],
    [0.250980, "#8DB0FE"],
    [0.254902, "#8FB1FE"],
    [0.258824, "#90B2FE"],
    [0.262745, "#92B4FE"],
    [0.266667, "#93B5FE"],
    [0.270588, "#94B6FF"],
    [0.274510, "#96B7FF"],
    [0.278431, "#97B8FF"],
    [0.282353, "#98B9FF"],
    [0.286275, "#9ABBFF"],
    [0.290196, "#9BBCFF"],
    [0.294118, "#9DBDFF"],
    [0.298039, "#9EBEFF"],
    [0.301961, "#9FBFFF"],
    [0.305882, "#A1C0FF"],
    [0.309804, "#A2C1FF"],
    [0.313725, "#A3C2FE"],
    [0.317647, "#A5C3FE"],
    [0.321569, "#A6C4FE"],
    [0.325490, "#A7C5FE"],
    [0.329412, "#A9C6FD"],
    [0.333333, "#AAC7FD"],
    [0.337255, "#ABC8FD"],
    [0.341176, "#ADC9FD"],
    [0.345098, "#AEC9FC"],
    [0.349020, "#AFCAFC"],
    [0.352941, "#B1CBFC"],
    [0.356863, "#B2CCFB"],
    [0.360784, "#B3CDFB"],
    [0.364706, "#B5CDFA"],
    [0.368627, "#B6CEFA"],
    [0.372549, "#B7CFF9"],
    [0.376471, "#B9D0F9"],
    [0.380392, "#BAD0F8"],
    [0.384314, "#BBD1F8"],
    [0.388235, "#BCD2F7"],
    [0.392157, "#BED2F6"],
    [0.396078, "#BFD3F6"],
    [0.400000, "#C0D4F5"],
    [0.403922, "#C1D4F4"],
    [0.407843, "#C3D5F4"],
    [0.411765, "#C4D5F3"],
    [0.415686, "#C5D6F2"],
    [0.419608, "#C6D6F1"],
    [0.423529, "#C7D7F0"],
    [0.427451, "#C9D7F0"],
    [0.431373, "#CAD8EF"],
    [0.435294, "#CBD8EE"],
    [0.439216, "#CCD9ED"],
    [0.443137, "#CDD9EC"],
    [0.447059, "#CEDAEB"],
    [0.450980, "#CFDAEA"],
    [0.454902, "#D1DAE9"],
    [0.458824, "#D2DBE8"],
    [0.462745, "#D3DBE7"],
    [0.466667, "#D4DBE6"],
    [0.470588, "#D5DBE5"],
    [0.474510, "#D6DCE4"],
    [0.478431, "#D7DCE3"],
    [0.482353, "#D8DCE2"],
    [0.486275, "#D9DCE1"],
    [0.490196, "#DADCE0"],
    [0.494118, "#DBDCDE"],
    [0.498039, "#DCDDDD"],
    [0.501961, "#DDDCDC"],
    [0.505882, "#DEDCDB"],
    [0.509804, "#DFDBD9"],
    [0.513725, "#E0DBD8"],
    [0.517647, "#E1DAD6"],
    [0.521569, "#E2DAD5"],
    [0.525490, "#E3D9D3"],
    [0.529412, "#E4D9D2"],
    [0.533333, "#E5D8D1"],
    [0.537255, "#E6D7CF"],
    [0.541176, "#E7D7CE"],
    [0.545098, "#E8D6CC"],
    [0.549020, "#E9D5CB"],
    [0.552941, "#EAD5C9"],
    [0.556863, "#EAD4C8"],
    [0.560784, "#EBD3C6"],
    [0.564706, "#ECD3C5"],
    [0.568627, "#EDD2C3"],
    [0.572549, "#EDD1C2"],
    [0.576471, "#EED0C0"],
    [0.580392, "#EFCFBF"],
    [0.584314, "#EFCEBD"],
    [0.588235, "#F0CDBB"],
    [0.592157, "#F1CDBA"],
    [0.596078, "#F1CCB8"],
    [0.600000, "#F2CBB7"],
    [0.603922, "#F2CAB5"],
    [0.607843, "#F2C9B4"],
    [0.611765, "#F3C8B2"],
    [0.615686, "#F3C7B1"],
    [0.619608, "#F4C6AF"],
    [0.623529, "#F4C5AD"],
    [0.627451, "#F5C4AC"],
    [0.631373, "#F5C2AA"],
    [0.635294, "#F5C1A9"],
    [0.639216, "#F5C0A7"],
    [0.643137, "#F6BFA6"],
    [0.647059, "#F6BEA4"],
    [0.650980, "#F6BDA2"],
    [0.654902, "#F7BCA1"],
    [0.658824, "#F7BA9F"],
    [0.662745, "#F7B99E"],
    [0.666667, "#F7B89C"],
    [0.670588, "#F7B79B"],
    [0.674510, "#F7B599"],
    [0.678431, "#F7B497"],
    [0.682353, "#F7B396"],
    [0.686275, "#F7B194"],
    [0.690196, "#F7B093"],
    [0.694118, "#F7AF91"],
    [0.698039, "#F7AD90"],
    [0.701961, "#F7AC8E"],
    [0.705882, "#F7AA8C"],
    [0.709804, "#F7A98B"],
    [0.713725, "#F7A889"],
    [0.717647, "#F7A688"],
    [0.721569, "#F6A586"],
    [0.725490, "#F6A385"],
    [0.729412, "#F6A283"],
    [0.733333, "#F5A081"],
    [0.737255, "#F59F80"],
    [0.741176, "#F59D7E"],
    [0.745098, "#F59C7D"],
    [0.749020, "#F49A7B"],
    [0.752941, "#F4987A"],
    [0.756863, "#F39778"],
    [0.760784, "#F39577"],
    [0.764706, "#F39475"],
    [0.768627, "#F29274"],
    [0.772549, "#F29072"],
    [0.776471, "#F18F71"],
    [0.780392, "#F18D6F"],
    [0.784314, "#F08B6E"],
    [0.788235, "#F08A6C"],
    [0.792157, "#EF886B"],
    [0.796078, "#EE8669"],
    [0.800000, "#EE8468"],
    [0.803922, "#ED8366"],
    [0.807843, "#EC8165"],
    [0.811765, "#EC7F63"],
    [0.815686, "#EB7D62"],
    [0.819608, "#EA7B60"],
    [0.823529, "#E97A5F"],
    [0.827451, "#E9785D"],
    [0.831373, "#E8765C"],
    [0.835294, "#E7745B"],
    [0.839216, "#E67259"],
    [0.843137, "#E57058"],
    [0.847059, "#E46E56"],
    [0.850980, "#E36C55"],
    [0.854902, "#E36B54"],
    [0.858824, "#E26952"],
    [0.862745, "#E16751"],
    [0.866667, "#E0654F"],
    [0.870588, "#DF634E"],
    [0.874510, "#DE614D"],
    [0.878431, "#DD5F4B"],
    [0.882353, "#DC5D4A"],
    [0.886275, "#DA5A49"],
    [0.890196, "#D95847"],
    [0.894118, "#D85646"],
    [0.898039, "#D75445"],
    [0.901961, "#D65244"],
    [0.905882, "#D55042"],
    [0.909804, "#D44E41"],
    [0.913725, "#D24B40"],
    [0.917647, "#D1493F"],
    [0.921569, "#D0473D"],
    [0.925490, "#CF453C"],
    [0.929412, "#CD423B"],
    [0.933333, "#CC403A"],
    [0.937255, "#CB3E38"],
    [0.941176, "#CA3B37"],
    [0.945098, "#C83836"],
    [0.949020, "#C73635"],
    [0.952941, "#C53334"],
    [0.956863, "#C43032"],
    [0.960784, "#C32E31"],
    [0.964706, "#C12B30"],
    [0.968627, "#C0282F"],
    [0.972549, "#BE242E"],
    [0.976471, "#BD1F2D"],
    [0.980392, "#BB1B2C"],
    [0.984314, "#BA162B"],
    [0.988235, "#B8122A"],
    [0.992157, "#B70D28"],
    [0.996078, "#B50927"],
    [1.000000, "#B40426"],
]

COLOR_SCALE_BLUE_SEQUENTIAL = [
    [0.0, "#EAF0F8"],
    [1.0, COLOR_NO],
]

# Qualitative blue/orange sequence for charts with more than two series
# (e.g. comparing several metrics at once), built from shades of the same
# two notebook colors so multi-series charts stay on-palette.
COLOR_SEQUENCE_BLUE_ORANGE = [
    "#4C72B0", "#DD8452", "#1F3B57", "#F4B183", "#8AA2D4", "#B15928",
]

# Chatbot avatar images (base64-encoded PNGs) — happy / neutral / sad robot
# faces used in the risk-result chat bubble, keyed by risk level.
BOT_AVATAR_LOW = "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"
BOT_AVATAR_MODERATE = "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"
BOT_AVATAR_HIGH = "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"

COLOR_SCALE_DIVERGING = COLOR_SCALE_BLUE_ORANGE_DIVERGING

MODEL_COLORS = {
    "GaussianNB (Baseline)": "#4C72B0",
    "Naive Bayes (Baseline)": "#4C72B0",
    "Random Forest": "#DD8452",
    "XGBoost": "#55A868",
    "Ensemble (Voting)": "#B15928",
}

# Matches matplotlib's "Set2" colormap as sampled by
# pandas.DataFrame.plot(kind="bar", colormap="Set2") for 6 series —
# i.e. the exact colours the notebook's "All Metrics Comparison"
# chart uses for each metric.
METRIC_COLORS_SET2 = {
    "Accuracy": "#66c2a5",
    "ROC-AUC": "#fc8d62",
    "PR-AUC": "#e78ac3",
    "MCC": "#a6d854",
    "Recall": "#e5c494",
    "F1-Score": "#b3b3b3",
}


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert a #RRGGBB hex color to an rgba() string for Plotly fills."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig, height=420, legend_title=None):
    """Apply a consistent, interactive look to every chart in the app."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=60, b=145),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            valign="middle",
        ),
        paper_bgcolor="#1e2a47",
        plot_bgcolor="#1e2a47",
        hoverlabel=dict(
            bgcolor="#131a2c",
            bordercolor="rgba(148,163,184,0.25)",
            font_size=13,
            font_family="Inter, Arial",
            font_color="#eef2f9",
        ),
        hovermode="closest",
        font=dict(size=13, color="#c2c9d6"),
    )
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


# ============================================================
# Load Dataset
# ============================================================
@st.cache_data
def get_eda_data():
    data_path = Path(__file__).parent / "heart_disease_dataset.csv"

    if not data_path.exists():
        return None

    return pd.read_csv(data_path)


hd_eda = get_eda_data()


# ============================================================
# Target Conversion Helper
# ============================================================
def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.map({0: 0, 1: 1})

    cleaned = series.astype(str).str.strip().str.lower()

    target_map = {
        "yes": 1,
        "y": 1,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "no": 0,
        "n": 0,
        "0": 0,
        "0.0": 0,
        "false": 0,
        "positive": 1,
        "negative": 0,
        "disease": 1,
        "no disease": 0,
        "heart disease": 1,
        "no heart disease": 0,
        "present": 1,
        "absent": 0,
        "abnormal": 1,
        "normal": 0
    }

    return cleaned.map(target_map)


# ============================================================
# Identify Dataset Features
# ============================================================
if hd_eda is not None:
    numeric_features = hd_eda.select_dtypes(include=np.number).columns.tolist()
    categorical_features = hd_eda.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
else:
    numeric_features = []
    categorical_features = []


# ============================================================
# Feature Definitions
# ============================================================
NUMERIC_RANGES = {
    "Age": (18, 100, 50, 1),
    "Blood Pressure": (80, 220, 120, 1),
    "Cholesterol Level": (100, 400, 200, 1),
    "BMI": (10.0, 50.0, 25.0, 0.1),
    "Sleep Hours": (0.0, 15.0, 7.0, 0.5),
    "Triglyceride Level": (50, 600, 150, 1),
    "Fasting Blood Sugar": (50, 300, 100, 1),
    "CRP Level": (0.0, 20.0, 2.0, 0.1),
    "Homocysteine Level": (0.0, 30.0, 10.0, 0.1)
}

numeric_columns = list(NUMERIC_RANGES.keys())

categorical_columns = [
    "Gender",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Alcohol Consumption",
    "Exercise Habits",
    "Stress Level",
    "Sugar Consumption"
]


# ============================================================
# Load Models and Preprocessors
# ============================================================
@st.cache_resource
def load_models():
    models = {}

    model_files = {
        "GaussianNB (Baseline)": "naive_bayes_baseline_model.pkl",
        "Random Forest": "random_forest_model.pkl",
    }

    for model_name, filename in model_files.items():
        model_path = MODEL_DIR / filename

        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")

    # XGBoost is loaded from the native JSON model format used by the
    # notebook. This avoids sklearn/joblib pickle compatibility problems.
    xgb_path = MODEL_DIR / "xgboost_model.json"
    if xgb_path.exists():
        try:
            xgb_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=400,
                n_jobs=-1
            )
            xgb_model.load_model(xgb_path)
            models["XGBoost"] = xgb_model
        except Exception as e:
            st.warning(f"Could not load XGBoost JSON model: {e}")

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------
    feature_path = MODEL_DIR / "feature_columns.pkl"

    if feature_path.exists():
        try:
            feature_columns = joblib.load(feature_path)
        except Exception as e:
            st.warning(f"Could not load feature columns: {e}")
            feature_columns = []
    else:
        feature_columns = []

    # --------------------------------------------------------
    # Preprocessors
    # --------------------------------------------------------
    preprocessors = {}

    preprocessor_files = {
        "onehot_encoder": "onehot_encoder.pkl",
        "standard_scaler": "standard_scaler.pkl",
        "median_imputer": "median_imputer.pkl",
        "mode_imputer": "mode_imputer.pkl",
        "constant_imputer": "constant_imputer.pkl"
    }

    for name, filename in preprocessor_files.items():
        path = MODEL_DIR / filename

        if path.exists():
            try:
                preprocessors[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")

    return models, feature_columns, preprocessors


models, feature_columns, preprocessors = load_models()


# ============================================================
# Encoded Test Split (for Permutation Importance)
#
# Mirrors the notebook's preprocessing exactly:
#   - numeric features: median imputation over the WHOLE dataset
#   - Gender / binary Yes-No / ordinal features: most-frequent
#     imputation, over the WHOLE dataset
#   - Alcohol Consumption: constant-imputed with "None"
#   - one-hot encoded with the saved onehot_encoder.pkl
#     (fit on the whole dataset, drop='first')
#   - split with test_size=0.30, random_state=400, stratify=y
#   - numeric features scaled with the saved standard_scaler.pkl
# ============================================================
@st.cache_data
def get_test_split_encoded():
    if hd_eda is None or "Heart Disease Status" not in hd_eda.columns:
        return None, None

    if "onehot_encoder" not in preprocessors:
        return None, None

    if "standard_scaler" not in preprocessors:
        return None, None

    hd = hd_eda.copy()

    binary_yn = [
        "Smoking",
        "Family Heart Disease",
        "Diabetes",
        "High Blood Pressure",
        "Low HDL Cholesterol",
        "High LDL Cholesterol"
    ]

    ordinal_features = [
        "Exercise Habits",
        "Stress Level",
        "Sugar Consumption"
    ]

    mode_impute_cols = ["Gender"] + binary_yn + ordinal_features

    target_col = "Heart Disease Status"

    # --------------------------------------------------------
    # Missing-value handling — exactly as in the notebook
    # --------------------------------------------------------
    # Numeric features: median imputation fitted on the whole dataset.
    if "median_imputer" in preprocessors:
        hd[numeric_features] = preprocessors["median_imputer"].transform(
            hd[numeric_features]
        )
    else:
        median_imputer = SimpleImputer(strategy="median")
        hd[numeric_features] = median_imputer.fit_transform(
            hd[numeric_features]
        )

    # Gender / binary Yes-No / ordinal: most-frequent imputation.
    mode_impute_cols = [c for c in mode_impute_cols if c in hd.columns]
    if mode_impute_cols:
        if "mode_imputer" in preprocessors:
            hd[mode_impute_cols] = preprocessors["mode_imputer"].transform(
                hd[mode_impute_cols]
            )
        else:
            mode_imputer = SimpleImputer(strategy="most_frequent")
            hd[mode_impute_cols] = mode_imputer.fit_transform(
                hd[mode_impute_cols]
            )

    # Alcohol Consumption: constant imputation with "None".
    if "Alcohol Consumption" in hd.columns:
        if "constant_imputer" in preprocessors:
            hd[["Alcohol Consumption"]] = preprocessors["constant_imputer"].transform(
                hd[["Alcohol Consumption"]]
            )
        else:
            constant_imputer = SimpleImputer(
                strategy="constant", fill_value="None"
            )
            hd[["Alcohol Consumption"]] = constant_imputer.fit_transform(
                hd[["Alcohol Consumption"]]
            )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    hd[target_col] = hd[target_col].map({"Yes": 1, "No": 0})

    valid_target = hd[target_col].notna() & hd[target_col].isin([0, 1])

    if not valid_target.any():
        return None, None

    hd = hd.loc[valid_target].copy()
    hd[target_col] = hd[target_col].astype(int)

    if hd[target_col].nunique() < 2:
        return None, None

    # --------------------------------------------------------
    # One-hot encode with the fitted encoder (whole dataset)
    # --------------------------------------------------------
    categorical_to_encode = [
        c for c in (["Gender"] + binary_yn + ["Alcohol Consumption"] + ordinal_features)
        if c in hd.columns
    ]

    try:
        ohe = preprocessors["onehot_encoder"]

        ohe_array = ohe.transform(hd[categorical_to_encode])

        hd_ohe = pd.DataFrame(
            ohe_array,
            columns=ohe.get_feature_names_out(categorical_to_encode),
            index=hd.index
        )

    except Exception:
        return None, None

    hd_enc = pd.concat(
        [hd.drop(columns=categorical_to_encode), hd_ohe],
        axis=1
    )

    X = hd_enc.drop(columns=[target_col])
    y = hd_enc[target_col]

    # --------------------------------------------------------
    # Same split used to train the saved models
    # --------------------------------------------------------
    _, X_test_enc, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=400, stratify=y
    )

    # --------------------------------------------------------
    # Scale numeric features with the saved scaler
    # --------------------------------------------------------
    scale_cols = [c for c in numeric_features if c in X_test_enc.columns]

    X_test_final = X_test_enc.copy()

    try:
        scaler = preprocessors["standard_scaler"]
        X_test_final[scale_cols] = scaler.transform(
            X_test_enc[scale_cols]
        )
    except Exception:
        return None, None

    # --------------------------------------------------------
    # Align to the exact feature order the models were trained on
    # --------------------------------------------------------
    if feature_columns:

        for col in feature_columns:
            if col not in X_test_final.columns:
                X_test_final[col] = 0

        X_test_final = X_test_final[feature_columns]

    return X_test_final, y_test


# ============================================================
# Encoded Train Split — Feature-to-Feature Correlation Heatmaps
#
# Mirrors the "Correlation & Association Analysis" tab's own
# train-only, before-SMOTE preparation (same train/test split,
# same per-column train median/mode imputation, same saved
# one-hot encoder) so the numeric and categorical correlation
# heatmaps line up with the rest of that tab and with the
# notebook's own correlation-heatmap cells.
# ============================================================
@st.cache_data
def get_train_encoded_for_feature_corr():
    if hd_eda is None or "Heart Disease Status" not in hd_eda.columns:
        return None, None, None

    if "onehot_encoder" not in preprocessors:
        return None, None, None

    analysis_data = hd_eda.copy()

    raw_target = analysis_data["Heart Disease Status"].copy()
    target_clean = raw_target.astype(str).str.strip().str.lower()

    target_mapping = {
        "0": 0, "1": 1, "0.0": 0, "1.0": 1,
        "yes": 1, "no": 0, "y": 1, "n": 0,
        "true": 1, "false": 0,
        "positive": 1, "negative": 0,
        "disease": 1, "no disease": 0,
        "heart disease": 1, "no heart disease": 0,
        "present": 1, "absent": 0,
        "abnormal": 1, "normal": 0,
        "high": 1, "low": 0,
    }

    target = target_clean.map(target_mapping)

    if target.notna().sum() == 0:
        numeric_target = pd.to_numeric(raw_target, errors="coerce")
        unique_numeric = set(numeric_target.dropna().unique())
        if unique_numeric.issubset({1, 2}):
            target = numeric_target.map({1: 0, 2: 1})

    valid_target = target.notna() & target.isin([0, 1])

    if not valid_target.any():
        return None, None, None

    analysis_data = analysis_data.loc[valid_target].copy()
    target = target.loc[valid_target].astype(int)

    if target.nunique() < 2:
        return None, None, None

    train_idx, _ = train_test_split(
        analysis_data.index, test_size=0.30, random_state=400, stratify=target
    )

    train_data = analysis_data.loc[train_idx].copy()
    X_train_raw = train_data.drop(columns=["Heart Disease Status"]).copy()

    train_numeric_cols = [c for c in numeric_columns if c in X_train_raw.columns]
    train_categorical_cols = [c for c in categorical_columns if c in X_train_raw.columns]

    X_train_num = X_train_raw[train_numeric_cols].copy()
    for col in train_numeric_cols:
        X_train_num[col] = pd.to_numeric(X_train_num[col], errors="coerce")
        median_value = X_train_num[col].median()
        if pd.isna(median_value):
            median_value = 0
        X_train_num[col] = X_train_num[col].fillna(median_value)

    X_train_cat = X_train_raw[train_categorical_cols].copy()
    for col in train_categorical_cols:
        X_train_cat[col] = X_train_cat[col].fillna("None").astype(str).str.strip()

    try:
        ohe = preprocessors["onehot_encoder"]
        ohe_array = ohe.transform(X_train_cat)
        ohe_cols = ohe.get_feature_names_out(train_categorical_cols)
        X_train_ohe = pd.DataFrame(ohe_array, columns=ohe_cols, index=X_train_raw.index)
    except Exception:
        return None, None, None

    X_train_num = X_train_num.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_train_ohe = X_train_ohe.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X_train_num, X_train_ohe, list(ohe_cols)


# ============================================================
# Load Evaluation Results
# ============================================================
@st.cache_data
def load_eval_results():
    eval_path = MODEL_DIR / "model_evaluation_results.csv"

    if not eval_path.exists():
        return None

    try:
        raw = pd.read_csv(eval_path, index_col=0)

        rename_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
            "f1_score": "F1-Score",
            "balanced_accuracy": "Balanced Accuracy",
            "mcc": "MCC",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC"
        }

        raw = raw.rename(columns=rename_map)

        return raw

    except Exception as e:
        st.error(f"Could not load model evaluation results: {e}")
        return None


eval_results = load_eval_results()


# ============================================================
# Sidebar Navigation
# ============================================================
NAV_ITEMS = [
    ("Background Overview", "🏠"),
    ("Risk Scorer", "🩺"),
    ("Visualization Analysis", "📊"),
]

if "hd_page" not in st.session_state:
    st.session_state.hd_page = NAV_ITEMS[0][0]

with st.sidebar:
    st.markdown(
        """
        <div class="hd-sidebar-brand">
            <span class="hd-sidebar-brand-icon">❤️</span>
            <span class="hd-sidebar-brand-text">Heart Disease<br>Status Predictor</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Sidebar navigation: one option per row, aligned to the left.
    for label, icon in NAV_ITEMS:
        is_active = st.session_state.hd_page == label
        if st.button(
            f"{icon}   {label}",
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.hd_page = label
            st.rerun()

page = st.session_state.hd_page


# ============================================================
# Page Router (replaces top-level tabs)
# ============================================================
# TAB 1 — INTRODUCTION
# ============================================================
if page == "Background Overview":

    # ========================================================
    # Hero — project identity
    # ========================================================
    st.markdown(
        """
        <div class="hd-dashboard-header">
            <div class="hd-header-row">
                <div class="hd-header-icon" style="width:64px;height:64px;font-size:30px;">❤️</div>
                <div>
                    <h2>Heart Disease Status Predictor</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 👥 Team Members")
    team = [
        ("👤", "Andrea Ng Wing Kay", "2508258", "GaussianNB (Baseline)"),
        ("👤", "Liew Yong Xin", "2508327", "Random Forest"),
        ("👤", "Yap Kah Ying", "2508390", "XGBoost"),
    ]
    team_cols = st.columns(3)
    for col, (icon, name, member_id, model_name) in zip(team_cols, team):
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='hd-field-head'><span class='hd-field-icon' style='background:rgba(34,211,238,.12);color:#22d3ee;'>{icon}</span><div><div class='hd-field-title'>{name}</div><div class='hd-field-desc'><b>ID:</b> {member_id}<br><b>Model:</b> {model_name}</div></div></div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📌 Business Understanding")
    overview_col, purpose_col = st.columns([1.35, 0.65], gap="large")
    with overview_col:
        with st.container(border=True):
            st.markdown("#### 🎯 Problem Statement & Background")
            st.write(
                "Cardiovascular diseases represent a major health concern worldwide. Early identification of individuals who may have heart disease can support further medical assessment and preventative action. This project presents an interactive machine learning prototype that analyses clinical measurements and lifestyle characteristics to predict Heart Disease Status using supervised classification models."
            )
    with purpose_col:
        with st.container(border=True):
            st.markdown("#### 💡 Project Purpose")
            st.write(
                "This project aims to develop a predictive framework that identifies individuals at high risk of cardiovascular disease by analysing clinical and lifestyle factors using machine learning."
            )

    st.write("")
    st.markdown("### 🚀 Project Functions")
    objective_cols = st.columns(5)
    objectives = [
        ("🎯", "Predictive Classification", "Predict whether Heart Disease Status is Present(Yes) or Absent(No)."),
        ("🔬", "Model Comparison", "Compares different machine learning classification models."),
        ("🩺", "Interactive Prediction", "Enable users to enter health and lifestyle characteristics."),
        ("📈", "Probability Estimation", "Display the estimated probability of Heart Disease Status = 1."),
        ("📊", "Model Evaluation", "Compare models using multiple classification metrics."),
    ]
    for i, (col, (icon, title, desc)) in enumerate(zip(objective_cols, objectives)):
        with col:
            with st.container(border=True, key=f"obj_card_{i}"):
                st.markdown(f"<div class='hd-field-head'><span class='hd-field-icon' style='background:rgba(167,139,250,.12);color:#a78bfa;'>{icon}</span><div><div class='hd-field-title'>{title}</div><div class='hd-field-desc'>{desc}</div></div></div>", unsafe_allow_html=True)

    st.write("")
    data_col, model_col = st.columns([1, 1], gap="large")
    total_records = len(hd_eda) if hd_eda is not None else None
    input_feature_count = (len([c for c in hd_eda.columns if c != "Heart Disease Status"]) if hd_eda is not None and "Heart Disease Status" in hd_eda.columns else None)
    disease_count = no_disease_count = None
    disease_pct = no_disease_pct = None
    if hd_eda is not None and "Heart Disease Status" in hd_eda.columns:
        normalized_status = normalize_target(hd_eda["Heart Disease Status"])
        disease_count = int((normalized_status == 1).sum())
        no_disease_count = int((normalized_status == 0).sum())
        valid_total = disease_count + no_disease_count
        if valid_total > 0:
            disease_pct = disease_count / valid_total * 100
            no_disease_pct = no_disease_count / valid_total * 100

    with data_col:
        with st.container(border=True):
            st.markdown("#### 🗄️ Dataset Overview")
            if total_records is not None:
                total_columns = (input_feature_count + 1) if input_feature_count is not None else None
                st.write(f"The project uses a heart disease dataset containing {total_records:,} records and {total_columns or 0} variables in total — {input_feature_count or 0} input features covering demographic, clinical and lifestyle information, and 1 target variable (Heart Disease Status).")
                k1, k2 = st.columns(2)
                with k1: kpi_card("Dataset Instances", f"{total_records:,}", icon="🗂️")
                with k2: kpi_card("Input Features", str(input_feature_count or 0), icon="🧩")
                k3, k4 = st.columns(2)
                with k3: kpi_card("Heart Disease", f"{disease_count:,}" if disease_count is not None else "N/A", subtitle=f"{disease_pct:.1f}%" if disease_pct is not None else None, icon="❤️")
                with k4: kpi_card("No Heart Disease", f"{no_disease_count:,}" if no_disease_count is not None else "N/A", subtitle=f"{no_disease_pct:.1f}%" if no_disease_pct is not None else None, icon="✓")
            else:
                st.warning("Dataset information is not currently available.")

    with model_col:
        with st.container(border=True):
            st.markdown("#### 🤖 Models Used")
            st.markdown("**Machine Learning Models**")
            st.markdown("- Gaussian Naive Bayes (Baseline)\n- Random Forest\n- XGBoost")
            st.markdown("**Evaluation Metrics**")
            st.markdown("Accuracy, Precision, Recall, F1-Score, Balanced Accuracy, MCC, ROC-AUC, PR-AUC")

    st.divider()

# ============================================================
# TAB 2 — RISK SCORER
# ============================================================
elif page == "Risk Scorer":

    st.markdown(
        """
        <div class="hd-dashboard-header">
            <div class="hd-header-row">
                <div class="hd-header-icon">🩺</div>
                <div>
                    <h2>Risk Scorer</h2>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not models:

        st.error(
            "No trained models were loaded. "
            "Please check your models folder."
        )

    else:

        # ------------------------------------------------------
        # Icon / description / accent-color metadata for each
        # input factor
        # ------------------------------------------------------
        FIELD_META = {
            "Age": ("👤", "Age of the patient in years.", "#38bdf8"),
            "Blood Pressure": ("❤️", "Systolic blood pressure in mmHg.", "#fb7185"),
            "Cholesterol Level": ("💧", "Total cholesterol level in mg/dL.", "#fb923c"),
            "BMI": ("⚖️", "Body Mass Index in kg/m².", "#34d399"),
            "Sleep Hours": ("😴", "Average hours of sleep per night.", "#94a3b8"),
            "Triglyceride Level": ("🧬", "Triglyceride level in mg/dL.", "#a78bfa"),
            "Fasting Blood Sugar": ("🍬", "Fasting blood sugar level in mg/dL.", "#fbbf24"),
            "CRP Level": ("🧪", "C-Reactive Protein level in mg/L.", "#a78bfa"),
            "Homocysteine Level": ("🧪", "Homocysteine level in µmol/L.", "#a78bfa"),
            "Gender": ("🧑", "Biological sex of the patient.", "#38bdf8"),
            "Smoking": ("🚬", "Whether the patient is a smoker.", "#c084fc"),
            "Family Heart Disease": ("👪", "Family history of heart disease.", "#fb923c"),
            "Diabetes": ("💉", "Whether the patient has diabetes.", "#38bdf8"),
            "High Blood Pressure": ("⚠️", "Whether the patient has hypertension.", "#fb7185"),
            "Low HDL Cholesterol": ("🔻", "Whether HDL (good) cholesterol is low.", "#fb7185"),
            "High LDL Cholesterol": ("🔺", "Whether LDL (bad) cholesterol is high.", "#fb7185"),
            "Alcohol Consumption": ("🍷", "Alcohol consumption frequency.", "#a78bfa"),
            "Exercise Habits": ("🏃", "Physical activity level.", "#2dd4bf"),
            "Stress Level": ("😖", "Level of stress experienced.", "#fb7185"),
            "Sugar Consumption": ("🍭", "Sugar consumption level.", "#fbbf24"),
        }

        CATEGORICAL_OPTIONS = {
            "Gender": ["Male", "Female"],
            "Smoking": ["Yes", "No"],
            "Family Heart Disease": ["Yes", "No"],
            "Diabetes": ["Yes", "No"],
            "High Blood Pressure": ["Yes", "No"],
            "Low HDL Cholesterol": ["Yes", "No"],
            "High LDL Cholesterol": ["Yes", "No"],
            "Alcohol Consumption": ["None", "Low", "Medium", "High"],
            "Exercise Habits": ["Low", "Medium", "High"],
            "Stress Level": ["Low", "Medium", "High"],
            "Sugar Consumption": ["Low", "Medium", "High"],
        }

        all_field_names = list(NUMERIC_RANGES.keys()) + list(CATEGORICAL_OPTIONS.keys())

        # All 20 factors grouped into four labeled sections, each laid
        # out as a two-column grid of cards. The 4th element is a short
        # subtitle used by the "Risk Factors Status" checklist card.
        SECTIONS = [
            (
                "Demographic Information", "🧑",
                ["Age", "Gender"],
                "Age, Gender",
            ),
            (
                "Clinical Measurements", "🩺",
                [
                    "BMI", "Blood Pressure", "Cholesterol Level",
                    "Triglyceride Level", "Fasting Blood Sugar",
                    "CRP Level", "Homocysteine Level", "Sleep Hours",
                ],
                "BMI, BP, Cholesterol, etc.",
            ),
            (
                "Lifestyle & Behavioral Indicators", "🏃",
                [
                    "Exercise Habits", "Stress Level", "Sugar Consumption",
                    "Alcohol Consumption", "Smoking",
                ],
                "Exercise, Stress, Sugar, etc.",
            ),
            (
                "Medical History & Risk Factors", "📋",
                [
                    "Family Heart Disease", "Diabetes", "High Blood Pressure",
                    "Low HDL Cholesterol", "High LDL Cholesterol",
                ],
                "Diabetes, Family History, etc.",
            ),
        ]

        # ------------------------------------------------------
        # "Healthy / normal" reference ranges used to decide when a
        # checklist group's status dot turns green. Age and Gender
        # have no clinical healthy range, so Demographic Information
        # is excluded from the health-status checklist entirely (see
        # is_field_healthy below).
        # ------------------------------------------------------
        NUMERIC_HEALTHY_RANGES = {
            "BMI": (18.5, 24.9),
            "Blood Pressure": (0, 120),          # normal systolic BP
            "Cholesterol Level": (0, 199),        # desirable total cholesterol
            "Triglyceride Level": (0, 149),        # normal triglycerides
            "Fasting Blood Sugar": (70, 99),        # normal fasting glucose
            "CRP Level": (0, 0.99),                # low cardiovascular risk
            "Homocysteine Level": (0, 15),          # normal homocysteine
            "Sleep Hours": (7, 9),                  # recommended sleep for adults
        }

        # Display text for the clickable information icon beside each
        # Risk Scorer group. These are reference healthy/normal ranges,
        # not diagnostic cut-offs.
        HEALTHY_THRESHOLD_DISPLAY = {
            "Clinical Measurements": [
                ("BMI", "18.5–24.9 kg/m² (healthy adult range)"),
                ("Blood Pressure", "<120/80 mmHg; systolic ≤120 mmHg is used"),
                ("Cholesterol Level", "<200 mg/dL (desirable total cholesterol)"),
                ("Triglyceride Level", "<150 mg/dL"),
                ("Fasting Blood Sugar", "70–99 mg/dL"),
                ("CRP Level", "<1.0 mg/L"),
                ("Homocysteine Level", "5–15 µmol/L"),
                ("Sleep Hours", "7–9 hours/night for most adults"),
            ],
            "Lifestyle & Behavioral Indicators": [
                ("Exercise Habits", "Medium / High"),
                ("Stress Level", "Low"),
                ("Sugar Consumption", "Low"),
                ("Alcohol Consumption", "None / Low"),
                ("Smoking", "No"),
            ],
            "Medical History & Risk Factors": [
                ("Family Heart Disease", "No family history"),
                ("Diabetes", "No"),
                ("High Blood Pressure", "No"),
                ("Low HDL Cholesterol", "No"),
                ("High LDL Cholesterol", "No"),
            ],
        }

        CATEGORICAL_HEALTHY_VALUES = {
            "Smoking": {"No"},
            "Family Heart Disease": {"No"},
            "Diabetes": {"No"},
            "High Blood Pressure": {"No"},
            "Low HDL Cholesterol": {"No"},
            "High LDL Cholesterol": {"No"},
            "Alcohol Consumption": {"None", "Low"},
            "Exercise Habits": {"Medium", "High"},
            "Stress Level": {"Low"},
            "Sugar Consumption": {"Low"},
        }

        def is_field_healthy(col_name):
            """True if the current value for col_name falls within its
            defined healthy/normal range. Fields with no defined range
            (Age, Gender) are treated as always satisfied."""

            value = st.session_state.get(f"input_{col_name}", None)

            if value is None or value == "":
                return False

            if col_name in NUMERIC_HEALTHY_RANGES:
                lo, hi = NUMERIC_HEALTHY_RANGES[col_name]
                try:
                    return lo <= float(value) <= hi
                except (TypeError, ValueError):
                    return False

            if col_name in CATEGORICAL_HEALTHY_VALUES:
                return value in CATEGORICAL_HEALTHY_VALUES[col_name]

            return True

        def group_is_healthy(fields):
            return all(is_field_healthy(f) for f in fields)

        def group_status(fields):
            """Traffic-light status for a group of fields.
            - 'healthy' (green): every field is within its healthy range.
            - 'danger' (red): more than half of the fields are NOT within
              their healthy range.
            - 'warning' (yellow): otherwise — some fields are out of
              range, but not more than half.
            """
            total = len(fields)
            if total == 0:
                return "healthy"
            healthy_count = sum(is_field_healthy(f) for f in fields)
            unhealthy_count = total - healthy_count
            if healthy_count == total:
                return "healthy"
            if unhealthy_count > total / 2:
                return "danger"
            return "warning"

        def render_risk_checklist_html():
            """Build the Risk Factors Status card as HTML so it can sit
            INSIDE the holographic patient container beside the body.
            This lets the biometric dots and future callout lines share
            the same visual coordinate space."""

            rows_html = ""

            for title, icon, fields, short_desc in SECTIONS:
                if title == "Demographic Information":
                    # Age/Gender have no healthy/unhealthy status, so they
                    # don't belong in a health-status checklist.
                    continue
                rows_html += textwrap.dedent(f"""
                    <div class="hd-checklist-row">
                        <div class="hd-checklist-icon">{icon}</div>
                        <div class="hd-checklist-text">
                            <div class="hd-checklist-title">{title}</div>
                            <div class="hd-checklist-desc">{short_desc}</div>
                        </div>
                    </div>
                """).strip()

            return textwrap.dedent(f"""
                <div class="hd-checklist-inside">
                    <div class="hd-checklist-header">🛡️ Risk Factors Status</div>
                    {rows_html}
                    <div class="hd-checklist-legend">
                        <span><span class="hd-checklist-dot danger"></span>High Health Risk</span>
                        <span><span class="hd-checklist-dot warning"></span>Needs Attention</span>
                        <span><span class="hd-checklist-dot healthy"></span>Healthy Range</span>
                    </div>
                </div>
            """).strip()

        col_main, col_summary = st.columns([2.0, 1.4], gap="large")

        # ========================================================
        # Right rail — model selector + risk-aware body figure
        # ========================================================
        def render_body_figure(risk_level="neutral"):
            """Render the risk-factor checklist card next to a large
            futuristic holographic patient body with animated HUD effects."""
            colors={"high":"#fb7185","moderate":"#fbbf24","low":"#34d399","neutral":"#94a3b8"}
            labels={"high":"High Risk","moderate":"Moderate Risk","low":"Low Risk","neutral":"Not Yet Assessed"}
            fill=colors.get(risk_level,colors["neutral"])
            status_label=labels.get(risk_level,labels["neutral"])

            # Keep the three biometric markers synchronized with the
            # corresponding "Risk Factors Status" checklist rows. Age/Gender
            # have no healthy/unhealthy status, so Demographic Information
            # isn't part of this.
            clinical_dot = group_status(SECTIONS[1][2])
            lifestyle_dot = group_status(SECTIONS[2][2])
            history_dot = group_status(SECTIONS[3][2])

            def val(col_name):
                v=st.session_state.get(f"input_{col_name}","")
                return v if v!="" else "—"

            # The checklist is intentionally rendered INSIDE the holographic
            # container. The left side is reserved for status cards/callouts;
            # the patient figure occupies the right side.
            checklist_html = render_risk_checklist_html()

            html=f"""
            <div class="hd-body-figure-title">Patient Overview <span class="hd-body-status" style="color:{fill};">● {status_label}</span></div>
            <div class="hd-holo-card">
                <div class="hd-holo-grid"></div><div class="hd-scanline"></div>
                <div class="hd-holo-label">VIRTUAL BIOMETRIC SCAN</div><div class="hd-holo-status">RISK PROFILE</div>
                {checklist_html}
                <div class="hd-callout-line line-1"></div>
                <div class="hd-callout-line line-2"></div>
                <div class="hd-callout-line line-3"></div>
                <svg class="hd-holo-figure" viewBox="0 0 360 650" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="bodyStroke" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset=".48" stop-color="#60a5fa"/><stop offset="1" stop-color="#a78bfa"/></linearGradient>
                        <linearGradient id="bodyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#123c69" stop-opacity=".60"/><stop offset="1" stop-color="#071326" stop-opacity=".24"/></linearGradient>
                        <radialGradient id="heartFill"><stop offset="0" stop-color="#fff"/><stop offset=".24" stop-color="#fb7185"/><stop offset="1" stop-color="#ec4899" stop-opacity=".08"/></radialGradient>
                        <filter id="glowPink"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                        <filter id="glowPinkStrong"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                        <filter id="glowCyan"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                    </defs>
                    <circle class="hd-hud-ring" cx="180" cy="315" r="174"/><circle class="hd-hud-ring dash" cx="180" cy="315" r="154"/><circle class="hd-hud-ring" cx="180" cy="315" r="126"/>
                    <path d="M180 42 C154 42 140 59 141 82 C142 105 153 118 180 119 C207 118 218 105 219 82 C220 59 206 42 180 42 Z" class="hd-holo-outline"/>
                    <path d="M151 119 C132 124 112 141 105 163 L86 224 C82 239 91 250 104 247 L128 238 L137 337 L151 421 L157 545 L174 545 L180 421 L186 545 L203 545 L209 421 L223 337 L232 238 L256 247 C269 250 278 239 274 224 L255 163 C248 141 228 124 209 119 Z" class="hd-holo-outline"/>
                    <path d="M137 338 L157 421 M223 338 L203 421 M151 120 C163 135 197 135 209 120 M137 177 C155 189 205 189 223 177" fill="none" stroke="#38bdf8" stroke-width="1.4" opacity=".62"/>
                    <path d="M180 205 C171 193 148 204 153 222 C158 239 180 253 180 253 C180 253 202 239 207 222 C212 204 189 193 180 205 Z" fill="url(#heartFill)" class="hd-heart-glow"/>
                    <path d="M151 222 H165 L171 212 L178 230 L184 216 L190 222 H207" fill="none" stroke="#fff" stroke-width="2.2" filter="url(#glowPink)"/>
                    <path d="M180 121 V180 M180 255 V420" stroke="#22d3ee" stroke-width="1" stroke-dasharray="2 7" opacity=".32"/>
                    <ellipse cx="180" cy="582" rx="112" ry="25" fill="none" stroke="#22d3ee" stroke-width="1.6" opacity=".65"/><ellipse cx="180" cy="582" rx="84" ry="17" fill="none" stroke="#60a5fa" stroke-width="1" opacity=".55"/>
                </svg>
                <div class="hd-hotspot" style="top:31%;left:70%;">
                    <span class="hd-hotspot-dot {clinical_dot}"></span>
                    <div class="hd-tooltip">
                        <b>🩺 Clinical Measurements</b><br>
                        BMI: {val("BMI")} kg/m²<br>
                        Blood Pressure: {val("Blood Pressure")} mmHg<br>
                        Cholesterol: {val("Cholesterol Level")} mg/dL<br>
                        Triglycerides: {val("Triglyceride Level")} mg/dL<br>
                        Fasting Sugar: {val("Fasting Blood Sugar")} mg/dL<br>
                        CRP: {val("CRP Level")} mg/L<br>
                        Homocysteine: {val("Homocysteine Level")} µmol/L<br>
                        Sleep: {val("Sleep Hours")} hrs
                    </div>
                </div>
                <div class="hd-hotspot" style="top:41%;left:70%;">
                    <span class="hd-hotspot-dot {lifestyle_dot}"></span>
                    <div class="hd-tooltip">
                        <b>🏃 Lifestyle &amp; Behavioral Indicators</b><br>
                        Exercise: {val("Exercise Habits")}<br>
                        Stress: {val("Stress Level")}<br>
                        Sugar: {val("Sugar Consumption")}<br>
                        Alcohol: {val("Alcohol Consumption")}<br>
                        Smoking: {val("Smoking")}
                    </div>
                </div>
                <div class="hd-hotspot" style="top:51%;left:70%;">
                    <span class="hd-hotspot-dot {history_dot}"></span>
                    <div class="hd-tooltip">
                        <b>📋 Medical History &amp; Risk Factors</b><br>
                        Family Heart Disease: {val("Family Heart Disease")}<br>
                        Diabetes: {val("Diabetes")}<br>
                        High Blood Pressure: {val("High Blood Pressure")}<br>
                        Low HDL Cholesterol: {val("Low HDL Cholesterol")}<br>
                        High LDL Cholesterol: {val("High LDL Cholesterol")}
                    </div>
                </div>
                <div class="hd-body-hint">Hover over a glowing marker for patient details</div>
            </div>"""

            st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

        with col_summary:
            st.markdown("#### 📋 Prediction Summary")
            available_models = list(models.keys())
            selected_model = st.selectbox(
                "Model Used",
                available_models,
                key="hd_model_select",
                filter_mode=None,
            )
            model = models[selected_model]

        # ========================================================
        # Main column — grouped factor cards (two-column grid per
        # section: Demographic / Clinical / Additional)
        # ========================================================
        def render_factor_card(col_name):
            """Render one factor card with a colored icon."""

            icon, desc, color = FIELD_META.get(col_name, ("📈", "", "#fb7185"))

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div class="hd-field-head">
                        <span class="hd-field-icon" style="background:{color}1F; color:{color};">{icon}</span>
                        <div>
                            <div class="hd-field-title" style="color:#FFFFFF;">{col_name}</div>
                            <div class="hd-field-desc">{desc}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if col_name in NUMERIC_RANGES:

                    lo, hi, default, step = NUMERIC_RANGES[col_name]
                    is_float = isinstance(default, float)

                    user_input[col_name] = st.number_input(
                        col_name,
                        min_value=float(lo) if is_float else int(lo),
                        max_value=float(hi) if is_float else int(hi),
                        value=float(default) if is_float else int(default),
                        step=0.01 if is_float else 1,
                        format="%.2f" if is_float else "%d",
                        key=f"input_{col_name}",
                        label_visibility="collapsed",
                    )

                else:
                    # All categorical Risk Scorer inputs use the same dropdown
                    # treatment, including Yes/No and Male/Female fields.
                    options = CATEGORICAL_OPTIONS[col_name]
                    user_input[col_name] = st.selectbox(
                        col_name,
                        options,
                        key=f"input_{col_name}",
                        label_visibility="collapsed",
                        filter_mode=None,
                    )

        with col_main:

            user_input = {}

            with st.form("prediction_form"):

                for title, icon, fields, short_desc in SECTIONS:

                    # Group heading + clickable info icon. The popover shows
                    # every feature in this group and its healthy reference
                    # threshold/range, while keeping the main UI uncluttered.
                    # Demographic Information (Age, Gender) has no healthy/
                    # unhealthy status, so it gets a plain heading with no
                    # popover.
                    if title == "Demographic Information":
                        st.markdown(
                            f'<div class="hd-section-title">{icon} {title}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        header_col, info_col = st.columns([12, 1], gap="small")
                        with header_col:
                            st.markdown(
                                f'<div class="hd-section-title">{icon} {title}</div>',
                                unsafe_allow_html=True,
                            )
                        with info_col:
                            with st.popover(
                                "",
                                use_container_width=True,
                            ):
                                st.markdown(f"**{icon} {title} — Recommended Ranges**")
                                for feature_name, threshold in HEALTHY_THRESHOLD_DISPLAY.get(title, []):
                                    st.markdown(
                                        f"**{feature_name}**  \n{threshold}"
                                    )

                    grid_cols = st.columns(2)

                    for i, col_name in enumerate(fields):
                        with grid_cols[i % 2]:
                            render_factor_card(col_name)

                st.markdown(
                    """
                    <div class="hd-info-box interpretation">
                        ℹ️ Fill in the factors above, then click
                        <b>Predict Risk</b> to update the prediction.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                submitted = st.form_submit_button(
                    "❤️ Predict Risk",
                    use_container_width=True,
                    type="primary",
                )

        # ========================================================
        # Prediction
        # ========================================================
        prediction = None
        heart_disease_probability = None
        input_df = None
        error_message = None

        if submitted:

            try:

                # ------------------------------------------------
                # Create raw input DataFrame
                # ------------------------------------------------
                raw_df = pd.DataFrame([user_input])

                # ------------------------------------------------
                # Validate preprocessors
                # ------------------------------------------------
                if "onehot_encoder" not in preprocessors:
                    raise FileNotFoundError("onehot_encoder.pkl is missing.")

                if "standard_scaler" not in preprocessors:
                    raise FileNotFoundError("standard_scaler.pkl is missing.")

                if not feature_columns:
                    raise ValueError(
                        "feature_columns.pkl is empty or missing."
                    )

                # ------------------------------------------------
                # Numerical Features
                # ------------------------------------------------
                numeric_part = raw_df[numeric_columns].copy()

                # ------------------------------------------------
                # One-Hot Encoding
                # ------------------------------------------------
                ohe = preprocessors["onehot_encoder"]

                ohe_array = ohe.transform(
                    raw_df[categorical_columns]
                )

                ohe_df = pd.DataFrame(
                    ohe_array,
                    columns=ohe.get_feature_names_out(categorical_columns),
                    index=raw_df.index
                )

                # ------------------------------------------------
                # Combine Numerical + Encoded Features
                # ------------------------------------------------
                input_df = pd.concat(
                    [numeric_part, ohe_df],
                    axis=1
                )

                # ------------------------------------------------
                # Match Exact Training Feature Order
                # ------------------------------------------------
                input_df = input_df.reindex(
                    columns=feature_columns,
                    fill_value=0
                )

                # ------------------------------------------------
                # Standard Scaling
                # ------------------------------------------------
                scaler = preprocessors["standard_scaler"]

                scaler_columns = [
                    col for col in numeric_columns
                    if col in input_df.columns
                ]

                if scaler_columns:
                    input_df[scaler_columns] = scaler.transform(
                        input_df[scaler_columns]
                    )

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------
                prediction = int(model.predict(input_df)[0])

                # ------------------------------------------------
                # Get ACTUAL Probability
                # ------------------------------------------------
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)[0]

                    if hasattr(model, "classes_"):

                        classes = list(model.classes_)

                        if 1 in classes:
                            class_index = classes.index(1)
                            heart_disease_probability = float(
                                probabilities[class_index]
                            )
                        else:
                            heart_disease_probability = float(
                                probabilities[-1]
                            )

                    else:
                        heart_disease_probability = float(
                            probabilities[-1]
                        )

            except Exception as e:

                error_message = str(e)

        # ========================================================
        # Right rail — render results underneath the model selector
        # ========================================================
        with col_summary:

            if submitted and error_message:

                render_body_figure("neutral")
                st.error("Prediction failed.")
                st.caption(error_message)

            elif submitted and prediction is not None:

                safe_probability = (
                    min(max(heart_disease_probability, 0.0), 1.0)
                    if heart_disease_probability is not None
                    else None
                )

                if safe_probability is None:
                    risk_text, risk_color, badge_class = "UNKNOWN", "#94a3b8", "moderate"
                elif safe_probability < 0.30:
                    risk_text, risk_color, badge_class = "LOW", "#34d399", "low"
                elif safe_probability < 0.70:
                    risk_text, risk_color, badge_class = "MODERATE", "#fbbf24", "moderate"
                else:
                    risk_text, risk_color, badge_class = "HIGH", "#fb7185", "high"

                render_body_figure(badge_class)

                badge_sub = "Heart Disease Likely" if prediction == 1 else "Heart Disease Unlikely"

                prediction_status = "YES" if prediction == 1 else "NO"
                prediction_icon = "❤️" if prediction == 1 else "✓"

                st.markdown(
                    f"""
                    <div class="hd-risk-badge {badge_class}">
                        <div class="hd-risk-badge-label">Heart Disease Status</div>
                        <div class="hd-risk-badge-value {badge_class}">
                            {prediction_icon} {prediction_status}
                        </div>
                        <div style="font-size:12px;color:var(--hd-text-dim);margin-top:2px;">
                            {badge_sub} · {risk_text.title()} Risk
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if safe_probability is not None:

                    safe_probability_pct = safe_probability * 100

                    gauge_fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=safe_probability_pct,
                            number={
                                "suffix": "%",
                                "font": {"size": 32, "color": risk_color},
                            },
                            title={
                                "text": "Probability of Heart Disease",
                                "font": {"size": 13, "color": "#c2c9d6"},
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "ticksuffix": "%",
                                    "tickwidth": 1,
                                },
                                "bar": {"color": risk_color, "thickness": 0.3},
                                "bgcolor": "rgba(19,26,44,0.45)",
                                "borderwidth": 1,
                                "bordercolor": "rgba(148,163,184,0.25)",
                                "steps": [
                                    {"range": [0, 30], "color": "rgba(52,211,153,0.18)"},
                                    {"range": [30, 70], "color": "rgba(251,191,36,0.18)"},
                                    {"range": [70, 100], "color": "rgba(251,113,133,0.18)"},
                                ],
                                "threshold": {
                                    "line": {"color": "#f8fafc", "width": 3},
                                    "thickness": 0.85,
                                    "value": safe_probability_pct,
                                },
                            },
                        )
                    )

                    gauge_fig.update_layout(
                        height=230,
                        margin=dict(l=20, r=20, t=50, b=10),
                        font=dict(size=12, color="#c2c9d6"),
                        # Darker background specifically for the probability gauge.
                        # This only affects the gauge chart, not the other charts.
                        paper_bgcolor="#1e2a47",
                        plot_bgcolor="#1e2a47",
                    )

                    st.plotly_chart(
                        gauge_fig,
                        use_container_width=True,
                        config={
                            "responsive": True,
                            "displayModeBar": False,
                            "scrollZoom": False,
                        },
                    )

                # The message shown depends first on whether the model
                # actually PREDICTED heart disease (YES/NO), and only then
                # on the numeric risk band (Low/Moderate/High):
                # - prediction == YES  -> regardless of whether the risk
                #   band is Moderate or High, treat this as a predicted
                #   case of heart disease and urge immediate action.
                # - prediction == NO   -> fall back to the softer,
                #   risk-band-based caution/guidance messages.
                if prediction_status == "YES":

                    risk_qualifier = (
                        f" ({risk_text.title()} Risk)"
                        if risk_text != "UNKNOWN"
                        else ""
                    )

                    results_sentence = (
                        f"Your results indicate a "
                        f"<b style=\"color:{risk_color};\">PREDICTED CASE OF "
                        f"HEART DISEASE</b>{risk_qualifier}."
                    )

                    interpretation_text = (
                        "This means the model predicts that you already "
                        "have heart disease. Please consult a healthcare "
                        "professional as soon as possible and take "
                        "immediate action to protect your heart health."
                    )

                else:

                    risk_headline = {
                        "LOW": "a <b style=\"color:#16a34a;\">LOW RISK</b>",
                        "MODERATE": "a <b style=\"color:#fbbf24;\">MODERATE RISK</b>",
                        "HIGH": "a <b style=\"color:#dc2626;\">HIGH RISK</b>",
                        "UNKNOWN": "an <b>UNCLEAR RISK</b>",
                    }[risk_text]

                    results_sentence = (
                        f"Your results indicate {risk_headline} of heart disease."
                    )

                    interpretation_text = {
                        "LOW": (
                            "That's good news! Keep up your healthy habits to "
                            "help maintain your heart health."
                        ),
                        "MODERATE": (
                            "Please pay closer attention to your lifestyle and "
                            "heart health, and take the necessary precautions "
                            "to reduce your risk."
                        ),
                        "HIGH": (
                            "It is highly recommended to consult a healthcare "
                            "professional for a thorough evaluation."
                        ),
                        "UNKNOWN": (
                            "The selected model does not expose a probability "
                            "score, so only the predicted class is shown."
                        ),
                    }[risk_text]

                if prediction_status == "YES":
                    # A predicted case of heart disease always gets the sad
                    # face, regardless of the numeric risk band (Low/
                    # Moderate/High/Unknown).
                    bot_avatar_b64 = BOT_AVATAR_HIGH
                else:
                    bot_avatar_b64 = {
                        "LOW": BOT_AVATAR_LOW,
                        "MODERATE": BOT_AVATAR_MODERATE,
                        "HIGH": BOT_AVATAR_HIGH,
                        "UNKNOWN": BOT_AVATAR_MODERATE,
                    }[risk_text]

                st.markdown(
                    f"""
                    <div class="hd-bot-wrap">
                        <div class="hd-bot-avatar">
                            <img src="data:image/png;base64,{bot_avatar_b64}" alt="Assistant avatar" />
                        </div>
                        <div class="hd-bot-bubbles">
                            <div class="hd-bot-bubble">
                                Hi! I've analyzed your health inputs and
                                here's what I found.<br><br>
                                {results_sentence} {interpretation_text}
                            </div>
                            <div class="hd-bot-bubble">
                                <b>Here are my recommendations for you:</b>
                                <ul>
                                    <li>🩺 Consult a healthcare professional.</li>
                                    <li>🧪 Consider additional diagnostic tests.</li>
                                    <li>🥗 Maintain a heart-healthy lifestyle.</li>
                                    <li>📅 Schedule regular health check-ups.</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if input_df is not None:
                    with st.expander("View Processed Input"):
                        st.dataframe(input_df, use_container_width=True)

            else:

                render_body_figure("neutral")
                st.info(
                    "Fill in the factors on the left and click "
                    "**Predict Risk** to see your results here."
                )

# ============================================================
# TAB 3 — Visualization Analysis
# ============================================================
elif page == "Visualization Analysis":

    st.markdown(
        """
        <div class="hd-dashboard-header">
            <div class="hd-header-row">
                <div class="hd-header-icon">📊</div>
                <div>
                    <h2>Visualization Analysis</h2>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    eda_tab, corr_tab, eval_tab, feat_tab = st.tabs([
        "Exploratory Data Analysis",
        "Correlation & Association Analysis",
        "Model Evaluation",
        "Feature Importance"
    ])

    # ========================================================
    # EDA — LANDING PAGE + DEDICATED CHART PAGES
    # ========================================================
    with eda_tab:

        if hd_eda is None:
            st.warning("heart_disease_dataset.csv was not found.")

        else:
            target_available = "Heart Disease Status" in hd_eda.columns
            plain_numeric_features = [
                c for c in numeric_features if c != "Heart Disease Status"
            ]
            plain_categorical_features = [
                c for c in categorical_features if c != "Heart Disease Status"
            ]

            # Keep the EDA page inside this tab independent from the
            # application's main sidebar navigation.
            if "eda_view" not in st.session_state:
                st.session_state.eda_view = "Overview"

            # --------------------------------------------------------
            # Shared chart helpers
            # --------------------------------------------------------
            def _eda_status_labels():
                return ["No", "Yes"]

            def _eda_status_colors():
                return [COLOR_NO, COLOR_YES]

            def _status_traces(values_no, values_yes):
                return [
                    ("No Heart Disease (0)", COLOR_NO, values_no),
                    ("Heart Disease (1)", COLOR_YES, values_yes),
                ]

            def _eda_back_button():
                if st.button("← Back to EDA Overview", key="eda_back"):
                    st.session_state.eda_view = "Overview"
                    st.rerun()

            def _box_stats(vals):
                """Compute the six box-plot statistics we actually want to
                surface on hover — deliberately no fence. The fence is just
                the whisker's outlier cutoff (Q3/Q1 +/- 1.5*IQR); for this
                dataset's bounded health features it lines up with the
                actual max/min far more often than not, so it's redundant
                noise rather than useful information."""
                return dict(
                    min=vals.min(),
                    q1=vals.quantile(0.25),
                    median=vals.median(),
                    mean=vals.mean(),
                    q3=vals.quantile(0.75),
                    max=vals.max(),
                )

            def _add_clean_box(fig, vals, label, color):
                """Draw a box trace for visuals only (its native hover is
                fully disabled) plus a small invisible marker overlay that
                supplies the one tooltip we want. This is the only way to
                keep Plotly from also popping up its own separate
                fence-labelled tooltip at the whisker tips."""
                stats = _box_stats(vals)

                fig.add_trace(
                    go.Box(
                        y=vals,
                        name=label,
                        marker_color=color,
                        boxmean=False,
                        hoverinfo="skip",
                    )
                )

                hover_text = (
                    f"<b>{label}</b><br>"
                    f"Max: {stats['max']:.2f}<br>"
                    f"Q3: {stats['q3']:.2f}<br>"
                    f"Median: {stats['median']:.2f}<br>"
                    f"Q1: {stats['q1']:.2f}<br>"
                    f"Min: {stats['min']:.2f}"
                    "<extra></extra>"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[label] * 5,
                        y=[
                            stats["min"], stats["q1"], stats["median"],
                            stats["q3"], stats["max"],
                        ],
                        mode="markers",
                        marker=dict(size=8, color=color, opacity=0.02),
                        hovertemplate=hover_text,
                        showlegend=False,
                        name=label,
                    )
                )

            # ========================================================
            # OVERVIEW
            # ========================================================
            if st.session_state.eda_view == "Overview":

                st.markdown("### 📊 Exploratory Data Analysis")

                # ----------------------------------------------------
                # Target balance — OUTSIDE the three containers
                # ----------------------------------------------------
                if target_available:
                    counts = (
                        hd_eda["Heart Disease Status"]
                        .value_counts()
                        .reindex(["No", "Yes"], fill_value=0)
                    )
                    no_count = int(counts["No"])
                    yes_count = int(counts["Yes"])
                    total_count = no_count + yes_count

                    st.markdown("#### 🎯 Target Balance — Heart Disease Status")

                    target_fig = go.Figure(
                        go.Pie(
                            labels=[
                                "No Heart Disease (0)",
                                "Heart Disease (1)",
                            ],
                            values=[no_count, yes_count],
                            hole=0.65,
                            marker=dict(
                                colors=[COLOR_NO, COLOR_YES]
                            ),
                            textinfo="none",
                            hovertemplate=(
                                "<b>%{label}</b><br>"
                                "%{value:,} (%{percent})"
                                "<extra></extra>"
                            ),
                        )
                    )

                    target_fig.update_layout(
                        annotations=[
                            dict(
                                text=f"Total<br><b>{total_count:,}</b>",
                                x=0.5,
                                y=0.5,
                                font_size=16,
                                showarrow=False,
                            )
                        ],
                        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            valign="middle",
        ),
                        margin=dict(l=50, r=50, t=40, b=110),
                    )

                    style_fig(target_fig, height=340, legend_title=None)

                    target_col, _ = st.columns([1, 1])
                    with target_col:
                        st.plotly_chart(
                            target_fig,
                            use_container_width=True,
                        )

                st.markdown("---")

                # ----------------------------------------------------
                # Three horizontally aligned navigation containers
                # ----------------------------------------------------
                st.markdown("#### 🔎 Explore the Data")

                card1, card2, card3 = st.columns(3, gap="medium")

                with card1:
                    with st.container(border=True, key="eda_nav_card_1"):
                        st.markdown("### 📈 Numerical & Categorical Distributions")
                        st.write(
                            "View how individual numerical and categorical "
                            "features are distributed across the dataset."
                        )
                        st.caption(
                            "Box plots for numerical variables and bar "
                            "charts for categorical variables."
                        )
                        if st.button(
                            "View Distributions →",
                            key="eda_open_distributions",
                            use_container_width=True,
                        ):
                            st.session_state.eda_view = "Distributions"
                            st.rerun()

                with card2:
                    with st.container(border=True, key="eda_nav_card_2"):
                        st.markdown("### 🎯 Features vs Heart Disease")
                        st.write(
                            "Compare individual numerical and categorical "
                            "features against the predicted target groups."
                        )
                        st.caption(
                            "Box plots and grouped bar charts show how "
                            "features differ between No and Yes cases."
                        )
                        if st.button(
                            "View Feature Comparisons →",
                            key="eda_open_target",
                            use_container_width=True,
                        ):
                            st.session_state.eda_view = "Target Comparisons"
                            st.rerun()

                with card3:
                    with st.container(border=True, key="eda_nav_card_3"):
                        st.markdown("### 🔗 Feature Relationships")
                        st.write(
                            "See how features relate to one another "
                            "using correlation heatmaps."
                        )
                        st.caption(
                            "Heatmap showing relationships among numerical features."
                        )
                        if st.button(
                            "View Relationships →",
                            key="eda_open_relationships",
                            use_container_width=True,
                        ):
                            st.session_state.eda_view = "Relationships"
                            st.rerun()

            # ========================================================
            # PAGE 1 — DISTRIBUTIONS
            # ========================================================
            elif st.session_state.eda_view == "Distributions":

                _eda_back_button()

                st.markdown("### 📈 Numerical & Categorical Distributions")

                dist_type = st.radio(
                    "Distribution Type",
                    ["Numerical Features", "Categorical Features"],
                    horizontal=True,
                    key="eda_distribution_type",
                )

                if dist_type == "Numerical Features":

                    if plain_numeric_features:

                        dist_num_cols = st.columns(2)

                        for i, selected_num in enumerate(plain_numeric_features):

                            with dist_num_cols[i % 2]:
                                with st.container(border=True):

                                    vals = hd_eda[selected_num].dropna()

                                    fig = go.Figure()
                                    _add_clean_box(fig, vals, selected_num, COLOR_DIST_BLUE)

                                    fig.update_layout(
                                        title=f"Distribution of {selected_num}",
                                        yaxis_title=selected_num,
                                        xaxis_title=selected_num,
                                        showlegend=False,
                                        margin=dict(l=70, r=70, t=70, b=60),
                                    )

                                    style_fig(fig, height=600, legend_title=None)
                                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.25)")
                                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.25)")
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"eda_dist_num_{selected_num}",
                                    )

                    else:
                        st.info("No numerical features are available.")

                else:

                    if plain_categorical_features:

                        dist_cat_cols = st.columns(2)

                        for i, selected_cat in enumerate(plain_categorical_features):

                            with dist_cat_cols[i % 2]:
                                with st.container(border=True):

                                    counts = hd_eda[selected_cat].value_counts()
                                    total_count = int(counts.sum())
                                    percentages = (counts / total_count * 100) if total_count else counts * 0

                                    # Per-category colours mirroring the notebook's
                                    # standard_color_map (Male/No/High -> blue,
                                    # Female/Yes -> orange, Low/Medium on the same
                                    # blue scale), falling back to the blue/orange
                                    # sequence for any other category values.
                                    _cat_color_map = {
                                        "Male": COLOR_NO,
                                        "Female": COLOR_YES,
                                        "No": COLOR_NO,
                                        "Yes": COLOR_YES,
                                        "Low": "#A4C2E6",
                                        "Medium": "#8AA2D4",
                                        "High": COLOR_NO,
                                    }
                                    bar_colors = [
                                        _cat_color_map.get(
                                            str(cat),
                                            COLOR_SEQUENCE_BLUE_ORANGE[
                                                j % len(COLOR_SEQUENCE_BLUE_ORANGE)
                                            ],
                                        )
                                        for j, cat in enumerate(counts.index)
                                    ]

                                    fig = go.Figure(
                                        go.Bar(
                                            x=counts.index.astype(str),
                                            y=counts.values,
                                            marker_color=bar_colors,
                                            name=selected_cat,
                                            customdata=percentages.values,
                                            hovertemplate=(
                                                f"<b>{selected_cat}</b>: %{{x}}<br>"
                                                "Count: %{y}<br>"
                                                "Percentage: %{customdata:.1f}%"
                                                "<extra></extra>"
                                            ),
                                        )
                                    )

                                    fig.update_layout(
                                        title=f"Distribution of {selected_cat}",
                                        xaxis_title=selected_cat,
                                        yaxis_title="Count",
                                        showlegend=False,
                                        margin=dict(l=70, r=70, t=70, b=60),
                                    )

                                    style_fig(fig, height=600, legend_title=None)
                                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.25)")
                                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.25)")
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"eda_dist_cat_{selected_cat}",
                                    )

                    else:
                        st.info("No categorical features are available.")

            # ========================================================
            # PAGE 2 — FEATURES VS TARGET
            # ========================================================
            elif st.session_state.eda_view == "Target Comparisons":

                _eda_back_button()

                st.markdown("### 🎯 Features vs Heart Disease Status")

                comparison_type = st.radio(
                    "Comparison Type",
                    [
                        "Numerical Feature",
                        "Categorical Feature",
                    ],
                    horizontal=True,
                    key="eda_comparison_type",
                )

                if not target_available:
                    st.warning(
                        "Heart Disease Status column was not found."
                    )

                elif comparison_type == "Numerical Feature":

                    if plain_numeric_features:

                        target_num_cols = st.columns(2)

                        for i, selected_num_target in enumerate(plain_numeric_features):

                            with target_num_cols[i % 2]:
                                with st.container(border=True):

                                    fig = go.Figure()

                                    for label, color, vals in _status_traces(
                                        hd_eda.loc[
                                            hd_eda["Heart Disease Status"] == "No",
                                            selected_num_target,
                                        ].dropna(),
                                        hd_eda.loc[
                                            hd_eda["Heart Disease Status"] == "Yes",
                                            selected_num_target,
                                        ].dropna(),
                                    ):
                                        _add_clean_box(fig, vals, label, color)

                                    fig.update_layout(
                                        title=(
                                            f"{selected_num_target} vs "
                                            "Heart Disease Status"
                                        ),
                                        yaxis_title=selected_num_target,
                                        xaxis_title="Heart Disease Status",
                                        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            valign="middle",
        ),
                                        margin=dict(l=70, r=70, t=70, b=150),
                                    )

                                    style_fig(fig, height=600, legend_title=None)
                                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.25)")
                                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.25)")
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"eda_target_num_{selected_num_target}",
                                    )

                    else:
                        st.info("No numerical features are available.")

                else:

                    if plain_categorical_features:

                        target_cat_cols = st.columns(2)

                        for i, selected_cat_target in enumerate(plain_categorical_features):

                            with target_cat_cols[i % 2]:
                                with st.container(border=True):

                                    grouped = (
                                        hd_eda
                                        .groupby(
                                            [
                                                selected_cat_target,
                                                "Heart Disease Status",
                                            ]
                                        )
                                        .size()
                                        .reset_index(name="Count")
                                    )
                                    grouped["Total"] = grouped.groupby(
                                        selected_cat_target
                                    )["Count"].transform("sum")
                                    grouped["Percentage"] = (
                                        grouped["Count"] / grouped["Total"] * 100
                                    )

                                    level_order = (
                                        hd_eda[selected_cat_target]
                                        .value_counts()
                                        .index
                                        .tolist()
                                    )

                                    fig = px.bar(
                                        grouped,
                                        x=selected_cat_target,
                                        y="Count",
                                        color="Heart Disease Status",
                                        barmode="group",
                                        category_orders={
                                            selected_cat_target: level_order
                                        },
                                        color_discrete_map={
                                            "No": COLOR_NO,
                                            "Yes": COLOR_YES,
                                        },
                                        custom_data=["Percentage"],
                                    )

                                    fig.update_traces(
                                        hovertemplate=(
                                            f"<b>{selected_cat_target}</b>: %{{x}}<br>"
                                            "Status: %{fullData.name}<br>"
                                            "Count: %{y}<br>"
                                            "Percentage: %{customdata[0]:.1f}%"
                                            "<extra></extra>"
                                        )
                                    )

                                    fig.update_layout(
                                        title=(
                                            f"{selected_cat_target} vs "
                                            "Heart Disease Status"
                                        ),
                                        xaxis_title=selected_cat_target,
                                        yaxis_title="Count",
                                        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            valign="middle",
        ),
                                        margin=dict(l=70, r=70, t=70, b=150),
                                    )

                                    style_fig(fig, height=600, legend_title=None)
                                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.25)")
                                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.25)")
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"eda_target_cat_{selected_cat_target}",
                                    )

                    else:
                        st.info("No categorical features are available.")

            # ========================================================
            # PAGE 3 — FEATURE RELATIONSHIPS (Correlation Heatmaps)
            # ========================================================
            elif st.session_state.eda_view == "Relationships":

                _eda_back_button()

                st.markdown("### 🔗 Feature Relationships")

                (
                    X_train_num_rel,
                    X_train_ohe_rel,
                    ohe_cols_rel,
                ) = get_train_encoded_for_feature_corr()

                if X_train_num_rel is None:

                    st.info(
                        "Could not rebuild the encoded train set needed "
                        "for the correlation heatmaps."
                    )

                else:

                    if X_train_num_rel.shape[1] >= 2:

                        num_corr_matrix = X_train_num_rel.corr()

                        fig = go.Figure(
                            go.Heatmap(
                                z=num_corr_matrix.values,
                                x=num_corr_matrix.columns,
                                y=num_corr_matrix.columns,
                                colorscale=COLOR_SCALE_NOTEBOOK_COOLWARM,
                                text=num_corr_matrix.values,
                                texttemplate="%{text:.2f}",
                                textfont=dict(size=10),
                                hovertemplate=(
                                    "<b>%{y}</b> vs <b>%{x}</b><br>"
                                    "r: %{z:.3f}<extra></extra>"
                                ),
                                colorbar=dict(
                                    title=dict(
                                        text="r",
                                        side="top",
                                    ),
                                    thickness=22,
                                    len=0.88,
                                    x=1.02,
                                    xpad=8,
                                    tickmode="auto",
                                    nticks=6,
                                    tickformat=".2f",
                                    ticks="outside",
                                    ticklen=4,
                                    tickfont=dict(size=12),
                                ),
                            )
                        )

                        fig.update_layout(
                            title=(
                                "Correlation Heatmap of Numeric "
                                "Features"
                            ),
                            xaxis=dict(tickangle=-90),
                            yaxis=dict(autorange="reversed"),
                        )
                        style_fig(
                            fig,
                            height=max(
                                500, 42 * len(num_corr_matrix) + 120
                            ),
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "responsive": True,
                                "displayModeBar": False,
                            },
                        )

                    else:
                        st.info(
                            "At least two numerical features are "
                            "required."
                        )

    with corr_tab:

        st.subheader("Correlation & Association Analysis")

        if hd_eda is None:

            st.warning("Dataset not available.")

        elif "Heart Disease Status" not in hd_eda.columns:

            st.warning("Heart Disease Status column not found.")

        else:


            # =================================================
            # Prepare Data
            #
            # IMPORTANT:
            # This follows the notebook flow exactly:
            #
            # 1. Start from the raw dataset
            # 2. Remove duplicates
            # 3. Identify numeric/categorical columns
            # 4. Impute missing values on the WHOLE dataset
            # 5. Encode the target
            # 6. One-hot encode categorical features on the
            #    WHOLE dataset (drop='first')
            # 7. Separate X and y
            # 8. Train/test split:
            #       test_size=0.3
            #       random_state=400
            #       stratify=y
            # 9. Calculate Pearson / Chi-square / MI on
            #    X_train_enc and y_train BEFORE SMOTE
            #
            # Do NOT use the model preprocessors here. The
            # notebook creates hd_enc independently for this
            # analysis, so the app does the same.
            # =================================================

            @st.cache_data
            def get_notebook_relationship_data(dataframe):

                hd = dataframe.copy()

                # ---------------------------------------------
                # Step 1 — Check & Handle Duplicate Records
                # Same as notebook Cell 8
                # ---------------------------------------------
                if hd.duplicated().sum() > 0:
                    hd = hd.drop_duplicates().reset_index(drop=True)

                # ---------------------------------------------
                # Step 2 — Separate Numerical and Categorical
                # Columns
                # Same as notebook Cell 10
                # ---------------------------------------------
                notebook_numeric_features = hd.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()

                notebook_categorical_features = hd.select_dtypes(
                    include=["object"]
                ).columns.tolist()

                # ---------------------------------------------
                # Step 3 — Missing-value handling
                # Same as notebook Cell 25
                # ---------------------------------------------
                binary_yn = [
                    "Smoking",
                    "Family Heart Disease",
                    "Diabetes",
                    "High Blood Pressure",
                    "Low HDL Cholesterol",
                    "High LDL Cholesterol"
                ]

                ordinal_features = [
                    "Exercise Habits",
                    "Stress Level",
                    "Sugar Consumption"
                ]

                mode_impute_cols = (
                    ["Gender"]
                    + binary_yn
                    + ordinal_features
                )

                # Numeric Features — Median imputation
                median_imputer = SimpleImputer(
                    strategy="median"
                )
                hd[notebook_numeric_features] = (
                    median_imputer.fit_transform(
                        hd[notebook_numeric_features]
                    )
                )

                # Gender / Binary / Ordinal — Most-frequent
                mode_imputer = SimpleImputer(
                    strategy="most_frequent"
                )
                hd[mode_impute_cols] = (
                    mode_imputer.fit_transform(
                        hd[mode_impute_cols]
                    )
                )

                # Alcohol Consumption — Constant "None"
                constant_imputer = SimpleImputer(
                    strategy="constant",
                    fill_value="None"
                )
                hd[["Alcohol Consumption"]] = (
                    constant_imputer.fit_transform(
                        hd[["Alcohol Consumption"]]
                    )
                )

                # ---------------------------------------------
                # Step 4 — Encode target
                # Same as notebook Cell 27
                # ---------------------------------------------
                hd["Heart Disease Status"] = hd[
                    "Heart Disease Status"
                ].map({
                    "Yes": 1,
                    "No": 0
                })

                # Keep only valid target rows
                valid_target = (
                    hd["Heart Disease Status"].notna()
                    & hd["Heart Disease Status"].isin([0, 1])
                )

                hd = hd.loc[valid_target].copy()
                hd["Heart Disease Status"] = (
                    hd["Heart Disease Status"].astype(int)
                )

                if hd["Heart Disease Status"].nunique() < 2:
                    return None

                # ---------------------------------------------
                # Step 5 — One-Hot Encode categorical features
                # Same as notebook Cell 27
                # ---------------------------------------------
                categorical_to_encode = (
                    ["Gender"]
                    + binary_yn
                    + ["Alcohol Consumption"]
                    + ordinal_features
                )

                ohe = OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int
                )

                ohe.fit(hd[categorical_to_encode])

                hd_ohe = pd.DataFrame(
                    ohe.transform(
                        hd[categorical_to_encode]
                    ),
                    columns=ohe.get_feature_names_out(
                        categorical_to_encode
                    ),
                    index=hd.index
                )

                hd_enc = pd.concat(
                    [
                        hd.drop(
                            columns=categorical_to_encode
                        ),
                        hd_ohe
                    ],
                    axis=1
                )

                # ---------------------------------------------
                # Step 6 — Separate Features and Target
                # Same as notebook Cell 29
                # ---------------------------------------------
                X = hd_enc.drop(
                    columns=["Heart Disease Status"]
                )

                y = hd_enc["Heart Disease Status"]

                # ---------------------------------------------
                # Step 7 — Train/Test Split
                # EXACT notebook settings
                # ---------------------------------------------
                X_train_enc, X_test_enc, y_train, y_test = (
                    train_test_split(
                        X,
                        y,
                        test_size=0.30,
                        random_state=400,
                        stratify=y
                    )
                )

                return {
                    "X_train_enc": X_train_enc,
                    "X_test_enc": X_test_enc,
                    "y_train": y_train,
                    "y_test": y_test,
                    "numeric_features": notebook_numeric_features,
                    "categorical_features": notebook_categorical_features,
                    "current_categorical_cols": [
                        c for c in X_train_enc.columns
                        if c not in notebook_numeric_features
                    ]
                }

            relationship_data = get_notebook_relationship_data(hd_eda)

            if relationship_data is None:

                st.error(
                    "The notebook-style preprocessing could not "
                    "create a valid training dataset."
                )

            else:

                X_train_enc = relationship_data["X_train_enc"]
                y_train = relationship_data["y_train"]
                notebook_numeric_features = relationship_data[
                    "numeric_features"
                ]
                current_categorical_cols = relationship_data[
                    "current_categorical_cols"
                ]

            corr_cols = st.columns(2)

            with corr_cols[0]:
                with st.container(border=True):
                    # =================================================
                    # Mutual Information
                    # Same as notebook Cell 33
                    # =================================================

                        st.markdown("### Mutual Information")

                        # IMPORTANT:
                        # Do not add discrete_features here.
                        # The notebook uses exactly:
                        #
                        # mutual_info_classif(
                        #     X_train_enc,
                        #     y_train,
                        #     random_state=400
                        # )
                        #
                        # so the app intentionally does the same.
                        mi_scores = mutual_info_classif(
                            X_train_enc,
                            y_train,
                            random_state=400
                        )

                        mi_table = pd.Series(
                            mi_scores,
                            index=X_train_enc.columns
                        ).sort_values(
                            ascending=False
                        )

                        mi_table = mi_table.round(4)

                        mi_sorted = mi_table.sort_values(
                            ascending=True
                        )

                        fig = go.Figure(
                            go.Bar(
                                x=mi_sorted.values,
                                y=mi_sorted.index,
                                orientation="h",
                                marker=dict(
                                    color=mi_sorted.values,
                                    colorscale=(
                                        COLOR_SCALE_BLUE_SEQUENTIAL
                                    ),
                                    colorbar=dict(title="MI"),
                                ),
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "Mutual information: %{x:.4f}"
                                    "<extra></extra>"
                                ),
                            )
                        )

                        fig.update_layout(
                            title=(
                                "Mutual Information — Feature vs Target"
                            ),
                            xaxis_title=(
                                "Mutual Information Score"
                            ),
                            yaxis_title="Feature",
                        )

                        style_fig(
                            fig,
                            height=max(
                                420,
                                22 * len(mi_sorted) + 100
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "responsive": True,
                                "displayModeBar": False
                            },
                            key="corr_mutual_info_chart",
                        )

                        mi_display = pd.DataFrame({
                            "Feature": mi_table.index,
                            "Mutual Information": mi_table.values
                        })

                        st.dataframe(
                            mi_display.style.format({
                                "Mutual Information": "{:.4f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )


            with corr_cols[1]:
                with st.container(border=True):
                    # =================================================
                    # Chi-Square Test
                    # Same as notebook Cell 32
                    # =================================================

                        st.markdown(
                            "### Chi-Square Test of Independence"
                        )

                        chi_square_results = []

                        for col in current_categorical_cols:

                            contingency = pd.crosstab(
                                X_train_enc[col],
                                y_train
                            )

                            chi2, p, dof, expected = (
                                chi2_contingency(contingency)
                            )

                            chi_square_results.append({
                                "Feature": col,
                                "Chi2 Statistic": round(
                                    chi2, 3
                                ),
                                "p-value": round(
                                    p, 4
                                )
                            })

                        chi_square_df = (
                            pd.DataFrame(
                                chi_square_results
                            )
                            .sort_values(
                                by="p-value"
                            )
                            .reset_index(drop=True)
                        )

                        chi_sorted = chi_square_df.sort_values(
                            "p-value",
                            ascending=True
                        )

                        fig = go.Figure(
                            go.Bar(
                                x=chi_sorted["p-value"],
                                y=chi_sorted["Feature"],
                                orientation="h",
                                marker_color=COLOR_YES,
                                customdata=chi_sorted[
                                    "Chi2 Statistic"
                                ],
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "p-value: %{x:.4f}<br>"
                                    "Chi2 statistic: %{customdata:.3f}"
                                    "<extra></extra>"
                                ),
                            )
                        )

                        fig.add_vline(
                            x=0.05,
                            line_width=1.5,
                            line_dash="dash",
                            line_color=(
                                "rgba(226,232,240,0.6)"
                            ),
                            annotation_text="α = 0.05",
                            annotation_position="top",
                        )

                        max_p = chi_square_df[
                            "p-value"
                        ].max()

                        fig.update_layout(
                            title=(
                                "Chi-square Test: Categorical "
                                "Features vs Heart Disease Status "
                            ),
                            xaxis_title="p-value",
                            yaxis=dict(
                                title="Feature",
                                autorange="reversed",
                            ),
                            xaxis_range=[
                                0,
                                max(max_p * 1.15, 0.06)
                            ],
                        )

                        style_fig(
                            fig,
                            height=max(
                                420,
                                40 * len(chi_sorted) + 100
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "responsive": True,
                                "displayModeBar": False
                            },
                            key="corr_chi_square_chart",
                        )

                        chi_square_df[
                            "Significant (α = 0.05)"
                        ] = (
                            chi_square_df["p-value"] < 0.05
                        )

                        st.dataframe(
                            chi_square_df.style.format({
                                "Chi2 Statistic": "{:.3f}",
                                "p-value": "{:.4f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )



            with st.container(border=True):
                # =================================================
                # Pearson Correlation
                # Same as notebook Cell 31
                # =================================================


                        st.markdown("### Pearson Correlation")

                        train_with_target = X_train_enc.copy()

                        train_with_target[
                            "Heart Disease Status"
                        ] = y_train

                        target_corr = (
                            train_with_target
                            .corr(numeric_only=True)[
                                "Heart Disease Status"
                            ]
                            .drop("Heart Disease Status")
                        )

                        target_corr = target_corr.sort_values()

                        fig = go.Figure(
                            go.Bar(
                                x=target_corr.values,
                                y=target_corr.index,
                                orientation="h",
                                marker=dict(
                                    color=target_corr.values,
                                    colorscale=COLOR_SCALE_DIVERGING,
                                    cmid=0,
                                    colorbar=dict(title="r"),
                                ),
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "Pearson r: %{x:.4f}"
                                    "<extra></extra>"
                                ),
                            )
                        )

                        fig.add_vline(
                            x=0,
                            line_width=1,
                            line_color=(
                                "rgba(226,232,240,0.6)"
                            )
                        )

                        fig.update_layout(
                            title=(
                                "Pearson Correlation of Each "
                                "Feature with Heart Disease Status"
                            ),
                            xaxis_title="Correlation Coefficient",
                            yaxis=dict(
                                title="Encoded Feature",
                                autorange="reversed",
                            ),
                        )

                        style_fig(
                            fig,
                            height=max(
                                420,
                                22 * len(target_corr) + 100
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "responsive": True,
                                "displayModeBar": False
                            },
                            key="corr_pearson_chart",
                        )

                        pearson_df = pd.DataFrame({
                            "Feature": target_corr.index,
                            "Pearson Correlation": target_corr.values
                        })

                        pearson_df[
                            "Absolute Correlation"
                        ] = pearson_df[
                            "Pearson Correlation"
                        ].abs()

                        # Keep the table in the same top-to-bottom order as
                        # the chart above (target_corr, displayed with a
                        # reversed y-axis) rather than sorting by magnitude,
                        # so the two are easy to line up and compare.
                        pearson_df = pearson_df.reset_index(drop=True)

                        st.dataframe(
                            pearson_df.style.format({
                                "Pearson Correlation": "{:.4f}",
                                "Absolute Correlation": "{:.4f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )



    # ========================================================
    # MODEL EVALUATION
    # ========================================================
    with eval_tab:

        st.subheader("Performance Metrics")

        if eval_results is None:

            st.warning(
                "No evaluation results found "
                "(`model_evaluation_results.csv`)."
            )

        else:

            metric_order = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Balanced Accuracy",
                "MCC",
                "ROC-AUC",
                "PR-AUC"
            ]

            available_metrics = [
                metric for metric in metric_order
                if metric in eval_results.columns
            ]

            if not available_metrics:

                st.error(
                    "No recognised evaluation metrics "
                    "were found in the evaluation file."
                )

            else:

                results_df = eval_results[
                    available_metrics
                ].copy()

                st.dataframe(
                    results_df.style.format("{:.3f}"),
                    use_container_width=True
                )

                # =================================================
                # Accuracy Chart
                # =================================================
                if "Accuracy" in results_df.columns:

                    st.subheader("Accuracy Comparison")

                    # Match the notebook: sort by accuracy (highest
                    # first) and colour bars by rank/position, not by
                    # fixed model identity.
                    accuracy_ranked = results_df[["Accuracy"]].sort_values(
                        by="Accuracy", ascending=False
                    )

                    model_names = (
                        accuracy_ranked.index.astype(str).tolist()
                    )

                    accuracy_values = accuracy_ranked["Accuracy"].tolist()

                    RANK_COLORS = ["#4C72B0", "#55A868", "#C44E52"]
                    bar_colors = [
                        RANK_COLORS[i % len(RANK_COLORS)]
                        for i in range(len(model_names))
                    ]

                    # Y-axis always starts at 0 so bar heights are
                    # comparable at a glance without exaggerating gaps.
                    max_acc = max(accuracy_values)
                    y_lower = 0
                    y_upper = min(1, max_acc + 0.08)

                    fig = go.Figure(
                        go.Bar(
                            x=model_names,
                            y=accuracy_values,
                            marker_color=bar_colors,
                            text=[f"{v:.3f}" for v in accuracy_values],
                            textposition="outside",
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Accuracy: %{y:.4f}<extra></extra>"
                            ),
                        )
                    )

                    fig.update_layout(
                        title="Model Accuracy Comparison",
                        xaxis_title="Model",
                        yaxis_title="Accuracy",
                        yaxis_range=[y_lower, y_upper],
                    )
                    style_fig(fig, height=420)
                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.25)")
                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.25)")
                    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})

                # =================================================
                # ALL METRICS COMPARISON CHART
                # =================================================
                st.subheader("All Metrics Comparison")

                metrics_to_plot = [
                    "Accuracy",
                    "ROC-AUC",
                    "PR-AUC",
                    "MCC",
                    "Recall",
                    "F1-Score"
                ]

                # Only keep metrics that exist in the CSV
                metrics_to_plot = [
                    metric
                    for metric in metrics_to_plot
                    if metric in results_df.columns
                ]

                if metrics_to_plot:

                    plot_df = results_df[
                        metrics_to_plot
                    ].copy()

                    # -------------------------------------------------
                    # Fix long model names
                    # -------------------------------------------------
                    label_mapping = {
                        "Random Forest": "Random Forest",
                        "XGBoost": "XGBoost",
                        "Ensemble (Voting)": "Ensemble (Voting)",
                        "GaussianNB (Baseline)": "GaussianNB (Baseline)",
                        "Naive Bayes (Baseline)": "Naive Bayes (Baseline)"
                    }

                    plot_df.index = [
                        label_mapping.get(str(x), str(x))
                        for x in plot_df.index
                    ]

                    long_df = plot_df.reset_index().melt(
                        id_vars="index",
                        var_name="Metric",
                        value_name="Score",
                    ).rename(columns={"index": "Model"})

                    fig = px.bar(
                        long_df,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        color_discrete_map=METRIC_COLORS_SET2,
                    )

                    fig.update_traces(
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "%{fullData.name}: %{y:.4f}<extra></extra>"
                        )
                    )

                    fig.update_layout(
                        title="Model Comparison — All Evaluation Metrics",
                        xaxis_title="Model",
                        yaxis_title="Score",
                        yaxis_range=[
                            max(0, plot_df.min().min() - 0.05),
                            min(1.0, plot_df.max().max() + 0.10)
                        ],
                    )
                    style_fig(fig, height=460, legend_title="Metrics")
                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.25)")
                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.25)")
                    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})

                else:

                    st.warning(
                        "No metrics are available for the "
                        "all-metrics comparison chart."
                    )

                # =================================================
                # Confusion Matrices
                # =================================================
                st.subheader("Confusion Matrices")

                confusion_models = [
                    model_name
                    for model_name in [
                        "GaussianNB (Baseline)",
                        "Random Forest",
                        "XGBoost",
                    ]
                    if model_name in models
                ]

                if not confusion_models:

                    st.info(
                        "No trained models are available to build "
                        "confusion matrices."
                    )

                else:

                    X_test_enc_cm, y_test_cm = get_test_split_encoded()

                    if X_test_enc_cm is None or y_test_cm is None:

                        st.info(
                            "Could not rebuild the encoded test set "
                            "needed for the confusion matrices."
                        )

                    else:
                        cm_cols = st.columns(len(confusion_models))

                        for cm_col, model_name in zip(
                            cm_cols, confusion_models
                        ):

                            with cm_col:

                                try:

                                    cm_model = models[model_name]
                                    y_pred_cm = cm_model.predict(
                                        X_test_enc_cm
                                    )

                                    cm = confusion_matrix(
                                        y_test_cm, y_pred_cm
                                    )

                                    cm_labels = ["No", "Yes"]

                                    fig = go.Figure(
                                        go.Heatmap(
                                            z=cm,
                                            x=[
                                                f"Predicted {lbl}"
                                                for lbl in cm_labels
                                            ],
                                            y=[
                                                f"Actual {lbl}"
                                                for lbl in cm_labels
                                            ],
                                            colorscale=(
                                                COLOR_SCALE_BLUE_SEQUENTIAL
                                            ),
                                            text=cm,
                                            texttemplate="%{text}",
                                            textfont=dict(size=16),
                                            showscale=False,
                                            hovertemplate=(
                                                "<b>%{y}</b> — %{x}<br>"
                                                "Count: %{z}<extra></extra>"
                                            ),
                                        )
                                    )

                                    fig.update_layout(
                                        title=(
                                            f"{model_name}"
                                        ),
                                        yaxis=dict(autorange="reversed"),
                                        margin=dict(
                                            l=60, r=20, t=50, b=40
                                        ),
                                    )
                                    style_fig(fig, height=340)
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        config={
                                            "responsive": True,
                                            "displayModeBar": False,
                                        },
                                    )

                                except Exception as e:

                                    st.warning(
                                        f"Could not build the confusion "
                                        f"matrix for {model_name}: {e}"
                                    )

                # =================================================
                # Metric Explanation
                # =================================================
                st.markdown("### Metric Explanation")

                st.write(
                    "**Accuracy:** Overall proportion of correctly classified observations."
                )

                st.write(
                    "**Precision:** Proportion of predicted positive cases that are actually positive."
                )

                st.write(
                    "**Recall:** Proportion of actual positive cases correctly identified."
                )

                st.write(
                    "**F1-Score:** Harmonic mean of Precision and Recall."
                )

                st.write(
                    "**Balanced Accuracy:** Average recall across the two classes."
                )

                st.write(
                    "**MCC:** Measures the quality of binary classifications while considering all four confusion-matrix categories."
                )

                st.write(
                    "**ROC-AUC:** Measures how well the model distinguishes between the two classes across classification thresholds."
                )

                st.write(
                    "**PR-AUC:** Summarises the Precision-Recall relationship across classification thresholds."
                )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================
    with feat_tab:

        st.subheader("Top 10 Permutation Feature Importances")

        importance_models = [
            model_name
            for model_name in [
                "GaussianNB (Baseline)",
                "Random Forest",
                "XGBoost"
            ]
            if model_name in models
        ]

        if not importance_models:

            st.info(
                "No tree-based models are available "
                "for feature importance."
            )

        else:

            X_test_enc, y_test = get_test_split_encoded()

            if X_test_enc is None or y_test is None:

                st.info(
                    "Could not rebuild the encoded test set "
                    "needed to compute permutation importance."
                )

            else:

                feat_imp_cols = st.columns(2)

                for i, model_choice in enumerate(importance_models):

                    with feat_imp_cols[i % 2]:
                        with st.container(border=True):

                            selected_model_object = models[model_choice]

                            try:

                                # ------------------------------------------------
                                # Permutation Importance (Test Set, ROC-AUC)
                                # ------------------------------------------------
                                with st.spinner(
                                    f"Computing permutation importance for "
                                    f"{model_choice}..."
                                ):

                                    perm_result = permutation_importance(
                                        selected_model_object,
                                        X_test_enc,
                                        y_test,
                                        scoring="roc_auc",
                                        n_repeats=10,
                                        random_state=400,
                                        n_jobs=-1
                                    )

                                importances = pd.Series(
                                    perm_result.importances_mean,
                                    index=X_test_enc.columns
                                ).sort_values(ascending=False)

                                top_importances = importances.head(10).sort_values(ascending=True)

                                bar_color = MODEL_COLORS.get(model_choice, COLOR_NO)

                                fig = go.Figure(
                                    go.Bar(
                                        x=top_importances.values,
                                        y=top_importances.index,
                                        orientation="h",
                                        marker_color=bar_color,
                                        hovertemplate=(
                                            "<b>%{y}</b><br>"
                                            "Mean decrease in ROC-AUC: %{x:.4f}"
                                            "<extra></extra>"
                                        ),
                                    )
                                )

                                fig.update_layout(
                                    title=(
                                        f"{model_choice} — Top 10 Permutation Importances\n"
                                        f" (Test Set)"
                                    ),
                                    xaxis_title="Mean decrease in ROC-AUC",
                                    yaxis_title="Feature",
                                )
                                style_fig(fig, height=460)
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    config={"responsive": True, "displayModeBar": False},
                                    key=f"feat_importance_chart_{model_choice}",
                                )

                                st.dataframe(
                                    top_importances.rename(
                                        "Mean decrease in ROC-AUC"
                                    ).to_frame(),
                                    use_container_width=True
                                )

                            except Exception as e:

                                st.error(
                                    f"Could not compute permutation importance "
                                    f"for {model_choice}: {e}"
                                )