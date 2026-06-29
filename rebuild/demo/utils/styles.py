"""Design tokens and CSS - Chase-inspired light theme, dark forest green accent.

Extracted from Chase's MDS design system (data-brand=cmb):
  Font:            Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif
  Body:            0.875rem / 1.25rem line-height / weight 400
  Label:           0.75rem / 1rem / weight 600
  Button:          1rem / 1.5rem / weight 600, side-padding 16px
  Border-radius:   xSmall 4px · small 8px · medium 12px · large 24px
  Border-weight:   1px default · 2px active
  Active field:    2px bottom border at brand color
  Transitions:     opacity ease-in 0.2s  (Chase body transition)
  Gain color:      #398100 (Chase)  → adapted to #16a34a
  Loss color:      #b80009 (Chase)  → kept as #dc2626

Brand: Chase blue #005eb8 replaced with dark forest green #0A5C36.
Background palette mirrors Chase's light system.
"""

from __future__ import annotations

import streamlit as st

# ── Brand accent ──────────────────────────────────────────────────────────────

ACCENT: str = "#0A5C36"                        # dark forest green (replaces Chase #005eb8)
FILL_COLOR: str = "rgba(10, 92, 54, 0.05)"    # accent at 5% opacity for chart fills

# ── Strategy colours and labels ───────────────────────────────────────────────

STRATEGY_COLORS: dict[str, str] = {
    "profit_scoring":  ACCENT,     # brand dark green - my model
    "default_scoring": "#475569",  # dark slate - PD baseline
    "grade_only":      "#94a3b8",  # medium gray - LC benchmark
    "invest_all":      "#cbd5e1",  # light slate - passive baseline
}

STRATEGY_LABELS: dict[str, str] = {
    "profit_scoring":  "Profit Scoring (ANR model)",
    "default_scoring": "Default Scoring (PD model)",
    "grade_only":      "LC Grade Only (benchmark)",
    "invest_all":      "Invest-All (passive)",
}

# ── Shared Plotly layout (light, Chase palette) ───────────────────────────────

PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(
        family="Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif",
        color="#434f5b",
        size=11,
    ),
    xaxis=dict(
        gridcolor="#e2e4e5",
        linecolor="#d3d9de",
        tickformat=".0%",
        zeroline=False,
        title_font=dict(color="#434f5b", size=11),
        tickfont=dict(color="#565a5d"),
    ),
    yaxis=dict(
        gridcolor="#e2e4e5",
        linecolor="#d3d9de",
        tickformat=".1%",
        zeroline=False,
        title_font=dict(color="#434f5b", size=11),
        tickfont=dict(color="#565a5d"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        borderwidth=0,
        font=dict(color="#434f5b", size=11),
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=55, r=20, t=40, b=50),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#e2e4e5",
        font=dict(
            color="#0f171f",
            family="Open Sans, Helvetica Neue, sans-serif",
            size=12,
        ),
    ),
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# All Chase design tokens applied; blue (#005eb8) replaced with #0A5C36.

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ─── Streamlit chrome removal ─── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ─── App shell ─── */
/* Chase body: gray #f5f7fa page bg, Open Sans, antialiased */
.stApp {
    background-color: #f5f7fa;
    font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    transition: opacity ease-in 0.2s;
    color: #0f171f;
}
.block-container {
    box-sizing: border-box !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1240px !important;
    background: transparent !important;
}

/* ─── Chase card ─── */
/* Chase: white cards on gray page, border #e2e4e5, radius 8px (small), subtle shadow */
.chase-card {
    box-sizing: border-box;
    background: #ffffff;
    border: 1px solid #e2e4e5;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(15, 23, 31, 0.06);
    overflow: hidden;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.chase-card:hover {
    box-shadow: 0 4px 14px rgba(15, 23, 31, 0.10);
    transform: translateY(-1px);
}
.chase-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    border-bottom: 1px solid #e2e4e5;
    background: #ffffff;
}
.chase-card-title {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    color: #0f171f;
    line-height: 1.25rem;
    margin: 0;
}
.chase-card-subtitle {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 400;
    color: #565a5d;
}
.chase-card-body {
    padding: 20px;
    background: #ffffff;
}

/* ─── Chase account name (bold link-style) ─── */
.chase-account-name {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    color: #0A5C36;
    letter-spacing: 0.02em;
    margin-bottom: 10px;
}

/* ─── Chase metrics row ─── */
/* Chase: large light-weight number, dotted-underline label below */
.chase-metrics-row {
    display: flex;
    align-items: flex-start;
    gap: 48px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.chase-metric-value {
    font-family: 'Open Sans', sans-serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #0f171f;
    line-height: 2rem;
    letter-spacing: -0.02em;
}
.chase-metric-value.positive { color: #0A5C36; }
.chase-metric-value.negative { color: #c0000a; }
.chase-metric-label {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 400;
    color: #434f5b;
    margin-top: 2px;
    border-bottom: 1px dotted #d3d9de;
    display: inline-block;
}

/* ─── Chase button pair ─── */
.chase-btn-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.chase-btn-primary {
    display: inline-block;
    background: #0A5C36;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    line-height: 1.5rem;
    white-space: nowrap;
    transition: background-color 0.15s ease-in;
    text-decoration: none;
}
.chase-btn-primary:hover { background: #0D7240; }
.chase-btn-outline {
    display: inline-block;
    background: #ffffff;
    color: #0f171f;
    border: 1px solid #d3d9de;
    border-radius: 4px;
    padding: 8px 16px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    line-height: 1.5rem;
    white-space: nowrap;
    transition: border-color 0.15s ease-in, color 0.15s ease-in;
}
.chase-btn-outline:hover { border-color: #0A5C36; color: #0A5C36; }

/* ─── Chase alert / notification strip ─── */
.chase-alert-success {
    background: rgba(10, 92, 54, 0.04);
    border: 1px solid rgba(10, 92, 54, 0.18);
    border-radius: 4px;
    padding: 10px 14px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 16px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    color: #434f5b;
    line-height: 1.5;
}
.chase-alert-error {
    background: rgba(192, 0, 10, 0.04);
    border: 1px solid rgba(192, 0, 10, 0.18);
    border-radius: 4px;
    padding: 10px 14px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 16px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    color: #434f5b;
    line-height: 1.5;
}

/* ─── Chase tab navigation ─── */
.chase-tab-nav {
    display: flex;
    border-bottom: 1px solid #e2e4e5;
    margin: 0 0 24px 0;
    background: #ffffff;
    padding: 0 20px;
    border-radius: 8px 8px 0 0;
    border: 1px solid #e2e4e5;
    border-bottom: none;
}
.chase-tab {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 400;
    color: #434f5b;
    padding: 12px 20px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    white-space: nowrap;
}
.chase-tab-active {
    font-weight: 600;
    color: #0f171f;
    border-bottom: 3px solid #0A5C36;
}
.chase-tab-bar {
    background: #ffffff;
    border: 1px solid #e2e4e5;
    border-top: none;
    border-radius: 0 0 8px 8px;
    height: 4px;
    margin-bottom: 24px;
}

/* ─── st.container(border=True) → Chase card (matches .chase-card) ─── */
/* Padding lives on the wrapper itself (border-box + overflow:hidden) so it can
   never add to the card's width and overflow its column. */
[data-testid="stVerticalBlockBorderWrapper"] {
    box-sizing: border-box !important;
    border: 1px solid #e2e4e5 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 4px rgba(15, 23, 31, 0.06) !important;
    background: #ffffff !important;
    overflow: hidden !important;
    padding: 4px 20px 18px 20px !important;
    margin-bottom: 16px !important;
}

/* ─── st.tabs() → Chase tab nav ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e4e5 !important;
    gap: 0 !important;
    padding: 0 8px !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Open Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    color: #434f5b !important;
    padding: 12px 20px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    font-weight: 600 !important;
    color: #0f171f !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #0A5C36 !important;
    height: 3px !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding: 20px 0 !important;
}

/* ─── Buttons: Chase secondary style (outline, green on white) ─── */
/* The global `p { color }` rule below would otherwise recolor button labels. */
.stButton > button,
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #0A5C36 !important;
    border: 1px solid #0A5C36 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: background-color 0.15s ease, color 0.15s ease !important;
}
.stButton > button p,
.stButton > button div,
.stButton > button span,
.stButton > button label {
    color: #0A5C36 !important;
    transition: color 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #0A5C36 !important;
    border-color: #0A5C36 !important;
}
.stButton > button:hover p,
.stButton > button:hover div,
.stButton > button:hover span,
.stButton > button:hover label {
    color: #ffffff !important;
}
.stButton > button:active { background-color: #094F30 !important; }

/* Slider/radio/checkbox colors come from theme primaryColor (#0A5C36).
   No manual overrides needed once .streamlit/config.toml is loaded. */

/* ─── Select dropdown active option ─── */
[data-baseweb="select"] [aria-selected="true"] {
    background-color: rgba(10, 92, 54, 0.08) !important;
    color: #0A5C36 !important;
}

/* ─── About panel ─── */
.chase-about-panel {
    box-sizing: border-box;
    background: #ffffff;
    border: 1px solid #e2e4e5;
    border-left: 4px solid #0A5C36;
    border-radius: 0 8px 8px 0;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(15, 23, 31, 0.06);
    font-family: 'Open Sans', sans-serif;
}
.chase-about-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: #0f171f;
    margin-bottom: 10px;
    line-height: 1.5rem;
}
.chase-about-body {
    font-size: 0.875rem;
    color: #434f5b;
    line-height: 1.65;
}
.chase-about-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.chase-about-tag {
    display: inline-block;
    background: rgba(10, 92, 54, 0.06);
    color: #0A5C36;
    border: 1px solid rgba(10, 92, 54, 0.2);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ─── Section header inside container(border=True) ─── */
/* Negative margins make the divider span the full card width (like .chase-card-header). */
.chase-section-header {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    color: #0f171f;
    margin: -4px -20px 16px -20px;
    padding: 14px 20px 12px 20px;
    border-bottom: 1px solid #e2e4e5;
    line-height: 1.25rem;
}

/* ─── Chase right sidebar widget ─── */
.chase-widget {
    box-sizing: border-box;
    background: #ffffff;
    border: 1px solid #e2e4e5;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 31, 0.04);
    overflow: hidden;
}
.chase-widget-header {
    padding: 12px 16px;
    border-bottom: 1px solid #e2e4e5;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    color: #0f171f;
    line-height: 1.25rem;
}
.chase-widget-body { padding: 4px 0; }
.chase-widget-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    border-bottom: 1px solid #f5f7fa;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    color: #434f5b;
    cursor: pointer;
    transition: background-color 0.12s ease-in;
}
.chase-widget-row:last-child { border-bottom: none; }
.chase-widget-row:hover { background: #f5f7fa; color: #0A5C36; }
.chase-widget-row-key { color: #434f5b; }
.chase-widget-row-val {
    font-weight: 600;
    color: #0f171f;
    font-size: 0.75rem;
}
.chase-widget-link {
    color: #0A5C36;
    font-size: 0.75rem;
    font-weight: 600;
    text-decoration: none;
    padding: 10px 16px;
    display: block;
    font-family: 'Open Sans', sans-serif;
    transition: color 0.12s ease-in;
}
.chase-widget-link:hover { text-decoration: underline; }

/* ─── Make Plotly chart look like a Chase card ─── */
[data-testid="stPlotlyChart"] {
    border: 1px solid #e2e4e5 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(15, 23, 31, 0.04) !important;
}

/* ─── Streamlit controls on light/gray bg ─── */
/* Radio and slider labels against gray background */
[data-testid="stRadio"],
[data-testid="stSlider"] {
    background: transparent;
}


/* ─── Sidebar ─── */
/* Chase: secondary bg #f5f7fa, border #e2e4e5 */
[data-testid="stSidebar"] {
    background-color: #f5f7fa !important;
    border-right: 1px solid #e2e4e5 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #f5f7fa !important;
}
[data-testid="stSidebarNavItems"] a {
    color: #434f5b !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    padding: 6px 12px !important;
    border-radius: 4px !important;   /* Chase xSmall */
    margin: 1px 8px !important;
    transition: background-color ease-in 0.15s !important;
}
[data-testid="stSidebarNavItems"] a:hover {
    color: #0f171f !important;
    background-color: #e2e4e5 !important;
}
[data-testid="stSidebarNavItems"] [aria-selected="true"],
[data-testid="stSidebarNavItems"] [aria-current="page"] {
    color: #0A5C36 !important;
    background-color: rgba(10, 92, 54, 0.06) !important;
    border-left: 2px solid #0A5C36 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebarNavSeparator"] {
    border-color: #e2e4e5 !important;
}

/* ─── Typography ─── */
/* Chase: chartTitleTextWeight 300, titleMediumHeavierTextSize 1.5rem weight 600 */
h1 {
    font-family: 'Open Sans', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 300 !important;
    color: #0f171f !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0 !important;
    line-height: 2rem !important;
}
h2 {
    font-family: 'Open Sans', sans-serif !important;
    font-size: 1.125rem !important;
    font-weight: 600 !important;
    color: #0f171f !important;
    letter-spacing: -0.005em !important;
    line-height: 1.75rem !important;
}
h3 {
    font-family: 'Open Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: #434f5b !important;
    line-height: 1.25rem !important;
}
/* Chase body medium: 0.875rem / 1.25rem / 400 */
p {
    font-family: 'Open Sans', sans-serif;
    color: #434f5b;
    font-size: 0.875rem;
    line-height: 1.25rem;
}
hr { border-color: #e2e4e5 !important; margin: 1.5rem 0 !important; }

/* ─── Buttons: Chase secondary (outline) - typography + sizing ─── */
/* Color/hover behaviour defined once in the consolidated block above. */
.stButton > button {
    border-radius: 4px !important;   /* Chase xSmall 4px */
    padding: 9px 14px !important;
    font-family: 'Open Sans', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
    line-height: 1.25rem !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    margin-bottom: 8px !important;
}
.stButton > button:focus {
    outline: 2px solid rgba(10, 92, 54, 0.35) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

/* ─── Sliders ─── */
/* Chase: label 0.75rem / 600 */
[data-testid="stSlider"] > label {
    font-size: 0.75rem !important;
    color: #434f5b !important;
    font-weight: 600 !important;
    font-family: 'Open Sans', sans-serif !important;
}

/* ─── Radio buttons ─── */
[data-testid="stRadio"] > label {
    font-size: 0.75rem !important;
    color: #434f5b !important;
    font-weight: 600 !important;
    font-family: 'Open Sans', sans-serif !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
    font-size: 0.875rem !important;
    color: #434f5b !important;
    font-family: 'Open Sans', sans-serif !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    color: #0A5C36 !important;
    font-weight: 600 !important;
}

/* ─── Select / number / text inputs ─── */
/* Chase: label 0.75rem / 600; input border 1px; active bottom-border 2px #0A5C36 */
[data-testid="stSelectbox"] > label,
[data-testid="stNumberInput"] > label,
[data-testid="stTextInput"] > label {
    font-size: 0.75rem !important;
    color: #434f5b !important;
    font-weight: 600 !important;
    font-family: 'Open Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stTextInput"] > div > div > input {
    background-color: #ffffff !important;
    border: 1px solid #d3d9de !important;
    border-radius: 4px !important;   /* Chase xSmall */
    color: #0f171f !important;
    font-size: 0.875rem !important;
    font-family: 'Open Sans', sans-serif !important;
    transition: border-color ease-in 0.2s !important;
}
[data-testid="stNumberInput"] > div > div > input:focus,
[data-testid="stTextInput"] > div > div > input:focus {
    border-bottom: 2px solid #0A5C36 !important;
    outline: none !important;
}
/* Disabled state from Chase: bg #f5f7fa, color #85888a, border #bdc0c2 */
[data-testid="stNumberInput"] > div > div > input:disabled,
[data-testid="stTextInput"] > div > div > input:disabled {
    background-color: #f5f7fa !important;
    color: #85888a !important;
    border-color: #bdc0c2 !important;
}

/* ─── Checkboxes ─── */
[data-testid="stCheckbox"] label {
    font-size: 0.875rem !important;
    color: #434f5b !important;
    font-family: 'Open Sans', sans-serif !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e4e5 !important;
    border-radius: 8px !important;   /* Chase small */
    overflow: hidden !important;
}

/* ─── Images ─── */
[data-testid="stImage"] img {
    border-radius: 4px;
}

/* ─── Caption / info ─── */
.stCaption, [data-testid="stCaption"] {
    color: #565a5d !important;
    font-size: 0.75rem !important;
    font-family: 'Open Sans', sans-serif !important;
}

/* ─── Expander ─── */
/* Chase MdsCustomAccordion: borderRadius 8px, borderColor #e2e4e5 */
[data-testid="stExpander"] {
    border: 1px solid #e2e4e5 !important;
    border-radius: 8px !important;
}

/* ─── Custom component classes ─────────────────────────────────────────────── */

/* Metric card - Chase: bg white, border #e2e4e5, radius 8px (small), padding 16px */
.lc-card {
    background: #ffffff;
    border: 1px solid #e2e4e5;
    border-radius: 8px;          /* Chase small */
    padding: 16px 20px;
    font-family: 'Open Sans', sans-serif;
    height: 100%;
    box-shadow: 0 1px 3px rgba(15, 23, 31, 0.04);
}
/* Chase: label 0.75rem / 600 / secondary color #434f5b */
.lc-card-label {
    font-size: 0.75rem;
    color: #434f5b;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
    line-height: 1rem;           /* Chase label line-height */
}
/* Chase: titleMediumHeavier 1.5rem / 2rem / 600 */
.lc-card-value {
    font-family: 'Open Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f171f;
    letter-spacing: -0.01em;
    line-height: 2rem;
    margin-bottom: 4px;
}
/* Chase: body 0.875rem / 1.25rem / 400 */
.lc-card-delta {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 400;
    margin-top: 4px;
    line-height: 1.25rem;
}

/* Colour utilities */
/* Chase gain #398100 adapted; loss #b80009 adapted */
.lc-positive { color: #16a34a !important; }
.lc-negative { color: #dc2626 !important; }
.lc-neutral  { color: #565a5d !important; }
.lc-purple   { color: #0A5C36 !important; }   /* accent = dark green, not purple */

/* Decision badge - green tint for approve, red tint for deny */
.lc-badge-approve {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(10, 92, 54, 0.06);
    color: #0A5C36;
    border: 1px solid rgba(10, 92, 54, 0.3);
    border-radius: 4px;           /* Chase xSmall */
    padding: 8px 20px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.lc-badge-deny {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(192, 0, 10, 0.05);
    color: #991b1b;
    border: 1px solid rgba(192, 0, 10, 0.25);
    border-radius: 4px;           /* Chase xSmall */
    padding: 8px 20px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Page title - Chase chartTitleTextSize 1.125rem / weight 300 (light) */
.lc-page-title {
    font-family: 'Open Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 300;
    color: #0f171f;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
    line-height: 1.75rem;
}
/* Chase body medium */
.lc-page-sub {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.875rem;
    color: #565a5d;
    margin-bottom: 28px;
    line-height: 1.25rem;
    max-width: 720px;
}

/* Section divider label - Chase: label 0.75rem / 600 / secondary */
.lc-section-label {
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    color: #434f5b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    padding-bottom: 8px;
    border-bottom: 1px solid #e2e4e5;
    margin-bottom: 14px;
    line-height: 1rem;
}

/* Vintage context block - Chase: secondary bg #f5f7fa, border #e2e4e5, radius 8px */
.lc-context {
    background: #f5f7fa;
    border: 1px solid #e2e4e5;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 12px;
    font-family: 'Open Sans', sans-serif;
    font-size: 0.75rem;
    color: #565a5d;
    line-height: 1.7;
}
.lc-context strong { color: #434f5b; font-weight: 600; }

/* Figure wrapper - Chase accordion style: border #e2e4e5, radius 8px */
.lc-figure {
    border: 1px solid #e2e4e5;
    border-radius: 8px;
    overflow: hidden;
    background: #ffffff;
    padding: 12px;
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def plotly_layout(**overrides) -> dict:
    """Return a deep-copy of PLOTLY_LAYOUT with selective key overrides.

    Prevents 'multiple values for keyword argument' errors when passing
    margin, height, or other keys that are already in PLOTLY_LAYOUT.
    """
    import copy
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    layout.update(overrides)
    return layout


def card_html(
    label: str,
    value: str,
    delta: str = "",
    delta_class: str = "lc-neutral",
    value_class: str = "",
) -> str:
    val_cls = f"lc-card-value {value_class}".strip()
    delta_block = (
        f'<div class="lc-card-delta {delta_class}">{delta}</div>' if delta else ""
    )
    return (
        f'<div class="lc-card">'
        f'<div class="lc-card-label">{label}</div>'
        f'<div class="{val_cls}">{value}</div>'
        f"{delta_block}"
        f"</div>"
    )
