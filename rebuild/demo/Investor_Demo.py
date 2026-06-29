"""LoanAlpha - single-page, 3-tab demo.

Tabs (top nav, always visible):
  1. Portfolio Explorer  - interactive return-curve backtest
  2. Loan Scorer         - per-loan default probability + expected return
  3. Research Story      - how the model was built, validated, and what it means

Run:  cd rebuild && make demo
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REBUILD_DIR = _DEMO_DIR.parent
sys.path.insert(0, str(_DEMO_DIR))
sys.path.insert(0, str(_REBUILD_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styles import (
    ACCENT, FILL_COLOR, STRATEGY_COLORS, inject_css, plotly_layout,
)
from utils import scorer_tab, evidence_tab

st.set_page_config(
    page_title="LoanAlpha · Investor Demo",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

# ── Data ──────────────────────────────────────────────────────────────────────

REPORTS = _REBUILD_DIR / "reports"

VINTAGE_FILES: dict[tuple[int, str], Path] = {
    (2012, "Grade-Blind"): REPORTS / "return_curve_2012.csv",
    (2012, "Grade-Aware"): REPORTS / "return_curve_with_grade_2012.csv",
    (2013, "Grade-Blind"): REPORTS / "return_curve_2013.csv",
    (2013, "Grade-Aware"): REPORTS / "return_curve_with_grade_2013.csv",
    (2014, "Grade-Blind"): REPORTS / "return_curve_2014.csv",
    (2014, "Grade-Aware"): REPORTS / "return_curve_with_grade_2014.csv",
    (2015, "Grade-Blind"): REPORTS / "return_curve.csv",
    (2015, "Grade-Aware"): REPORTS / "return_curve_with_grade.csv",
}

INVEST_ALL_ANR: dict[int, float] = {2012: 0.0112, 2013: 0.0258, 2014: 0.0212, 2015: 0.0056}

VINTAGE_META: dict[int, dict] = {
    2012: {"loans": 53_367,  "capital": 718,   "maturity": 100, "train_end": 2011},
    2013: {"loans": 134_793, "capital": 1_982, "maturity": 100, "train_end": 2012},
    2014: {"loans": 170_643, "capital": 2_203, "maturity": 77,  "train_end": 2013},
    2015: {"loans": 282_853, "capital": 3_622, "maturity": 75,  "train_end": 2014},
}

VINTAGE_OPTIONS = [2015, 2014, 2013, 2012]

PRESETS = [
    {"label": "Selective (3% budget, 2015)", "vintage_idx": 0, "budget": 3},
    {"label": "Moderate (10%, 2013)",         "vintage_idx": 2, "budget": 10},
    {"label": "Broad (25%, 2012)",             "vintage_idx": 3, "budget": 25},
    {"label": "Best result (3%, 2012)",        "vintage_idx": 3, "budget": 3},
]


@st.cache_data
def load_all_curves() -> dict[tuple[int, str], pd.DataFrame]:
    return {k: pd.read_csv(p) for k, p in VINTAGE_FILES.items() if p.exists()}


curves = load_all_curves()

if "budget_pct" not in st.session_state:
    st.session_state["budget_pct"] = 3
if "vintage_idx" not in st.session_state:
    st.session_state["vintage_idx"] = 0


def _apply_preset(vintage_idx: int, budget: int) -> None:
    st.session_state["vintage_idx"] = vintage_idx
    st.session_state["budget_pct"]  = budget

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    '<div style="padding:8px 0 14px;font-family:Open Sans,sans-serif">'
    '<div style="font-size:1.875rem;font-weight:600;color:#0f171f;line-height:2.25rem;'
    'letter-spacing:-0.02em">Investor Portfolio Intelligence</div>'
    '<div style="font-size:0.875rem;color:#565a5d;margin-top:5px;line-height:1.4rem">'
    "LendingClub loan portfolio optimization &nbsp;·&nbsp; Machine-learning profit scoring"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Top-level tab navigation ──────────────────────────────────────────────────

tab_scorer, tab_explorer, tab_evidence = st.tabs([
    "Loan Scorer",
    "Portfolio Explorer",
    "Research Story",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - Loan Scorer
# ══════════════════════════════════════════════════════════════════════════════

with tab_scorer:
    scorer_tab.render()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - Portfolio Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab_explorer:

    # About panel
    st.markdown(
        '<div class="chase-about-panel">'
        '<div class="chase-about-title">What is this?</div>'
        '<div class="chase-about-body">'
        "<strong>LendingClub</strong> is a platform where investors lend money directly to borrowers. "
        "You earn returns from interest, but lose money if the borrower stops paying. "
        "LendingClub assigns each loan a <strong>letter grade</strong> (A = safest, G = riskiest) "
        "and sets the interest rate accordingly. Most investors follow these grades. "
        "The problem: <em>the grade is already in the price</em> - following it gives no edge over "
        "just investing in every loan."
        "<br><br>"
        "My ML model predicts the <strong>actual yearly return</strong> on each loan, "
        "then selects only the top-ranked ones - delivering consistently 2–5% more per year "
        "than the standard grade-based approach."
        '<div class="chase-about-tags">'
        '<span class="chase-about-tag">Retail investors</span>'
        '<span class="chase-about-tag">Portfolio managers</span>'
        '<span class="chase-about-tag">Fintech researchers</span>'
        '<span class="chase-about-tag">Credit analysts</span>'
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    main_col, side_col = st.columns([3, 1], gap="large")

    with side_col:
        with st.container(border=True):
            st.markdown('<div class="chase-section-header">Try a scenario</div>', unsafe_allow_html=True)
            for p in PRESETS:
                st.button(
                    p["label"],
                    use_container_width=True,
                    on_click=_apply_preset,
                    kwargs={"vintage_idx": p["vintage_idx"], "budget": p["budget"]},
                )
        vintage_placeholder = st.empty()

    with main_col:

        with st.container(border=True):
            st.markdown('<div class="chase-section-header">Select scenario</div>', unsafe_allow_html=True)
            vc1, vc2 = st.columns(2)
            with vc1:
                vintage = st.radio(
                    "Test year",
                    options=VINTAGE_OPTIONS,
                    format_func=str,
                    index=st.session_state["vintage_idx"],
                    help="Each year is an independent out-of-time backtest.",
                )
                st.session_state["vintage_idx"] = VINTAGE_OPTIONS.index(vintage)
            with vc2:
                variant = st.radio(
                    "Model variant",
                    ["Grade-Blind", "Grade-Aware"],
                    help=(
                        "Grade-Blind: no access to LC's interest rate or grade. "
                        "Grade-Aware: allowed to use those signals."
                    ),
                )
            budget_pct = st.slider(
                "Capital budget - what % of available loans to invest in?",
                min_value=1, max_value=75,
                key="budget_pct",
                format="%d%%",
                help="Lower % = more selective. Model picks the top N% by predicted return.",
            )

        key         = (vintage, variant)
        ia_anr      = INVEST_ALL_ANR[vintage]
        budget_frac = budget_pct / 100.0
        meta        = VINTAGE_META[vintage]

        if key not in curves:
            st.error(f"Data not found for {vintage} {variant}. Run `cd rebuild && make portfolio`.")
            st.stop()

        df    = curves[key]
        ps_df = df[df["policy"] == "profit_scoring"].sort_values("fraction_invested").reset_index(drop=True)
        go_df = df[df["policy"] == "grade_only"].sort_values("fraction_invested").reset_index(drop=True)

        def _at(sub: pd.DataFrame) -> pd.Series:
            return sub.loc[(sub["fraction_invested"] - budget_frac).abs().idxmin()]

        ps_row   = _at(ps_df)
        go_row   = _at(go_df)
        ps_anr   = float(ps_row["port_anr"])
        go_anr   = float(go_row["port_anr"])
        n_loans  = int(ps_row["n_invested"])
        vs_grade = (ps_anr - go_anr) * 10_000
        vs_all   = (ps_anr - ia_anr) * 10_000

        is_winning = vs_grade >= 0
        alert_cls  = "chase-alert-success" if is_winning else "chase-alert-error"
        icon_color = "#0A5C36" if is_winning else "#c0000a"
        alert_text = (
            f"My model returns <strong>{ps_anr:+.2%}</strong> annualized - "
            f"<strong>{vs_grade:+.0f} basis points</strong> more per year than LC grade."
            if is_winning else
            f"At {budget_pct}% budget in {vintage}, LC grade narrowly leads. "
            f"Try a lower budget or a different year."
        )

        st.markdown(
            f'<div class="chase-card">'
            f'<div class="chase-card-header">'
            f'<span class="chase-card-title">Portfolio Performance</span>'
            f'<span class="chase-card-subtitle">'
            f'{variant} · {vintage} vintage · top {budget_pct}% of loans'
            f'</span>'
            f"</div>"
            f'<div class="chase-card-body">'
            f'<div class="chase-account-name">PROFIT SCORING &nbsp;·&nbsp; {vintage} VINTAGE</div>'
            f'<div class="{alert_cls}" style="margin-bottom:16px">'
            f'<span style="color:{icon_color};flex-shrink:0">{"✓" if is_winning else "✕"}</span>'
            f'<span style="font-family:Open Sans,sans-serif;font-size:0.875rem;color:#434f5b">'
            f"{alert_text}</span>"
            f"</div>"
            f'<div class="chase-metrics-row">'
            f'<div><div class="chase-metric-value positive">{ps_anr:+.2%}</div>'
            f'<div class="chase-metric-label">My model - annualized return</div></div>'
            f'<div><div class="chase-metric-value" style="color:{"#0A5C36" if vs_grade >= 0 else "#c0000a"}">'
            f"{vs_grade:+.0f} bp</div>"
            f'<div class="chase-metric-label">vs LC grade (basis points)</div></div>'
            f'<div><div class="chase-metric-value">{vs_all:+.0f} bp</div>'
            f'<div class="chase-metric-label">vs investing in everything</div></div>'
            f'<div><div class="chase-metric-value">{n_loans:,}</div>'
            f'<div class="chase-metric-label">Loans selected</div></div>'
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=go_df["fraction_invested"], y=go_df["port_anr"],
            name="LC Grade (standard approach)",
            mode="lines",
            line=dict(color=STRATEGY_COLORS["grade_only"], width=1.5, dash="dot"),
            hovertemplate="Budget: %{x:.1%}<br>LC Grade: %{y:.2%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=ps_df["fraction_invested"], y=ps_df["port_anr"],
            name="My Model - Profit Scoring",
            mode="lines",
            line=dict(color=ACCENT, width=3),
            fill="tonexty",
            fillcolor=FILL_COLOR,
            hovertemplate="Budget: %{x:.1%}<br>My model: %{y:.2%}<extra></extra>",
        ))
        x_lo = float(ps_df["fraction_invested"].min())
        x_hi = float(ps_df["fraction_invested"].max())
        fig.add_trace(go.Scatter(
            x=[x_lo, x_hi], y=[ia_anr, ia_anr],
            name="Invest in everything (passive)",
            mode="lines",
            line=dict(color=STRATEGY_COLORS["invest_all"], width=1.5, dash="dash"),
            hovertemplate=f"Invest-all: {ia_anr:.2%}<extra></extra>",
        ))
        fig.add_vline(
            x=budget_frac,
            line=dict(color=ACCENT, dash="dot", width=1.5),
            annotation_text=f"  Budget: {budget_pct}%",
            annotation_position="top right",
            annotation_font=dict(color=ACCENT, size=12, family="Open Sans, sans-serif"),
        )
        fig.update_layout(**plotly_layout(
            height=360,
            xaxis_title="Fraction of available loans invested",
            yaxis_title="Annualized portfolio return",
            margin=dict(l=55, r=20, t=50, b=50),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            '<div style="font-family:Open Sans,sans-serif;font-size:0.75rem;color:#85888a;'
            'margin-top:4px;line-height:1.5">'
            "<strong>bp = basis points</strong> - 1 bp = 0.01% annualized return. "
            "Grade-Blind = model never saw LC's interest rate or grade."
            "</div>",
            unsafe_allow_html=True,
        )

    vintage_placeholder.markdown(
        f'<div class="chase-widget">'
        f'<div class="chase-widget-header">{vintage} Test Cohort</div>'
        f'<div class="chase-widget-body">'
        f'<div class="chase-widget-row"><span class="chase-widget-row-key">Trained on</span>'
        f'<span class="chase-widget-row-val">2007–{meta["train_end"]}</span></div>'
        f'<div class="chase-widget-row"><span class="chase-widget-row-key">Tested on</span>'
        f'<span class="chase-widget-row-val">{vintage} only</span></div>'
        f'<div class="chase-widget-row"><span class="chase-widget-row-key">Fully settled</span>'
        f'<span class="chase-widget-row-val">{meta["maturity"]}%</span></div>'
        f'<div class="chase-widget-row"><span class="chase-widget-row-key">Loans evaluated</span>'
        f'<span class="chase-widget-row-val">{meta["loans"]:,}</span></div>'
        f'<div class="chase-widget-row"><span class="chase-widget-row-key">Total capital</span>'
        f'<span class="chase-widget-row-val">${meta["capital"]:,}M</span></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - Research Story
# ══════════════════════════════════════════════════════════════════════════════

with tab_evidence:
    evidence_tab.render()
