"""Research Story tab - the research narrative, validation results, and charts.

Call render() inside a st.tabs() block. No set_page_config or inject_css here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent
_REBUILD_DIR = _DEMO_DIR.parent

import streamlit as st

REPORTS = _REBUILD_DIR / "reports"


def _prose(text: str) -> None:
    st.markdown(
        f'<div style="font-family:Open Sans,sans-serif;font-size:0.9rem;'
        f'color:#434f5b;line-height:1.7;max-width:820px;margin-bottom:14px">{text}</div>',
        unsafe_allow_html=True,
    )


def _callout(text: str, color: str = "#0A5C36") -> None:
    bg = "rgba(10, 92, 54, 0.05)"
    st.markdown(
        f'<div style="border-left:3px solid {color};padding:12px 20px;'
        f'background:{bg};border-radius:0 4px 4px 0;'
        f'font-family:Open Sans,sans-serif;font-size:0.9rem;color:#434f5b;'
        f'line-height:1.7;margin:16px 0;max-width:820px">'
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def _stat_band(stats: list[tuple[str, str, str]]) -> None:
    """Row of headline stat cards: (value, label, sublabel)."""
    cards = "".join(
        f'<div style="flex:1;min-width:160px;background:#fff;border:1px solid #e2e4e5;'
        f'border-radius:8px;padding:16px 18px;box-shadow:0 1px 4px rgba(15,23,31,0.06)">'
        f'<div style="font-size:1.5rem;font-weight:600;color:#0A5C36;'
        f'line-height:1.875rem;letter-spacing:-0.02em">{value}</div>'
        f'<div style="font-size:0.8125rem;font-weight:600;color:#0f171f;margin-top:4px">{label}</div>'
        f'<div style="font-size:0.75rem;color:#85888a;margin-top:2px;line-height:1.1rem">{sub}</div>'
        f"</div>"
        for value, label, sub in stats
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:8px;'
        f'font-family:Open Sans,sans-serif">{cards}</div>',
        unsafe_allow_html=True,
    )


def _section(title: str) -> None:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:Open Sans,sans-serif;font-size:1.05rem;'
        f'font-weight:600;color:#0f171f;letter-spacing:-0.01em;margin-bottom:6px;">'
        f"{title}</div>"
        f'<hr style="border:none;border-top:1px solid #e2e4e5;margin-bottom:16px">',
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render the full research story inside the current Streamlit context."""

    st.markdown(
        '<div style="padding:4px 0 18px;font-family:Open Sans,sans-serif">'
        '<div style="font-size:1.75rem;font-weight:600;color:#0f171f;line-height:2.25rem;'
        'letter-spacing:-0.01em">Research Story</div>'
        '<div style="font-size:0.875rem;color:#565a5d;margin-top:6px;line-height:1.4rem">'
        "How the model was built, validated, and what the results actually mean."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Headline results band ─────────────────────────────────────────────────
    _stat_band([
        ("+200 to +400 bp", "Typical annual uplift", "vs LC grade at 3% budget"),
        ("4", "Vintages backtested", "2012 through 2015, out-of-time"),
        ("642K", "Loans evaluated", "across all test years"),
        ("2", "Fully-settled vintages", "2012 and 2013, exact cash flows"),
    ])

    # ── 1. The Problem ────────────────────────────────────────────────────────

    _section("1. The Problem with Grade-Based Investing")

    _prose(
        "LendingClub is a peer-to-peer lending platform where investors fund personal loans "
        "directly. The platform assigns each loan a <strong>letter grade</strong> (A through G) "
        "and sets the interest rate accordingly - A loans are the 'safest' with the lowest rate; "
        "G loans are the riskiest with the highest rate."
        "<br><br>"
        "The intuitive investor strategy is to follow the grade: "
        "pick high-grade loans to be safe, or low-grade loans to earn more interest. "
        "But this is a trap."
    )
    _callout(
        "The grade is already priced in. By the time you see a loan, the interest rate "
        "has already been set to compensate for the grade's expected default rate. "
        "An investor who buys every loan earns roughly the same return as one who carefully "
        "cherry-picks by grade - the market has already done that work for you."
    )
    _prose(
        "I tested this empirically. On the 2015 vintage, investing in everything yielded "
        "<strong>+0.56% annualized</strong>. Picking only A-grade loans yielded roughly the "
        "same. Grade selection adds no edge because the <em>information is already in the price</em>."
    )

    # ── 2. The Approach ───────────────────────────────────────────────────────

    _section("2. My Approach: Predict the Actual Return, Not the Grade")

    _prose(
        "I built a machine-learning model that predicts the <strong>annualized net return</strong> "
        "(ANR) an investor actually receives on each loan - accounting for both interest income "
        "and the risk of the borrower stopping payment. This is a regression on realized cash flows, "
        "not a classification of 'good' vs 'bad'."
        "<br><br>"
        "Crucially, the <strong>Grade-Blind variant</strong> of my model deliberately excludes "
        "interest rate and LC grade from its features. This forces the model to discover mispriced "
        "risk from origination data alone: income, debt-to-income ratio, employment length, "
        "credit utilization, delinquency history, and loan purpose."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div style="background:#fff;border:1px solid #e2e4e5;border-top:2px solid #475569;'
            'border-radius:8px;padding:16px 20px;font-family:Open Sans,sans-serif">'
            '<div style="font-size:0.75rem;font-weight:600;color:#475569;'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">'
            "Default Risk Model (PD)</div>"
            '<div style="font-size:0.875rem;color:#434f5b;line-height:1.6">'
            "Trained to predict <em>will this borrower stop paying?</em> using gradient boosting. "
            "Used to compute Expected Loss and set approval thresholds."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="background:#fff;border:1px solid #e2e4e5;border-top:2px solid #0A5C36;'
            'border-radius:8px;padding:16px 20px;font-family:Open Sans,sans-serif">'
            '<div style="font-size:0.75rem;font-weight:600;color:#0A5C36;'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">'
            "Profit Scoring Model (ANR)</div>"
            '<div style="font-size:0.875rem;color:#434f5b;line-height:1.6">'
            "Trained to predict <em>what actual return will an investor earn?</em> "
            "Used to rank loans by expected profitability and build a selective portfolio."
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ── 3. Validation ─────────────────────────────────────────────────────────

    _section("3. Out-of-Time Validation (No Data Leakage)")

    _prose(
        "Standard cross-validation would be misleading here - loans issued in the same year "
        "share macroeconomic conditions, creating data leakage. Instead, I used "
        "<strong>out-of-time (OOT) validation</strong>: train strictly on loans issued "
        "<em>before</em> the test year, then evaluate on that year alone."
    )

    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'
        'margin:16px 0;max-width:720px">'
        + "".join([
            f'<div style="background:#fff;border:1px solid #e2e4e5;border-radius:8px;'
            f'padding:14px 16px;font-family:Open Sans,sans-serif">'
            f'<div style="font-size:1.25rem;font-weight:600;color:#0f171f">{yr}</div>'
            f'<div style="font-size:0.75rem;color:#434f5b;margin-top:3px">Trained on 2007–{tr}</div>'
            f'<div style="font-size:0.75rem;color:#565a5d">{loans}</div>'
            f"</div>"
            for yr, tr, loans in [
                (2012, 2011, "53K loans, 100% settled"),
                (2013, 2012, "135K loans, 100% settled"),
                (2014, 2013, "171K loans, 77% settled"),
                (2015, 2014, "283K loans, 75% settled"),
            ]
        ])
        + "</div>",
        unsafe_allow_html=True,
    )

    _callout(
        "2012 and 2013 vintages are <strong>fully matured</strong> - every loan has reached "
        "its final outcome. The returns shown are <em>exact realized cash flows</em>, not estimates. "
        "This is the gold standard of retrospective backtesting."
    )

    # ── 4. Results ────────────────────────────────────────────────────────────

    _section("4. Results - Return Curves Across All 4 Vintages")

    _prose(
        "The chart below shows the <strong>return curve</strong> for each vintage: "
        "what annualized return you'd earn as a function of how selective you are. "
        "Investing in the top 3% of loans by my model consistently "
        "outperforms the passive 'invest in everything' strategy and the grade-based approach."
    )

    fig1_path = REPORTS / "fig1_return_curves_all_vintages.png"
    if fig1_path.exists():
        with st.container(border=True):
            st.image(str(fig1_path), use_column_width=True)
    else:
        st.info("Figure not found - run `cd rebuild && make portfolio` to generate it.")

    st.markdown(
        '<div style="font-family:Open Sans,sans-serif;font-size:0.75rem;color:#85888a;'
        'margin-top:8px;line-height:1.5;max-width:720px">'
        "Each panel = one independent backtest year. X axis = what fraction of available loans "
        "you fund. Y axis = annualized return. Green line = my profit scoring model. "
        "Gray dotted = LC grade baseline."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 5. Strategy Comparison ────────────────────────────────────────────────

    _section("5. Head-to-Head: 4 Investment Strategies")

    _prose(
        "I compared four strategies at two budget levels (3% = highly selective, 25% = broad):"
        "<br>"
        "• <strong style='color:#0A5C36'>Profit Scoring</strong> - my model, ranked by predicted return<br>"
        "• <strong style='color:#475569'>Default Scoring</strong> - ranked by predicted safety (credit score approach)<br>"
        "• <strong style='color:#94a3b8'>LC Grade Only</strong> - follow LendingClub's own rating<br>"
        "• <strong>Invest in Everything</strong> - passive, no selection"
    )

    fig3_path = REPORTS / "fig3_bar_policy_comparison.png"
    if fig3_path.exists():
        with st.container(border=True):
            st.image(str(fig3_path), use_column_width=True)
    else:
        st.info("Figure not found - run `cd rebuild && make portfolio` to generate it.")

    _callout(
        "<strong>Key finding:</strong> Profit scoring outperforms all alternatives across all "
        "4 vintages at 3% budget, by an average of <strong>+200–400 basis points</strong> "
        "annualized. Even at 25% budget, the advantage holds - the model identifies genuinely "
        "mispriced loans that grade-based investors systematically miss."
    )

    # ── 6. Why it works ───────────────────────────────────────────────────────

    _section("6. Why It Works: Mispriced Risk")

    _prose(
        "The Grade-Blind model never sees the interest rate or LC grade. Yet it still beats "
        "grade-based selection. This proves it has learned something the grade does not capture: "
        "the <strong>actual repayment behavior</strong> conditional on the borrower's profile."
        "<br><br>"
        "The top predictive features are: "
        "<strong>debt-to-income ratio</strong>, "
        "<strong>revolving credit utilization</strong>, "
        "<strong>number of recent credit inquiries</strong>, and "
        "<strong>employment length</strong>. "
        "These interact in non-linear ways that a simple grade cannot summarize."
    )

    _prose(
        "<strong>What about the Grade-Aware model?</strong> "
        "Including the grade improves performance further - but the Grade-Blind model's edge "
        "shows the model isn't just re-learning the grade. It's finding signal the grade misses."
    )

    # ── 7. Summary stats ─────────────────────────────────────────────────────

    _section("7. Summary Statistics")

    summary_path = REPORTS / "cross_vintage_summary_table.csv"
    if summary_path.exists():
        import pandas as pd
        df = pd.read_csv(summary_path)
        keep_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ["vintage", "variant", "3%", "25%", "optimal", "grade", "bp"]
        )]
        if keep_cols:
            df = df[keep_cols]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Summary table not found - run `cd rebuild && make portfolio` to generate it.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Open Sans,sans-serif;font-size:0.75rem;color:#85888a;'
        'line-height:1.6;max-width:720px;padding-top:16px;border-top:1px solid #e2e4e5">'
        "<strong>References:</strong> "
        "Emekter et al. (2015), Zhang et al. (2016), Serrano-Cinca et al. (2015), "
        "Jagtiani & Lemieux (2019) - all find that LC grades leave substantial pricing inefficiency. "
        "My work extends this by framing the problem as return prediction rather than default classification."
        "</div>",
        unsafe_allow_html=True,
    )
