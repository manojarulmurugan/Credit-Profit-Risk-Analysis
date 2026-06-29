"""Loan Scorer tab - real-time credit risk assessment.

Renders as a self-contained section; call render() inside a st.tabs() block.
No set_page_config or inject_css here (caller handles both).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent
_REBUILD_DIR = _DEMO_DIR.parent
sys.path.insert(0, str(_DEMO_DIR))
sys.path.insert(0, str(_REBUILD_DIR))

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styles import plotly_layout

try:
    from src import config as C
    _SRC_OK = True
except ImportError:
    _SRC_OK = False

# ── Label maps ────────────────────────────────────────────────────────────────

PURPOSE_LABELS: dict[str, str] = {
    "debt_consolidation": "Debt Consolidation",
    "credit_card":        "Credit Card Payoff",
    "home_improvement":   "Home Improvement",
    "major_purchase":     "Major Purchase",
    "medical":            "Medical Expenses",
    "small_business":     "Small Business",
    "car":                "Car / Auto Loan",
    "other":              "Other",
}
_PURPOSE_REVERSE = {v: k for k, v in PURPOSE_LABELS.items()}

HOME_LABELS: dict[str, str] = {
    "MORTGAGE": "Has Mortgage",
    "RENT":     "Renting",
    "OWN":      "Owns Outright",
    "OTHER":    "Other",
}
_HOME_REVERSE = {v: k for k, v in HOME_LABELS.items()}

VERIFICATION_LABELS: dict[str, str] = {
    "Verified":        "Income Verified",
    "Source Verified": "Source Verified",
    "Not Verified":    "Not Verified",
}
_VERIF_REVERSE = {v: k for k, v in VERIFICATION_LABELS.items()}


# ── Model loading (cached at module level so it persists across reruns) ───────

@st.cache_resource
def _load_models() -> dict:
    if not _SRC_OK:
        return {}
    out: dict = {}
    for name, path in [
        ("pd",  C.MODELS_DIR / "pd_model.pkl"),
        ("lgd", C.MODELS_DIR / "lgd_model.pkl"),
        ("anr", C.MODELS_DIR / "return_model.pkl"),
    ]:
        if path.exists():
            out[name] = joblib.load(path)
    econ_path = C.MODELS_DIR / "economics.json"
    out["econ"] = (
        json.loads(econ_path.read_text()) if econ_path.exists()
        else {"portfolio_lgd": 0.6, "optimal_threshold": 0.2}
    )
    return out


def _build_feature_row(inp: dict) -> pd.DataFrame:
    numeric, categorical = C.feature_columns(include_grade=False)
    row = {c: 0 for c in numeric + categorical}
    row.update(inp)
    return pd.DataFrame([row])[numeric + categorical]


def _clean(f: str) -> str:
    return f.split("__", 1)[-1] if "__" in f else f


# ── Main render function ──────────────────────────────────────────────────────

def render() -> None:
    """Render the full Loan Scorer UI inside the current Streamlit context."""

    st.markdown(
        '<div style="padding:4px 0 20px;font-family:Open Sans,sans-serif">'
        '<div style="font-size:1.75rem;font-weight:600;color:#0f171f;line-height:2.25rem;'
        'letter-spacing:-0.01em">Loan Application Scorer</div>'
        '<div style="font-size:0.875rem;color:#565a5d;margin-top:6px;line-height:1.4rem">'
        "<strong style='color:#434f5b'>For loan officers and credit analysts.</strong> "
        "Enter the applicant's loan details to get an AI-powered risk assessment - "
        "probability of default, expected loss, and predicted investor return."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    models = _load_models()
    demo_mode = "pd" not in models

    if demo_mode:
        econ           = {"portfolio_lgd": 0.58, "optimal_threshold": 0.20}
        portfolio_lgd  = 0.58
        default_thresh = 0.20
    else:
        econ           = models["econ"]
        portfolio_lgd  = float(econ.get("portfolio_lgd", 0.6))
        default_thresh = float(econ.get("optimal_threshold", 0.2))

    form_col, result_col = st.columns([5, 4], gap="large")

    with form_col:
        with st.container(border=True):
            st.markdown('<div class="chase-section-header">Loan Information</div>', unsafe_allow_html=True)
            fa, fb = st.columns(2)
            loan_amnt   = fa.number_input("Requested Amount ($)", 1_000, 40_000, 12_000, step=500)
            term_months = fb.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
            purpose_display = st.selectbox("Loan Purpose", list(PURPOSE_LABELS.values()))
            purpose = _PURPOSE_REVERSE[purpose_display]

        with st.container(border=True):
            st.markdown('<div class="chase-section-header">Borrower Profile</div>', unsafe_allow_html=True)
            fc, fd = st.columns(2)
            annual_inc = fc.number_input("Annual Income ($)", 5_000, 500_000, 65_000, step=1_000)
            dti        = fd.slider("Debt-to-Income Ratio (DTI) %", 0.0, 50.0, 18.0, step=0.5,
                                   help="Monthly debt payments ÷ gross monthly income × 100.")
            fe, ff = st.columns(2)
            emp_length_num = fe.slider("Years at Current Job", 0, 10, 5,
                                       help="0 = less than 1 year; 10 = 10+ years.")
            home_display = ff.selectbox("Housing Status", list(HOME_LABELS.values()))
            home = _HOME_REVERSE[home_display]
            verif_display = st.selectbox("Income Verification",
                                          list(VERIFICATION_LABELS.values()),
                                          help="Has the applicant's income been independently verified?")
            verification = _VERIF_REVERSE[verif_display]

        with st.container(border=True):
            st.markdown('<div class="chase-section-header">Credit History</div>', unsafe_allow_html=True)
            fg, fh = st.columns(2)
            revol_util = fg.slider("Credit Card Usage (% of Limit)", 0.0, 100.0, 45.0,
                                   help="Revolving credit balance ÷ credit limit × 100.")
            open_acc = fh.number_input("Open Credit Lines", 0, 60, 11)
            fi, fj = st.columns(2)
            total_acc = fi.number_input("Total Credit Accounts (All Time)", 0, 120, 24)
            revol_bal = fj.number_input("Total Credit Card Balance ($)", 0, 250_000, 13_000, step=500)
            fk, fl = st.columns(2)
            delinq_2yrs = fk.number_input("Late Payments in Past 2 Years", 0, 20, 0)
            inq_6m      = fl.number_input("Credit Inquiries in Past 6 Months", 0, 20, 1,
                                          help="Hard credit pulls in the last 6 months.")
            fm, fn, fo = st.columns(3)
            pub_rec    = fm.checkbox("Public Derogatory Record",
                                     help="Any bankruptcies, tax liens, or judgments.")
            mort       = fn.checkbox("Has Mortgage Account")
            bankruptcy = fo.checkbox("Bankruptcy on Record")

        with st.container(border=True):
            st.markdown('<div class="chase-section-header">Approval Settings</div>', unsafe_allow_html=True)
            threshold = st.slider("Approval PD Threshold", 0.05, 0.60, default_thresh, step=0.01,
                                  help="Approve if predicted default probability ≤ this value. Lower = stricter.")

    # ── Scoring ───────────────────────────────────────────────────────────────

    monthly_rate = 0.13 / 12
    installment  = (loan_amnt * monthly_rate) / (1 - (1 + monthly_rate) ** -term_months)

    if demo_mode:
        # Pre-computed values for the default form inputs (realistic model output)
        pd_score = 0.112
        lgd_val  = 0.58
        el_val   = pd_score * lgd_val * loan_amnt
        approve  = pd_score <= threshold
        anr_pred = 0.048
    else:
        applicant = {
            "loan_amnt": loan_amnt, "installment": installment, "annual_inc": annual_inc,
            "dti": dti, "open_acc": open_acc, "revol_bal": revol_bal,
            "revol_util": revol_util, "total_acc": total_acc, "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_6m, "emp_length_num": emp_length_num,
            "term_months": term_months, "earliest_cr_line_year": 2005,
            "pub_rec_flag": int(pub_rec), "mort_acc_flag": int(mort),
            "pub_rec_bankruptcies_flag": int(bankruptcy),
            "home_ownership": home, "verification_status": verification,
            "purpose": purpose, "initial_list_status": "w",
            "application_type": "Individual", "addr_state": "CA",
        }
        X = _build_feature_row(applicant)

        pd_score = float(models["pd"].predict_proba(X)[:, 1][0])
        lgd_val  = float(models["lgd"].predict(X)[0]) if "lgd" in models else portfolio_lgd
        el_val   = pd_score * lgd_val * loan_amnt
        approve  = pd_score <= threshold
        anr_pred = float(models["anr"].predict(X)[0]) if "anr" in models else None

    # ── Results ───────────────────────────────────────────────────────────────

    with result_col:

        if approve:
            badge_html = (
                '<span style="display:inline-flex;align-items:center;gap:8px;'
                'background:rgba(10,92,54,0.06);color:#0A5C36;'
                'border:1px solid rgba(10,92,54,0.25);border-radius:4px;'
                'padding:8px 20px;font-family:Open Sans,sans-serif;'
                'font-size:0.875rem;font-weight:600;letter-spacing:0.04em;'
                'text-transform:uppercase">✓&ensp;Approved</span>'
            )
            alert_html = (
                f'<div class="chase-alert-success">'
                f'<span style="color:#0A5C36;flex-shrink:0">✓</span>'
                f'<span style="font-family:Open Sans,sans-serif;font-size:0.75rem;color:#434f5b">'
                f"Default probability {pd_score:.1%} is within the {threshold:.0%} approval threshold."
                f"</span></div>"
            )
        else:
            badge_html = (
                '<span style="display:inline-flex;align-items:center;gap:8px;'
                'background:rgba(192,0,10,0.05);color:#991b1b;'
                'border:1px solid rgba(192,0,10,0.2);border-radius:4px;'
                'padding:8px 20px;font-family:Open Sans,sans-serif;'
                'font-size:0.875rem;font-weight:600;letter-spacing:0.04em;'
                'text-transform:uppercase">✕&ensp;Declined</span>'
            )
            alert_html = (
                f'<div class="chase-alert-error">'
                f'<span style="color:#c0000a;flex-shrink:0">✕</span>'
                f'<span style="font-family:Open Sans,sans-serif;font-size:0.75rem;color:#434f5b">'
                f"Default probability {pd_score:.1%} exceeds the {threshold:.0%} approval threshold."
                f"</span></div>"
            )

        card_subtitle = (
            'Sample output &nbsp;<span style="background:#e9f0eb;color:#0A5C36;'
            'font-size:0.7rem;font-weight:600;padding:2px 7px;border-radius:3px;'
            'letter-spacing:0.04em">DEMO</span>'
            if demo_mode else "Real-time ML assessment"
        )
        st.markdown(
            f'<div class="chase-card">'
            f'<div class="chase-card-header">'
            f'<span class="chase-card-title">Application Decision</span>'
            f'<span class="chase-card-subtitle">{card_subtitle}</span>'
            f"</div>"
            f'<div class="chase-card-body">'
            f'<div class="chase-account-name">LOAN APPLICATION - ${loan_amnt:,} · {term_months} MONTHS</div>'
            f"{alert_html}"
            f'<div style="margin:12px 0 8px">{badge_html}</div>'
            f'<div class="chase-metrics-row">'
            f'<div>'
            f'<div class="chase-metric-value" style="color:{"#0A5C36" if approve else "#c0000a"}">'
            f'{pd_score:.1%}</div>'
            f'<div class="chase-metric-label">Probability of default</div>'
            f"</div>"
            f"<div>"
            f'<div class="chase-metric-value">${el_val:,.0f}</div>'
            f'<div class="chase-metric-label">Expected loss</div>'
            f"</div>"
            + (
                f"<div>"
                f'<div class="chase-metric-value" style="color:{"#0A5C36" if anr_pred and anr_pred > 0 else "#c0000a"}">'
                f"{anr_pred:+.2%}</div>"
                f'<div class="chase-metric-label">Predicted investor return</div>'
                f"</div>"
                if anr_pred is not None else ""
            ) +
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="chase-card">'
            f'<div class="chase-card-header">'
            f'<span class="chase-card-title">Loss Given Default (LGD)</span>'
            f"</div>"
            f'<div class="chase-card-body">'
            f'<div class="chase-metrics-row">'
            f'<div><div class="chase-metric-value">{lgd_val:.0%}</div>'
            f'<div class="chase-metric-label">Estimated severity</div></div>'
            f'<div><div class="chase-metric-value">{pd_score:.1%}</div>'
            f'<div class="chase-metric-label">× Probability of default</div></div>'
            f'<div><div class="chase-metric-value">${el_val:,.0f}</div>'
            f'<div class="chase-metric-label">= Expected credit loss</div></div>'
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                '<div class="chase-section-header">Key Risk Drivers'
                '<span style="font-weight:400;color:#434f5b;margin-left:8px;font-size:0.75rem">'
                "Which factors drove the score?"
                "</span></div>",
                unsafe_allow_html=True,
            )
            if demo_mode:
                # Pre-computed SHAP values for the default form inputs
                feat_names = [
                    "Debt-to-Income Ratio", "Annual Income", "Credit Card Usage %",
                    "Years at Current Job", "Has Mortgage", "Recent Inquiries",
                    "Open Credit Lines", "Loan Amount",
                ]
                shap_vals = [-0.031, -0.026, 0.021, -0.017, -0.013, 0.009, -0.006, 0.004]
            else:
                try:
                    from src import explain as EX
                    contrib   = EX.explain_applicant(models["pd"], X, top_n=8)
                    raw_names = (
                        contrib["feature"].tolist() if "feature" in contrib.columns
                        else contrib.index.tolist()
                    )
                    label_map = {
                        "loan_amnt": "Loan Amount", "installment": "Monthly Payment",
                        "annual_inc": "Annual Income", "dti": "Debt-to-Income Ratio",
                        "open_acc": "Open Credit Lines", "revol_bal": "Credit Card Balance",
                        "revol_util": "Credit Card Usage %", "total_acc": "Total Credit Accounts",
                        "delinq_2yrs": "Late Payments (2yr)", "inq_last_6mths": "Recent Inquiries",
                        "emp_length_num": "Years at Current Job", "term_months": "Loan Term",
                        "earliest_cr_line_year": "Credit History Age",
                        "pub_rec_flag": "Public Derogatory Record",
                        "mort_acc_flag": "Has Mortgage", "pub_rec_bankruptcies_flag": "Bankruptcy Record",
                    }
                    feat_names = [
                        label_map.get(_clean(f), _clean(f).replace("_", " ").title())
                        for f in raw_names
                    ]
                    shap_vals = contrib["shap"].tolist()
                except Exception as exc:
                    st.caption(f"Feature impact chart unavailable: {exc}")
                    feat_names, shap_vals = [], []

            if feat_names:
                bar_colors = ["#16a34a" if v < 0 else "#dc2626" for v in shap_vals]
                shap_fig = go.Figure(go.Bar(
                    x=shap_vals, y=feat_names, orientation="h",
                    marker_color=bar_colors,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                shap_fig.update_layout(**plotly_layout(
                    height=280,
                    margin=dict(l=10, r=20, t=10, b=40),
                    showlegend=False,
                    xaxis=dict(
                        title="Impact on default probability",
                        gridcolor="#e2e4e5", linecolor="#d3d9de",
                        tickformat=".3f", zeroline=True,
                        zerolinecolor="#9ca3af", zerolinewidth=1,
                        title_font=dict(color="#434f5b", size=11),
                        tickfont=dict(color="#565a5d"),
                    ),
                    yaxis=dict(
                        gridcolor="#e2e4e5", linecolor="#d3d9de",
                        tickformat="", zeroline=False,
                        tickfont=dict(color="#434f5b", size=11),
                    ),
                ))
                st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    '<div style="font-family:Open Sans,sans-serif;font-size:0.75rem;'
                    'color:#85888a;margin-top:-6px">'
                    '<span style="color:#16a34a">■</span> Reduces default risk &ensp;'
                    '<span style="color:#dc2626">■</span> Increases default risk'
                    "</div>",
                    unsafe_allow_html=True,
                )
