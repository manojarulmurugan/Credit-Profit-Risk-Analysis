"""Streamlit decision demo: applicant inputs -> PD, Expected Loss, Approve/Deny.

Run: rebuild/.venv/bin/python -m streamlit run src/app.py   (or: make app)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from . import config as C
from . import profit as P

st.set_page_config(page_title="Credit Risk + Profit Decision", layout="wide")

MODEL_PATH = C.MODELS_DIR / "pd_model.pkl"
LGD_PATH = C.MODELS_DIR / "lgd_model.pkl"
RETURN_PATH = C.MODELS_DIR / "return_model.pkl"
ECON_PATH = C.MODELS_DIR / "economics.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_lgd_model():
    if not LGD_PATH.exists():
        return None
    return joblib.load(LGD_PATH)


@st.cache_resource
def load_return_model():
    if not RETURN_PATH.exists():
        return None
    return joblib.load(RETURN_PATH)


def load_economics() -> dict:
    if ECON_PATH.exists():
        return json.loads(Path(ECON_PATH).read_text())
    return {"portfolio_lgd": 0.6, "optimal_threshold": 0.2}


def build_feature_row(inp: dict) -> pd.DataFrame:
    """Assemble a one-row frame using the model's engineered feature columns."""
    numeric, categorical = C.feature_columns(include_grade=False)
    row = {c: 0 for c in numeric + categorical}
    row.update(inp)
    return pd.DataFrame([row])[numeric + categorical]


def main():
    st.title("Credit Risk + Profit Decision Engine")
    st.caption("Lending Club PD model -> Expected Loss (PD x LGD x EAD) -> approve/deny")

    model = load_model()
    if model is None:
        st.error("No trained model found. Run `make train` first.")
        st.stop()
    lgd_model = load_lgd_model()
    return_model = load_return_model()
    econ = load_economics()
    portfolio_lgd = econ.get("portfolio_lgd", 0.6)
    default_threshold = econ.get("optimal_threshold", 0.2)

    with st.sidebar:
        st.header("Applicant")
        loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 12000, step=500)
        term_months = st.selectbox("Term (months)", [36, 60], index=0)
        annual_inc = st.number_input("Annual income ($)", 5000, 500000, 65000, step=1000)
        dti = st.slider("Debt-to-income (DTI)", 0.0, 50.0, 18.0)
        emp_length_num = st.slider("Employment length (years)", 0, 10, 5)
        home = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        purpose = st.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "medical", "small_business", "car", "other"])
        verification = st.selectbox("Income verification",
                                    ["Verified", "Source Verified", "Not Verified"])
        revol_util = st.slider("Revolving utilization (%)", 0.0, 150.0, 45.0)
        open_acc = st.number_input("Open credit lines", 0, 60, 11)
        total_acc = st.number_input("Total credit lines", 0, 120, 24)
        revol_bal = st.number_input("Revolving balance ($)", 0, 250000, 13000, step=500)
        delinq_2yrs = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)
        inq_6m = st.number_input("Inquiries (6 mo)", 0, 20, 1)
        pub_rec = st.checkbox("Any public derogatory record?")
        mort = st.checkbox("Any mortgage account?")
        bankruptcy = st.checkbox("Any public-record bankruptcy?")
        threshold = st.slider("Approval PD threshold", 0.0, 1.0, float(default_threshold))

    # Approximate installment from amortization for the expected-revenue piece.
    monthly_rate = 0.13 / 12  # assumed APR for installment estimate
    installment = (loan_amnt * monthly_rate) / (1 - (1 + monthly_rate) ** -term_months)

    inp = {
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
    X = build_feature_row(inp)

    pd_score = float(model.predict_proba(X)[:, 1][0])
    ead = loan_amnt
    if lgd_model is not None:
        lgd = float(lgd_model.predict(X)[0])
        lgd_label = "model"
    else:
        lgd = portfolio_lgd
        lgd_label = "portfolio avg"
    el = pd_score * lgd * ead
    approve = pd_score <= threshold
    exp_ret = float(return_model.predict(X)[0]) if return_model is not None else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of Default", f"{pd_score:.1%}")
    c2.metric("Expected Loss (PD x LGD x EAD)", f"${el:,.0f}")
    c3.metric("Expected Return (ANR)",
              f"{exp_ret:+.1%}" if exp_ret is not None else "n/a")
    c4.metric("Decision", "APPROVE" if approve else "DENY")

    st.progress(min(pd_score, 1.0))
    st.write(f"LGD = {lgd:.2f} ({lgd_label}), EAD = ${ead:,.0f}, threshold = {threshold:.2f}")

    st.subheader("Why this decision? (SHAP)")
    try:
        from . import explain as EX
        contrib = EX.explain_applicant(model, X, top_n=8)
        contrib["effect"] = contrib["shap"].apply(
            lambda v: "increases risk" if v > 0 else "reduces risk")
        st.dataframe(contrib, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Explanation unavailable: {exc}")


if __name__ == "__main__":
    main()
