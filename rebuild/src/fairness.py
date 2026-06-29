"""Adverse-action reason codes (ECOA / Regulation B compliance).

When a lender denies credit, ECOA and Regulation B require a statement of the
specific principal reasons. The CFPB's Circular 2022-03 confirms this applies to
machine-learning models with no "black-box exception": a model must be able to
produce specific, accurate denial reasons.

This module turns the PD model's per-applicant SHAP contributions into ranked,
consumer-friendly reason codes. It uses the same SHAP machinery as ``explain.py``
and maps transformed feature names back to plain-language reasons.

Scope note: the public Lending Club dataset contains no protected-class
attributes (race, sex, age, etc.), so formal disparate-impact testing and
least-discriminatory-alternative search are out of scope here and are not
fabricated. This module covers the piece the data genuinely supports - explainable
adverse-action notices.
"""

from __future__ import annotations

import pandas as pd

from . import config as C
from . import explain as EX

# Plain-language reason text for each origination feature. Phrased as the reason
# the factor RAISED the applicant's risk (i.e. an adverse-action reason).
REASON_MAP = {
    "dti": "Debt-to-income ratio too high",
    "annual_inc": "Income too low for the requested amount",
    "loan_amnt": "Requested loan amount too high",
    "installment": "Monthly payment burden too high",
    "revol_util": "Revolving credit utilization too high",
    "revol_bal": "Revolving balance too high",
    "open_acc": "Number of open credit lines outside policy",
    "total_acc": "Total number of credit lines outside policy",
    "delinq_2yrs": "Recent delinquencies on file",
    "inq_last_6mths": "Too many recent credit inquiries",
    "emp_length_num": "Length of employment too short",
    "term_months": "Requested loan term carries elevated risk",
    "earliest_cr_line_year": "Length of credit history too short",
    "pub_rec_flag": "Public derogatory record on file",
    "mort_acc_flag": "Mortgage-account profile outside policy",
    "pub_rec_bankruptcies_flag": "Public-record bankruptcy on file",
    "home_ownership": "Home-ownership status carries elevated risk",
    "verification_status": "Income verification status",
    "purpose": "Stated loan purpose carries elevated risk",
    "initial_list_status": "Listing status carries elevated risk",
    "application_type": "Application type carries elevated risk",
    "addr_state": "Geographic risk factors",
    "int_rate": "Assigned interest rate reflects elevated risk",
    "sub_grade": "Assigned credit sub-grade reflects elevated risk",
}

_ALL_FEATURES = (C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
                 + C.GRADE_NUMERIC + C.GRADE_CATEGORICAL)


def _base_feature(transformed_name: str) -> str:
    """Map a transformed feature name (e.g. 'cat__purpose_small_business' or
    'num__dti') back to its base origination feature ('purpose', 'dti')."""
    name = transformed_name
    if "__" in name:
        name = name.split("__", 1)[1]
    # Exact numeric/grade match first
    if name in _ALL_FEATURES:
        return name
    # One-hot categorical: 'purpose_small_business' -> 'purpose'
    for feat in sorted(C.CATEGORICAL_FEATURES + C.GRADE_CATEGORICAL,
                       key=len, reverse=True):
        if name.startswith(feat + "_") or name == feat:
            return feat
    return name


def reason_codes(model, x_row: pd.DataFrame, top_n: int = 4) -> list[dict]:
    """Return ranked adverse-action reason codes for a single applicant.

    Only risk-INCREASING factors (positive SHAP contribution to PD) are returned,
    de-duplicated by base feature, highest contribution first.
    """
    contrib = EX.explain_applicant(model, x_row, top_n=top_n * 4)
    seen, out = set(), []
    for _, r in contrib.iterrows():
        if r["shap"] <= 0:  # only factors that raised risk
            continue
        base = _base_feature(str(r["feature"]))
        if base in seen:
            continue
        seen.add(base)
        out.append({
            "feature": base,
            "reason": REASON_MAP.get(base, f"{base} carries elevated risk"),
            "contribution": float(r["shap"]),
        })
        if len(out) >= top_n:
            break
    return out


def reason_code_frequency(model, X: pd.DataFrame, top_n: int = 3,
                          max_rows: int = 500) -> pd.Series:
    """Frequency of adverse-action reasons across a set of (high-risk) applicants."""
    rows = X.head(max_rows)
    reasons = []
    for i in range(len(rows)):
        for c in reason_codes(model, rows.iloc[[i]], top_n=top_n):
            reasons.append(c["reason"])
    return pd.Series(reasons).value_counts()
