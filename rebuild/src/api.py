"""FastAPI scoring service: applicant -> PD, LGD, Expected Loss, decision, reasons.

Endpoints:
    GET  /health   liveness + whether models are loaded
    POST /score    score one applicant and return a full credit decision

Run locally:
    rebuild/.venv/bin/python -m uvicorn src.api:app --reload   (or: make api)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config as C
from . import fairness as FA

app = FastAPI(title="Credit Risk + Profit Decision API", version="1.0")

_MODELS: dict = {}


def _load() -> dict:
    """Lazy-load the PD model, LGD model, and economics config once."""
    if _MODELS:
        return _MODELS
    pd_path = C.MODELS_DIR / "pd_model.pkl"
    if pd_path.exists():
        _MODELS["pd"] = joblib.load(pd_path)
    lgd_path = C.MODELS_DIR / "lgd_model.pkl"
    if lgd_path.exists():
        _MODELS["lgd"] = joblib.load(lgd_path)
    ret_path = C.MODELS_DIR / "return_model.pkl"
    if ret_path.exists():
        _MODELS["return"] = joblib.load(ret_path)
    econ_path = C.MODELS_DIR / "economics.json"
    _MODELS["econ"] = (json.loads(Path(econ_path).read_text())
                       if econ_path.exists()
                       else {"portfolio_lgd": 0.6, "optimal_threshold": 0.3})
    return _MODELS


class Applicant(BaseModel):
    loan_amnt: float = Field(12000, ge=1000, le=40000)
    term_months: int = Field(36, description="36 or 60")
    annual_inc: float = Field(65000, ge=0)
    dti: float = Field(18.0, ge=0)
    emp_length_num: int = Field(5, ge=0, le=10)
    home_ownership: str = "MORTGAGE"
    purpose: str = "debt_consolidation"
    verification_status: str = "Verified"
    revol_util: float = 45.0
    open_acc: int = 11
    total_acc: int = 24
    revol_bal: float = 13000
    delinq_2yrs: int = 0
    inq_last_6mths: int = 1
    installment: Optional[float] = None
    earliest_cr_line_year: int = 2005
    pub_rec_flag: int = 0
    mort_acc_flag: int = 0
    pub_rec_bankruptcies_flag: int = 0
    initial_list_status: str = "w"
    application_type: str = "Individual"
    addr_state: str = "CA"


class ScoreResponse(BaseModel):
    probability_of_default: float
    lgd: float
    exposure_at_default: float
    expected_loss: float
    expected_return: Optional[float] = None  # predicted annualized net return (ANR)
    decision: str
    threshold: float
    reason_codes: list[dict]


def _feature_row(a: Applicant) -> pd.DataFrame:
    numeric, categorical = C.feature_columns(include_grade=False)
    row = {c: 0 for c in numeric + categorical}
    data = a.model_dump()
    if data.get("installment") is None:
        r = 0.13 / 12
        data["installment"] = (data["loan_amnt"] * r) / (1 - (1 + r) ** -data["term_months"])
    row.update({k: v for k, v in data.items() if k in row})
    return pd.DataFrame([row])[numeric + categorical]


@app.get("/health")
def health() -> dict:
    m = _load()
    return {"status": "ok", "pd_model": "pd" in m, "lgd_model": "lgd" in m,
            "return_model": "return" in m}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    m = _load()
    if "pd" not in m:
        return ScoreResponse(probability_of_default=-1, lgd=0, exposure_at_default=0,
                             expected_loss=0, decision="MODEL_NOT_TRAINED",
                             threshold=0, reason_codes=[])
    X = _feature_row(applicant)
    pd_score = float(m["pd"].predict_proba(X)[:, 1][0])
    lgd = (float(m["lgd"].predict(X)[0]) if "lgd" in m
           else float(m["econ"].get("portfolio_lgd", 0.6)))
    ead = float(applicant.loan_amnt)
    el = pd_score * lgd * ead
    exp_ret = round(float(m["return"].predict(X)[0]), 4) if "return" in m else None
    threshold = float(m["econ"].get("optimal_threshold", 0.3))
    decision = "APPROVE" if pd_score <= threshold else "DENY"
    reasons = FA.reason_codes(m["pd"], X, top_n=4) if decision == "DENY" else []
    return ScoreResponse(
        probability_of_default=round(pd_score, 4),
        lgd=round(lgd, 4),
        exposure_at_default=ead,
        expected_loss=round(el, 2),
        expected_return=exp_ret,
        decision=decision,
        threshold=threshold,
        reason_codes=reasons,
    )
