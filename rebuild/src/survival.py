"""Lifetime PD via discrete-time survival analysis + IFRS 9 staging / ECL.

The headline PD model answers a single yes/no question ("will this loan ever
default?"). IFRS 9 needs more: a *term structure* of default risk - the
probability of default in each future month - so that 12-month and lifetime
Expected Credit Loss (ECL) can be computed.

Approach (discrete-time survival, a.k.a. pooled logistic hazard):

1. Each loan is expanded into one row per month it was observed (person-period
   format). The monthly event indicator is 1 only in the month a charged-off
   loan defaults, and 0 otherwise. Fully-paid loans are censored at payoff.
2. A logistic regression models the monthly hazard h(t) = P(default in month t |
   survived to t) from origination features plus a flexible function of loan age
   t (the baseline hazard).
3. From h(t) we derive the survival curve S(t), the cumulative default
   probability F(t) = 1 - S(t), 12-month PD = F(12), and lifetime PD = F(term).
4. IFRS 9 staging (Stage 1/2/3) and ECL = sum_t marginal_PD(t) x LGD x EAD,
   discounted, give a 12-month vs lifetime loss-allowance comparison.

Origination features only enter the hazard model; the time-to-event label is
derived from issue_d / last_pymnt_d (label construction, not features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

# Origination numeric drivers used in the hazard model (kept numeric-only so the
# person-period matrix stays compact). These are a subset of the PD allowlist.
HAZARD_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "revol_util", "open_acc", "total_acc",
    "delinq_2yrs", "inq_last_6mths", "emp_length_num", "term_months",
    "earliest_cr_line_year", "pub_rec_flag",
]
TIME_COLS = ["period", "period_sq", "log_period"]


# --------------------------------------------------------------------------- #
# Time-to-event construction
# --------------------------------------------------------------------------- #
def loan_duration_months(df: pd.DataFrame) -> pd.Series:
    """Observed months from origination to last activity, capped at the term.

    Uses the gap between issue date and last payment date; falls back to the
    contractual term when the last-payment date is missing.
    """
    issue = df["issue_date"]
    last = df.get("last_pymnt_date")
    months = pd.Series(np.nan, index=df.index)
    if last is not None:
        months = (last.dt.year - issue.dt.year) * 12 + (last.dt.month - issue.dt.month)
    term = df["term_months"].fillna(36)
    months = months.where(months.notna(), term)
    months = months.clip(lower=1)
    months = np.minimum(months, term)
    return months.astype(int)


def build_person_period(df: pd.DataFrame, max_months: int = 60) -> pd.DataFrame:
    """Expand loans into one row per observed month (discrete-time format).

    Returns a long frame with the hazard features, the time basis (period,
    period_sq, log_period), and the monthly event indicator ``event``.
    """
    df = df.copy()
    df["_dur"] = loan_duration_months(df).clip(upper=max_months)
    feats = [f for f in HAZARD_FEATURES if f in df.columns]

    base = df[feats + ["_dur", C.TARGET]].reset_index(drop=True)
    base["_loan"] = base.index

    # Repeat each loan _dur times; period = 1..dur
    durations = base["_dur"].to_numpy()
    loan_rep = np.repeat(base["_loan"].to_numpy(), durations)
    period = np.concatenate([np.arange(1, d + 1) for d in durations])

    pp = base.loc[loan_rep, feats].reset_index(drop=True)
    pp["period"] = period
    pp["period_sq"] = period ** 2
    pp["log_period"] = np.log(period)

    # Event = 1 only in the final observed month of a charged-off loan.
    is_last = period == np.repeat(durations, durations)
    defaulted = np.repeat(base[C.TARGET].to_numpy(), durations) == 1
    pp["event"] = (is_last & defaulted).astype(int)
    return pp


# --------------------------------------------------------------------------- #
# Hazard model
# --------------------------------------------------------------------------- #
def fit_hazard_model(person_period: pd.DataFrame) -> Pipeline:
    """Pooled logistic regression for the monthly discrete-time hazard."""
    feats = [f for f in HAZARD_FEATURES if f in person_period.columns] + TIME_COLS
    X = person_period[feats]
    y = person_period["event"]
    # No class_weight rebalancing: the discrete-time hazard must reflect the TRUE
    # (low) monthly default probability. Rebalancing would recalibrate the rare
    # per-period event to ~50% and saturate the cumulative PD toward 1.
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=C.RANDOM_STATE)),
    ])
    model.fit(X, y)
    model._feats = feats  # stash for prediction
    return model


# --------------------------------------------------------------------------- #
# Survival curves / lifetime PD
# --------------------------------------------------------------------------- #
def survival_curves(model: Pipeline, df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Per-loan term structure: hazard, survival, cumulative PD by month.

    Returns a long frame indexed by (loan position, period) with columns
    hazard, survival, cum_pd, marginal_pd. ``horizon`` is capped per loan at its
    contractual term.
    """
    df = df.copy().reset_index(drop=True)
    df["_term"] = df["term_months"].fillna(36).clip(upper=horizon).astype(int)
    feats_static = [f for f in HAZARD_FEATURES if f in df.columns]

    terms = df["_term"].to_numpy()
    loan_rep = np.repeat(df.index.to_numpy(), terms)
    period = np.concatenate([np.arange(1, t + 1) for t in terms])

    grid = df.loc[loan_rep, feats_static].reset_index(drop=True)
    grid["period"] = period
    grid["period_sq"] = period ** 2
    grid["log_period"] = np.log(period)
    grid["_loan"] = loan_rep

    hazard = model.predict_proba(grid[model._feats])[:, 1]
    grid["hazard"] = hazard
    # survival = cumulative product of (1 - hazard) within each loan
    grid["one_minus_h"] = 1.0 - grid["hazard"]
    grid["survival"] = grid.groupby("_loan")["one_minus_h"].cumprod()
    grid["cum_pd"] = 1.0 - grid["survival"]
    # marginal PD in month t = S(t-1) * h(t); S(0) = 1 for the first month
    grid["surv_prev"] = grid.groupby("_loan")["survival"].shift(1).fillna(1.0)
    grid["marginal_pd"] = grid["surv_prev"] * grid["hazard"]
    return grid[["_loan", "period", "hazard", "survival", "cum_pd", "marginal_pd"]]


def summarize_pd(curves: pd.DataFrame) -> pd.DataFrame:
    """Collapse the term structure into 12-month PD and lifetime PD per loan."""
    g = curves.groupby("_loan")
    lifetime = g["cum_pd"].last().rename("lifetime_pd")
    pd12 = (curves[curves["period"] <= 12].groupby("_loan")["cum_pd"].last()
            .rename("pd_12m"))
    out = pd.concat([pd12, lifetime], axis=1)
    out["pd_12m"] = out["pd_12m"].fillna(out["lifetime_pd"])
    return out


# --------------------------------------------------------------------------- #
# IFRS 9 staging + ECL
# --------------------------------------------------------------------------- #
def assign_ifrs9_stage(pd_summary: pd.DataFrame, defaulted: pd.Series,
                       sicr_multiple: float = 3.0,
                       abs_floor: float = 0.15) -> pd.Series:
    """Assign IFRS 9 stages (simplified, single-snapshot proxy for SICR).

    - Stage 3: already credit-impaired (defaulted).
    - Stage 2: Significant Increase in Credit Risk - here proxied as a lifetime
      PD that is high in absolute terms (>= abs_floor) AND well above the
      portfolio's origination median (>= sicr_multiple x median lifetime PD).
    - Stage 1: everything else (performing, no SICR).

    A real IFRS 9 SICR test compares each loan's *current* lifetime PD to the PD
    expected at origination using behavioural re-measurement; with a single
    snapshot we approximate that with a relative + absolute threshold and label
    it as such.
    """
    median_lt = pd_summary["lifetime_pd"].median()
    sicr = ((pd_summary["lifetime_pd"] >= abs_floor) &
            (pd_summary["lifetime_pd"] >= sicr_multiple * median_lt))
    stage = pd.Series(1, index=pd_summary.index, name="ifrs9_stage")
    stage[sicr] = 2
    stage[defaulted.reindex(pd_summary.index).fillna(0).astype(bool)] = 3
    return stage


def expected_credit_loss(curves: pd.DataFrame, ead: pd.Series, lgd: pd.Series | float,
                         annual_rate: pd.Series | float = 0.12,
                         horizon_months: int | None = None) -> pd.Series:
    """Discounted ECL = sum_t marginal_PD(t) x LGD x EAD / (1+r)^(t/12).

    ``horizon_months`` caps the sum (12 -> 12-month ECL; None -> lifetime).
    EAD/LGD/rate are aligned to the loan position index of ``curves``.
    """
    c = curves.copy()
    if horizon_months is not None:
        c = c[c["period"] <= horizon_months]
    if isinstance(annual_rate, (int, float)):
        rate = pd.Series(annual_rate, index=c["_loan"].unique())
    else:
        rate = annual_rate
    c = c.merge(rate.rename("_rate"), left_on="_loan", right_index=True, how="left")
    c["_rate"] = c["_rate"].fillna(0.12)
    c["discount"] = 1.0 / (1.0 + c["_rate"]) ** (c["period"] / 12.0)
    c["loss"] = c["marginal_pd"] * c["discount"]
    per_loan = c.groupby("_loan")["loss"].sum()

    lgd_s = lgd if isinstance(lgd, pd.Series) else pd.Series(lgd, index=per_loan.index)
    ead_s = ead.reindex(per_loan.index) if isinstance(ead, pd.Series) else pd.Series(ead, index=per_loan.index)
    lgd_s = lgd_s.reindex(per_loan.index).fillna(float(np.nanmean(lgd_s)))
    return per_loan * lgd_s * ead_s


if __name__ == "__main__":
    import json

    import joblib

    from .features import clean_frame
    from . import data as D

    print(f"Loading survival sample (n={C.SURVIVAL_SAMPLE:,}) ...")
    df = clean_frame(D.load_resolved(n=C.SURVIVAL_SAMPLE))
    train_idx, test_idx = D.vintage_holdout_split(df)

    print("Building person-period table + fitting hazard model ...")
    pp = build_person_period(df.loc[train_idx])
    print(f"  person-period rows: {len(pp):,}")
    model = fit_hazard_model(pp)

    print("Scoring survival curves on a test sample ...")
    score = df.loc[test_idx].head(5000).reset_index(drop=True)
    curves = survival_curves(model, score)
    pd_sum = summarize_pd(curves)
    stage = assign_ifrs9_stage(pd_sum, score[C.TARGET])

    from . import profit as P
    lgd_const = P.portfolio_lgd(df.loc[train_idx])
    ead = score["funded_amnt"]
    rate = (score["int_rate"] / 100.0) if "int_rate" in score else 0.12
    ecl_12m = expected_credit_loss(curves, ead, lgd_const, rate, horizon_months=12)
    ecl_life = expected_credit_loss(curves, ead, lgd_const, rate, horizon_months=None)

    summary = {
        "mean_pd_12m": float(pd_sum["pd_12m"].mean()),
        "mean_lifetime_pd": float(pd_sum["lifetime_pd"].mean()),
        "stage_counts": {int(k): int(v) for k, v in stage.value_counts().items()},
        "total_ecl_12m": float(ecl_12m.sum()),
        "total_ecl_lifetime": float(ecl_life.sum()),
        "portfolio_lgd": float(lgd_const),
    }
    print(json.dumps(summary, indent=2))

    joblib.dump(model, C.MODELS_DIR / "survival_hazard_model.pkl")
    (C.REPORTS_DIR / "survival_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved survival_hazard_model.pkl + survival_summary.json")
