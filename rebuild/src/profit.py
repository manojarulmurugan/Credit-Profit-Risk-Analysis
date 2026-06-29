"""Profit-risk analysis using real Lending Club loan economics.

    EAD (Exposure at Default)  = funded_amnt
    LGD (Loss Given Default)   = 1 - recovery_rate, from realized principal +
                                 recoveries on charged-off loans
    EL  (Expected Loss)        = PD x LGD x EAD

It then builds an approval-rate vs expected-profit curve and finds the
business-cost-optimal approval threshold by sweeping the PD cutoff against
realized portfolio profit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# LGD / EAD / EL
# --------------------------------------------------------------------------- #
def realized_lgd(df: pd.DataFrame) -> pd.Series:
    """Per-loan realized LGD = principal lost / exposure, clipped to [0, 1].

    principal lost = funded_amnt - principal repaid - recoveries.
    Only meaningful for defaulted loans; returned for all rows for inspection.
    """
    ead = df["funded_amnt"].replace(0, np.nan)
    recovered_principal = df.get("total_rec_prncp", 0).fillna(0)
    recoveries = df.get("recoveries", 0).fillna(0)
    loss = df["funded_amnt"] - recovered_principal - recoveries
    lgd = (loss / ead).clip(lower=0, upper=1)
    return lgd.fillna(0)


def portfolio_lgd(df: pd.DataFrame) -> float:
    """Average realized LGD across actually-defaulted loans (the LGD assumption
    used for forward-looking expected loss on new applicants)."""
    bad = df[df[C.TARGET] == 1]
    if len(bad) == 0:
        return 0.5
    return float(realized_lgd(bad).mean())


def expected_loss(df: pd.DataFrame, pd_col: str = "pd_score",
                  lgd: float | pd.Series | None = None) -> pd.Series:
    """EL = PD x LGD x EAD.

    ``lgd`` may be a scalar (portfolio assumption) or a per-loan Series (from the
    LGD model). When omitted, a per-loan ``lgd_pred`` column is used if present,
    otherwise the portfolio average is the fallback.
    """
    if lgd is None:
        lgd = df["lgd_pred"] if "lgd_pred" in df.columns else portfolio_lgd(df)
    if isinstance(lgd, pd.Series):
        lgd = lgd.reindex(df.index)
    return df[pd_col] * lgd * df["funded_amnt"]


# --------------------------------------------------------------------------- #
# Realized per-loan profit (backtest on actual outcomes)
# --------------------------------------------------------------------------- #
def realized_profit(df: pd.DataFrame) -> pd.Series:
    """Net cash to the lender = total payments received - amount funded.

    Positive for fully-paid loans (interest earned), negative for charged-off
    loans (principal lost net of recoveries).
    """
    return df["total_pymnt"].fillna(0) - df["funded_amnt"].fillna(0)


def expected_profit(df: pd.DataFrame, pd_col: str = "pd_score",
                    lgd: float | None = None) -> pd.Series:
    """Model-based expected profit per loan if approved.

    expected_profit = (1 - PD) * scheduled_interest - PD * LGD * EAD
    scheduled_interest approximated from installment * term - funded_amnt.
    """
    if lgd is None:
        lgd = df["lgd_pred"] if "lgd_pred" in df.columns else portfolio_lgd(df)
    if isinstance(lgd, pd.Series):
        lgd = lgd.reindex(df.index)
    scheduled_total = df["installment"].fillna(0) * df["term_months"].fillna(0)
    scheduled_interest = (scheduled_total - df["funded_amnt"].fillna(0)).clip(lower=0)
    pd_ = df[pd_col]
    return (1 - pd_) * scheduled_interest - pd_ * lgd * df["funded_amnt"].fillna(0)


# --------------------------------------------------------------------------- #
# Approval-threshold optimization
# --------------------------------------------------------------------------- #
def profit_curve(df: pd.DataFrame, pd_col: str = "pd_score",
                 n_grid: int = 101) -> pd.DataFrame:
    """Sweep PD approval thresholds; report approval rate, realized profit, etc.

    A loan is approved when its PD <= threshold. Profit is the REALIZED profit of
    the approved subset (an honest backtest on observed outcomes).
    """
    df = df.copy()
    df["_profit"] = realized_profit(df)
    thresholds = np.linspace(0.0, 1.0, n_grid)
    rows = []
    n = len(df)
    for t in thresholds:
        approved = df[df[pd_col] <= t]
        rows.append({
            "threshold": t,
            "approval_rate": len(approved) / n,
            "total_profit": approved["_profit"].sum(),
            "avg_profit_per_loan": approved["_profit"].mean() if len(approved) else 0.0,
            "default_rate_approved": approved[C.TARGET].mean() if len(approved) else 0.0,
            "n_approved": len(approved),
        })
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> dict:
    """Pick the PD threshold that maximizes total realized profit."""
    best = curve.loc[curve["total_profit"].idxmax()]
    return best.to_dict()


# --------------------------------------------------------------------------- #
# Weight of Evidence / Information Value scorecard (optional, Basel/IFRS9 flavor)
# --------------------------------------------------------------------------- #
def woe_iv(feature: pd.Series, target: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Compute WoE/IV for a single feature (numeric -> quantile-binned)."""
    df = pd.DataFrame({"x": feature, "y": target}).dropna()
    if pd.api.types.is_numeric_dtype(df["x"]) and df["x"].nunique() > bins:
        df["bin"] = pd.qcut(df["x"], q=bins, duplicates="drop")
    else:
        df["bin"] = df["x"].astype(str)
    grp = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp.columns = ["total", "bad"]
    grp["good"] = grp["total"] - grp["bad"]
    total_bad = max(int(grp["bad"].sum()), 1)
    total_good = max(int(grp["good"].sum()), 1)
    grp["bad_rate"] = grp["bad"] / total_bad
    grp["good_rate"] = grp["good"] / total_good
    eps = 1e-6
    grp["woe"] = np.log((grp["good_rate"] + eps) / (grp["bad_rate"] + eps))
    grp["iv"] = (grp["good_rate"] - grp["bad_rate"]) * grp["woe"]
    return grp


def information_value(feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
    return float(woe_iv(feature, target, bins=bins)["iv"].sum())


if __name__ == "__main__":
    import json

    path = C.PROCESSED_DIR / "test_with_pd.parquet"
    if not path.exists():
        raise SystemExit("Run train.py first to produce the scored test set.")
    test = pd.read_parquet(path)
    lgd = portfolio_lgd(test)
    # Use the per-loan LGD model output when present (column lgd_pred), else the
    # portfolio constant. The profit curve itself uses REALIZED profit, so the
    # optimal threshold is unaffected by the LGD assumption.
    test["expected_loss"] = expected_loss(test)
    curve = profit_curve(test)
    best = optimal_threshold(curve)
    print(f"Portfolio LGD: {lgd:.3f}")
    print(f"Total expected loss (book): ${test['expected_loss'].sum():,.0f}")
    print(f"Total realized profit (all): ${realized_profit(test).sum():,.0f}")
    print("Optimal approval threshold:")
    for k, v in best.items():
        print(f"  {k:22s}: {v:,.4f}")

    # Persist artifacts: profit curve + economics for the Streamlit app.
    curve.to_csv(C.REPORTS_DIR / "profit_curve.csv", index=False)
    (C.MODELS_DIR / "economics.json").write_text(json.dumps({
        "portfolio_lgd": float(lgd),
        "optimal_threshold": float(best["threshold"]),
        "optimal_approval_rate": float(best["approval_rate"]),
        "optimal_total_profit": float(best["total_profit"]),
        "default_rate_approved": float(best["default_rate_approved"]),
    }, indent=2))
    print("\nSaved profit_curve.csv and economics.json")
