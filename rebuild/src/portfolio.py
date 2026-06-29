"""Portfolio-selection backtest: default scoring vs profit scoring vs grade.

This is the headline experiment of the return arm. On the same out-of-time
hold-out used everywhere else, I compare four *investor* policies that each rank
the candidate loans and invest in the top slice under a budget:

    invest_all       - the baseline (buy the whole book)
    default_scoring  - rank by the PD model (lowest default risk first)
    profit_scoring   - rank by the predicted ANR (highest return first)
    grade_only       - rank by LendingClub's own int_rate (safest first)

and measure the **realized annualized portfolio return** each one earns. The
literature (Serrano-Cinca 2016; "How can lenders prosper", 2021) predicts profit
scoring should win, because the drivers of default differ from the drivers of
return.

Maturity (HANDOVER §5) is handled honestly:
    PRIMARY    - realized ANR on loans whose term has fully run (seasoned book).
    ROBUSTNESS - a survival-adjusted *expected* ANR for the unmatured loans
                 (reusing the discrete-time hazard model), folded back in so the
                 comparison can use the whole cohort without trusting immature
                 realized cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import returns as R
from . import survival as S


def policy_returns(df: pd.DataFrame, budget_fractions, anr_col: str,
                   policies: dict = None) -> pd.DataFrame:
    """Realized/expected portfolio return for each policy at each budget.

    ``anr_col`` is the per-loan return column to credit (realized for the matured
    book, survival-adjusted for the robustness run). Returns a tidy frame with the
    capital-weighted portfolio return, equal-weighted mean, default rate, and size.
    """
    policies = policies or R.POLICIES
    funded = df["funded_amnt"].to_numpy()
    rows = []
    for name, (col, asc) in policies.items():
        if name == "invest_all":
            order = np.arange(len(df))
        elif col not in df.columns:
            continue
        else:
            order = df[col].to_numpy().argsort(kind="mergesort")
            if not asc:
                order = order[::-1]
        anr = df[anr_col].to_numpy()[order]
        fnd = funded[order]
        bad = df[C.TARGET].to_numpy()[order]
        n = len(df)
        for b in budget_fractions:
            k = n if name == "invest_all" else int(round(b * n))
            if k == 0:
                continue
            cap = fnd[:k].sum()
            rows.append({
                "policy": name,
                "budget": 1.0 if name == "invest_all" else b,
                "n_invested": k,
                "port_anr": float((anr[:k] * fnd[:k]).sum() / cap) if cap else 0.0,
                "avg_anr": float(anr[:k].mean()),
                "default_rate_invested": float(bad[:k].mean()),
            })
            if name == "invest_all":
                break
    return pd.DataFrame(rows)


def survival_adjusted_anr(df: pd.DataFrame, hazard_model) -> pd.Series:
    """Expected ANR for loans whose realized cash is not yet trustworthy.

        E[ANR] = (1 - lifetime_PD) * contractual_rate - lifetime_PD * LGD

    Lifetime PD comes from the discrete-time survival hazard model; the
    contractual rate is LC's ``int_rate`` and LGD the per-loan ``lgd_pred``. This
    keeps unmatured loans in the experiment without crediting them immature
    realized profit. Returned aligned to ``df.index``.
    """
    work = df.reset_index(drop=True)
    curves = S.survival_curves(hazard_model, work)
    lifetime_pd = S.summarize_pd(curves)["lifetime_pd"].reindex(work.index).fillna(0.0)
    rate = (work["int_rate"] / 100.0) if "int_rate" in work else pd.Series(0.12, index=work.index)
    lgd = work["lgd_pred"] if "lgd_pred" in work else pd.Series(0.55, index=work.index)
    exp_anr = (1.0 - lifetime_pd) * rate - lifetime_pd * lgd
    exp_anr = exp_anr.clip(lower=C.ANR_CLIP[0], upper=C.ANR_CLIP[1])
    exp_anr.index = df.index
    return exp_anr


if __name__ == "__main__":
    import argparse, joblib

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--suffix", default="", help="Parquet suffix, e.g. '_with_grade'")
    _args = _ap.parse_args()

    BUDGETS = [0.25, 0.5, 0.75]
    path = C.PROCESSED_DIR / f"test_with_pd{_args.suffix}.parquet"
    if not path.exists():
        raise SystemExit(f"Run train.py first to produce {path.name}.")
    test = pd.read_parquet(path)
    test["_anr"] = R.realized_anr(test)
    matured = R.is_matured(test)

    print(f"Scored test loans: {len(test):,}  |  matured: {matured.sum():,} "
          f"({matured.mean():.1%})")

    # PRIMARY: realized ANR on the seasoned (matured) book
    primary = policy_returns(test[matured], BUDGETS, anr_col="_anr")
    primary.insert(0, "population", "matured_realized")

    tables = [primary]

    # ROBUSTNESS: realized ANR where matured, survival-adjusted expected ANR where not
    haz_path = C.MODELS_DIR / "survival_hazard_model.pkl"
    if haz_path.exists() and "int_rate" in test.columns:
        hazard = joblib.load(haz_path)
        blended = test.copy()
        unmatured = ~matured
        if unmatured.any():
            blended.loc[unmatured, "_anr"] = survival_adjusted_anr(
                blended.loc[unmatured], hazard)
        robust = policy_returns(blended, BUDGETS, anr_col="_anr")
        robust.insert(0, "population", "blended_survival_adjusted")
        tables.append(robust)
    else:
        print("  ! skipping survival-adjusted robustness "
              "(need survival_hazard_model.pkl + int_rate column)")

    sfx = _args.suffix
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(C.REPORTS_DIR / f"portfolio_comparison{sfx}.csv", index=False)

    print("\nPortfolio comparison (capital-weighted annualized return):")
    for pop, grp in out.groupby("population", sort=False):
        print(f"\n  [{pop}]")
        pivot = grp.pivot_table(index="policy", columns="budget", values="port_anr")
        print(pivot.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"\nSaved portfolio_comparison{sfx}.csv")
