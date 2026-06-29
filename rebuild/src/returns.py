"""Profit-scoring target and return curves (the investor view of the book).

The PD/profit layer (``profit.py``) frames the problem as a *lender* deciding
whom to **accept or reject**. On a fully-priced, accepted-only LendingClub book
that decision is near-degenerate: interest already compensates risk, so rejecting
loans barely moves realized profit (see HANDOVER §5).

This module reframes the task the way the credit-risk literature does for
peer-to-peer data (Serrano-Cinca & Gutierrez-Nieto, EJOR 2016; Bastani et al.,
"How can lenders prosper", EJOR 2021): model the loan's **return**, not its
default, and treat it as an *investor* selecting a portfolio. The dependent
variable is the realized **Annualized Net Return (ANR)**, computed from the loan's
cash flows. ANR inherently penalizes early high-interest defaults, so the "the
interest already paid for the risk" effect that flattens the accept/reject story
is priced into the target itself.

Leakage note: ANR is a TARGET built from economics columns + dates (exactly like
the realized-LGD target in ``lgd.py``). Model FEATURES stay on the origination
allowlist; economics never enter X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import profit as P
from . import survival as S
# re-exported for convenience: WoE/IV are feature-agnostic scorecard helpers
from .profit import information_value, woe_iv  # noqa: F401


# --------------------------------------------------------------------------- #
# Maturity (calendar seasoning) — defines the honest backtest population
# --------------------------------------------------------------------------- #
def snapshot_date(df: pd.DataFrame, override=None) -> pd.Timestamp:
    """Data-snapshot date used to judge maturity.

    Defaults to ``config.SNAPSHOT_DATE``; when that is None, derive it from the
    data as the latest observed last-payment date.
    """
    override = C.SNAPSHOT_DATE if override is None else override
    if override is not None:
        return pd.Timestamp(override)
    if "last_pymnt_date" in df.columns and df["last_pymnt_date"].notna().any():
        return pd.Timestamp(df["last_pymnt_date"].max())
    # Fallback when last-payment dates are unavailable: use the latest origination
    # month as a conservative snapshot (fewer loans count as matured).
    return pd.Timestamp(df["issue_date"].max())


def is_matured(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.Series:
    """Boolean mask: loans whose contractual term fully elapsed before snapshot.

    A loan's economic life is only trustworthy once ``issue_date + term_months``
    has passed (otherwise late recoveries / unseasoned 60-month loans distort the
    realized P&L — the core of HANDOVER §5).
    """
    snap = snapshot or snapshot_date(df)
    issue_period = df["issue_date"].dt.year * 12 + df["issue_date"].dt.month
    snap_period = snap.year * 12 + snap.month
    term_end_period = issue_period + df["term_months"].fillna(36).astype(int)
    return term_end_period <= snap_period


# --------------------------------------------------------------------------- #
# Realized Annualized Net Return (the profit-scoring target)
# --------------------------------------------------------------------------- #
def realized_anr(df: pd.DataFrame) -> pd.Series:
    """Per-loan realized Annualized Net Return.

        ANR = (total_pymnt / funded_amnt) ** (12 / months_on_book) - 1

    ``months_on_book`` is the observed life (issue -> last payment, capped at
    term) reused from the survival module. The ratio is >= 0 (payments and
    exposure are non-negative), so the annualization never produces complex
    values; a total loss (no payments) yields ANR = -1 (-100%). Clipped to a sane
    consumer-loan band (``config.ANR_CLIP``).
    """
    months = S.loan_duration_months(df).clip(lower=1)
    funded = df["funded_amnt"].replace(0, np.nan)
    ratio = (df["total_pymnt"].fillna(0) / funded).clip(lower=0)
    anr = ratio ** (12.0 / months) - 1.0
    return anr.fillna(0.0).clip(lower=C.ANR_CLIP[0], upper=C.ANR_CLIP[1])


# --------------------------------------------------------------------------- #
# Return curve: portfolio frontier over the invested fraction
# --------------------------------------------------------------------------- #
def return_curve(df: pd.DataFrame, score_col: str, ascending: bool,
                 n_grid: int = 101, anr_col: str = "_anr") -> pd.DataFrame:
    """Sweep the invested fraction; report realized portfolio return at each level.

    Loans are ranked by ``score_col`` (``ascending=True`` invests in the lowest
    scores first — e.g. lowest PD or lowest rate; ``False`` invests in the highest
    first — e.g. highest predicted ANR). For each budget fraction we "invest" in
    the top slice and report:

    - ``port_anr``  : capital-weighted realized annualized return (what an
                      investor actually earns) — sum(ANR*funded)/sum(funded)
    - ``avg_anr``   : equal-weighted mean realized ANR
    - ``total_dollar_return`` : realized profit dollars (total_pymnt - funded)
    - ``default_rate_invested`` : realized default rate of the invested slice
    """
    df = df.copy()
    if anr_col not in df.columns:
        df[anr_col] = realized_anr(df)
    df["_profit"] = P.realized_profit(df)
    df = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)

    funded = df["funded_amnt"].to_numpy()
    anr = df[anr_col].to_numpy()
    profit = df["_profit"].to_numpy()
    bad = df[C.TARGET].to_numpy()
    n = len(df)

    rows = []
    for f in np.linspace(0.0, 1.0, n_grid):
        k = int(round(f * n))
        if k == 0:
            continue
        cap = funded[:k].sum()
        rows.append({
            "fraction_invested": k / n,
            "n_invested": k,
            "port_anr": float((anr[:k] * funded[:k]).sum() / cap) if cap else 0.0,
            "avg_anr": float(anr[:k].mean()),
            "total_dollar_return": float(profit[:k].sum()),
            "default_rate_invested": float(bad[:k].mean()),
        })
    return pd.DataFrame(rows)


def optimal_fraction(curve: pd.DataFrame) -> dict:
    """Invested fraction maximizing the capital-weighted annualized portfolio return."""
    best = curve.loc[curve["port_anr"].idxmax()]
    return best.to_dict()


# Ranking policies shared with portfolio.py. (column, ascending) — ascending=True
# means "invest in the lowest score first".
POLICIES = {
    "invest_all": (None, None),
    "default_scoring": ("pd_score", True),    # lowest PD first
    "profit_scoring": ("anr_pred", False),    # highest predicted return first
    "grade_only": ("int_rate", True),         # LC's own risk order: safest first
}


if __name__ == "__main__":
    import argparse, json

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--suffix", default="", help="Parquet suffix, e.g. '_with_grade'")
    _args = _ap.parse_args()

    path = C.PROCESSED_DIR / f"test_with_pd{_args.suffix}.parquet"
    if not path.exists():
        raise SystemExit(f"Run train.py first to produce {path.name}.")
    test = pd.read_parquet(path)
    test["_anr"] = realized_anr(test)

    matured = is_matured(test)
    book = test[matured].copy()
    # invest-all baseline: capital-weighted return of the whole matured book
    cap = book["funded_amnt"].sum()
    baseline_anr = float((book["_anr"] * book["funded_amnt"]).sum() / cap)
    print(f"Scored test loans: {len(test):,}  |  matured (honest backtest): "
          f"{len(book):,} ({matured.mean():.1%})")
    print(f"Invest-all baseline port_ANR (matured book): {baseline_anr:+.4f}")

    curves, summary = [], {"invest_all": {"port_anr": baseline_anr}}
    # Ranking policies only (invest_all is the fraction=1.0 endpoint / baseline).
    for name, (col, asc) in POLICIES.items():
        if name == "invest_all":
            continue
        if col not in book.columns:
            print(f"  ! skipping '{name}': column '{col}' not in scored set")
            continue
        curve = return_curve(book, score_col=col, ascending=asc)
        curve["policy"] = name
        curves.append(curve)
        opt = optimal_fraction(curve)
        at50 = curve.iloc[(curve["fraction_invested"] - 0.5).abs().idxmin()]
        summary[name] = {
            "port_anr_at_50pct": float(at50["port_anr"]),
            "default_rate_at_50pct": float(at50["default_rate_invested"]),
            "opt_fraction": float(opt["fraction_invested"]),
            "opt_total_dollar_return": float(opt["total_dollar_return"]),
        }
        print(f"  {name:16s} port_ANR@50%={at50['port_anr']:+.4f}  "
              f"(vs invest-all {baseline_anr:+.4f})  "
              f"default_rate@50%={at50['default_rate_invested']:.3f}")

    sfx = _args.suffix
    pd.concat(curves, ignore_index=True).to_csv(
        C.REPORTS_DIR / f"return_curve{sfx}.csv", index=False)
    (C.REPORTS_DIR / f"return_portfolio_comparison{sfx}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved return_curve{sfx}.csv and return_portfolio_comparison{sfx}.json")
