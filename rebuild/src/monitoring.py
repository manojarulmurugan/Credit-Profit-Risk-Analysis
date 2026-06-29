"""Model monitoring: Population Stability Index (PSI) drift detection.

PSI is the banking-standard early-warning metric for population drift. It
compares a variable's distribution in a reference window against a current
window:

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

Standard bands: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant
shift requiring recalibration. PSI is the same divergence as the Information
Value used in the scorecard (profit.woe_iv), computed between two time windows
instead of between good/bad outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two samples of a single variable.

    Bin edges are quantiles of the reference (``expected``) sample so each
    reference bin holds ~10% of mass; the actual sample is bucketed into those
    same edges. Works for numeric variables; categorical handled by ``_psi_cat``.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if not pd.api.types.is_numeric_dtype(expected) or expected.nunique() <= bins:
        return _psi_cat(expected, actual)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_pct = np.clip(exp_pct, _EPS, None)
    act_pct = np.clip(act_pct, _EPS, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _psi_cat(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for a categorical / low-cardinality variable."""
    exp = expected.astype(str).value_counts(normalize=True)
    act = actual.astype(str).value_counts(normalize=True)
    cats = exp.index.union(act.index)
    exp_pct = np.clip(exp.reindex(cats).fillna(0).to_numpy(), _EPS, None)
    act_pct = np.clip(act.reindex(cats).fillna(0).to_numpy(), _EPS, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_band(value: float) -> str:
    """Human-readable PSI severity band."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"


def psi_report(reference: pd.DataFrame, current: pd.DataFrame,
               features: list[str], bins: int = 10) -> pd.DataFrame:
    """Per-feature PSI between a reference and a current population."""
    rows = []
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        value = psi(reference[f], current[f], bins=bins)
        rows.append({"feature": f, "psi": round(value, 4), "band": psi_band(value)})
    return (pd.DataFrame(rows).sort_values("psi", ascending=False)
            .reset_index(drop=True))


if __name__ == "__main__":
    import json

    import joblib

    from . import config as C
    from . import data as D
    from .features import clean_frame, split_X_y

    print("Loading data for PSI monitoring ...")
    df = clean_frame(D.load_resolved(n=C.TRAIN_SAMPLE))
    years = sorted(df["issue_year"].dropna().unique())
    cut = years[len(years) // 2]
    reference = df[df["issue_year"] < cut]
    current = df[df["issue_year"] >= cut]
    print(f"Reference (< {int(cut)}): {len(reference):,}  |  current (>= {int(cut)}): {len(current):,}")

    feats = ["loan_amnt", "annual_inc", "dti", "revol_util", "open_acc",
             "total_acc", "inq_last_6mths", "emp_length_num", "term_months"]
    tbl = psi_report(reference, current, feats)
    print(tbl.to_string(index=False))

    out = {"feature_psi": tbl.set_index("feature")["psi"].to_dict()}
    model_path = C.MODELS_DIR / "pd_model.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        Xref, _ = split_X_y(reference)
        Xcur, _ = split_X_y(current)
        sref = model.predict_proba(Xref)[:, 1]
        scur = model.predict_proba(Xcur)[:, 1]
        sp = psi(sref, scur)
        out["score_psi"] = float(sp)
        out["score_psi_band"] = psi_band(sp)
        print(f"\nPD score PSI: {sp:.4f} -> {psi_band(sp)}")

    (C.REPORTS_DIR / "monitoring_psi.json").write_text(json.dumps(out, indent=2))
    print("Saved reports/monitoring_psi.json")
