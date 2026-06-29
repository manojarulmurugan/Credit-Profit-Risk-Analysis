"""SHAP explainability for the PD model.

Provides global feature importance and per-applicant ("adverse action" style)
explanations. Works on the leakage-free pipeline by explaining the underlying
XGBoost estimator on the transformed feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _unwrap_pipeline(model):
    """Return (fitted_feature_pipeline, fitted_xgb) from a CalibratedClassifierCV
    or an imblearn Pipeline."""
    from sklearn.calibration import CalibratedClassifierCV

    pipe = model
    if isinstance(model, CalibratedClassifierCV):
        # Use the first calibrated sub-estimator's underlying pipeline.
        cc = model.calibrated_classifiers_[0]
        pipe = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
    # pipe is an imblearn Pipeline: feature steps + ('clf', xgb)
    clf = pipe.named_steps["clf"]
    feature_steps = [(n, t) for n, t in pipe.steps if n != "clf" and n != "smote"]
    return feature_steps, clf


def _transform(feature_steps, X):
    Xt = X
    for _, t in feature_steps:
        Xt = t.transform(Xt)
    return Xt


def global_importance(model, X_sample: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    """Mean |SHAP value| per transformed feature (global importance)."""
    import shap

    feature_steps, clf = _unwrap_pipeline(model)
    Xt = _transform(feature_steps, X_sample)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)
    importance = np.abs(shap_values).mean(axis=0)
    names = _feature_names(feature_steps, n=Xt.shape[1])
    out = (pd.DataFrame({"feature": names, "mean_abs_shap": importance})
           .sort_values("mean_abs_shap", ascending=False)
           .head(max_display)
           .reset_index(drop=True))
    return out


def explain_applicant(model, x_row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Per-applicant SHAP contributions (signed) for a single row."""
    import shap

    feature_steps, clf = _unwrap_pipeline(model)
    Xt = _transform(feature_steps, x_row)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)[0]
    names = _feature_names(feature_steps, n=len(shap_values))
    out = (pd.DataFrame({"feature": names, "shap": shap_values})
           .assign(abs_shap=lambda d: d["shap"].abs())
           .sort_values("abs_shap", ascending=False)
           .head(top_n)
           .drop(columns="abs_shap")
           .reset_index(drop=True))
    return out


def _feature_names(feature_steps, n: int) -> list[str]:
    """Best-effort transformed-feature names from the ColumnTransformer."""
    try:
        pre = dict(feature_steps)["preprocess"]
        names = list(pre.get_feature_names_out())
        # Subsequent steps (corr_drop, mi_select) subset columns via masks.
        for step_name, t in feature_steps:
            if hasattr(t, "keep_mask_"):
                names = [nm for nm, keep in zip(names, t.keep_mask_) if keep]
            elif hasattr(t, "get_support"):
                support = t.get_support()
                names = [nm for nm, keep in zip(names, support) if keep]
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f"f{i}" for i in range(n)]
