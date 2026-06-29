"""Experiment: Two-stage ANR return model (no grade features).

E[ANR] = (1 - pd_score) * E[ANR | paid] + pd_score * E[ANR | default]

Trains two separate LightGBM regressors on matured training loans:
  - model_paid:    fitted on Fully Paid loans only
  - model_default: fitted on Charged Off / Default loans only

At test time combines using pd_score from the already-fitted PD model.
Writes anr_pred_twostage into test_with_pd.parquet and then prints
portfolio comparison so we can compare directly to the single-stage result.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/manoja/Documents/GitHub/Credit-Profit-Risk-Analysis/rebuild")

import joblib, json
import numpy as np
import pandas as pd
from src import config as C, data as D, returns as R
from src.features import clean_frame, split_X_y
from src.train import _make_return_estimator
from src.resampling import build_model_pipeline
from src.portfolio import policy_returns

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading full dataset …")
df = D.load_resolved(n=None)
df = clean_frame(df)
X, y = split_X_y(df, include_grade=False)

train_idx, test_idx = D.vintage_holdout_split(
    df, train_max_year=C.OOT_TRAIN_MAX_YEAR, test_year=C.OOT_TEST_YEAR)
X_train = X.loc[train_idx]
X_test  = X.loc[test_idx]
df_train_full = df.loc[train_idx]
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 2. Matured training loans only ────────────────────────────────────────────
matured_mask = R.is_matured(df_train_full)
df_mat       = df_train_full[matured_mask]
X_mat        = X_train.loc[df_mat.index]
anr_mat      = R.realized_anr(df_mat)

paid_mask    = df_mat["loan_status"].isin(["Fully Paid"])
default_mask = df_mat["loan_status"].isin(["Charged Off", "Default"])
print(f"Matured: {len(X_mat):,}  →  paid: {paid_mask.sum():,}  |  "
      f"defaulted: {default_mask.sum():,}")

# ── 3. Train sub-models (default LightGBM params — no Optuna for speed) ──────
BASE_PARAMS = dict(n_estimators=400, num_leaves=63, learning_rate=0.05,
                   subsample=0.9, colsample_bytree=0.8, min_child_samples=30)

print("Fitting E[ANR | paid] model …")
est_paid  = _make_return_estimator("lightgbm", BASE_PARAMS)
pipe_paid = build_model_pipeline(est_paid, include_grade=False, use_smote=False)
pipe_paid.fit(X_mat[paid_mask], anr_mat[paid_mask])

print("Fitting E[ANR | default] model …")
est_def  = _make_return_estimator("lightgbm", BASE_PARAMS)
pipe_def = build_model_pipeline(est_def, include_grade=False, use_smote=False)
pipe_def.fit(X_mat[default_mask], anr_mat[default_mask])

# ── 4. Predict on test set: E[ANR] = (1-pd)*E[paid] + pd*E[default] ─────────
print("Computing two-stage ANR predictions on test set …")
test = pd.read_parquet(C.PROCESSED_DIR / "test_with_pd.parquet")
pd_score = test["pd_score"].values
pred_paid    = pipe_paid.predict(X_test)
pred_default = pipe_def.predict(X_test)
anr_twostage = (1 - pd_score) * pred_paid + pd_score * pred_default

print(f"Two-stage mean predicted ANR: {anr_twostage.mean():+.4f}")
print(f"Single-stage mean predicted ANR (existing): {test['anr_pred'].mean():+.4f}")
print(f"Realized ANR (matured test): {R.realized_anr(df.loc[test_idx]).mean():+.4f}")

test["anr_pred_twostage"] = anr_twostage
test.to_parquet(C.PROCESSED_DIR / "test_with_pd.parquet")

# ── 5. Portfolio comparison: single-stage vs two-stage ────────────────────────
test["_anr"] = R.realized_anr(test)
matured      = R.is_matured(test)
book         = test[matured].copy()

from src.returns import POLICIES
BUDGETS = [0.25, 0.50, 0.75]

# Extend base policies with the two-stage profit scoring column
policies_ext = dict(POLICIES)
policies_ext["profit_scoring_2stage"] = ("anr_pred_twostage", False)

results = policy_returns(book, BUDGETS, anr_col="_anr", policies=policies_ext)

print("\nPortfolio comparison — single-stage vs two-stage ANR model:")
print(results.pivot_table(index="policy", columns="budget", values="port_anr")
      .to_string(float_format=lambda v: f"{v:+.4f}"))

# Save sub-models
joblib.dump(pipe_paid,  C.MODELS_DIR / "return_model_twostage_paid.pkl")
joblib.dump(pipe_def,   C.MODELS_DIR / "return_model_twostage_default.pkl")
meta = {"arch": "lightgbm_twostage", **BASE_PARAMS}
(C.MODELS_DIR / "return_model_twostage_params.json").write_text(
    json.dumps(meta, indent=2))
print("\nDone. Saved two-stage models and updated test_with_pd.parquet.")
