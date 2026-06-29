"""Experiment: ANR model that sees OOF pd_score as an extra feature (no grade).

The "OOF" trick (Wolpert 1992 stacked generalisation):
  - On the TRAINING set: use 5-fold CV to produce out-of-fold pd_score
    predictions — each training loan's pd_score is predicted by a model
    that was fitted WITHOUT that loan, so there is no leakage.
  - On the TEST set: use the fully-fitted PD model (already in the parquet).

The ANR model then sees [origination features + pd_score] and can learn
"loans where our model says default risk is lower than LC's pricing implies."

Writes anr_pred_oof into test_with_pd.parquet and prints portfolio comparison.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/manoja/Documents/GitHub/Credit-Profit-Risk-Analysis/rebuild")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src import config as C, data as D, returns as R
from src.features import clean_frame, split_X_y
from src.resampling import build_model_pipeline
from src.train import tune_return_model, fit_return_model
from src.portfolio import policy_returns
from src.returns import POLICIES

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading full dataset …")
df = D.load_resolved(n=None)
df = clean_frame(df)

# Base features WITHOUT grade
X_base, y = split_X_y(df, include_grade=False)

train_idx, test_idx = D.vintage_holdout_split(
    df, train_max_year=C.OOT_TRAIN_MAX_YEAR, test_year=C.OOT_TEST_YEAR)
X_train_base = X_base.loc[train_idx]
X_test_base  = X_base.loc[test_idx]
y_train      = y.loc[train_idx]
df_train_full = df.loc[train_idx]
print(f"Train: {len(X_train_base):,}  |  Test: {len(X_test_base):,}")

# ── 2. Generate OOF pd_score for training set via 5-fold CV ──────────────────
print("Generating OOF pd_score on training set (5-fold CV) …")
# Load best PD params from the already-tuned model run
pd_params_path = C.MODELS_DIR / "best_params.json"
if pd_params_path.exists():
    pd_params = json.load(open(pd_params_path))
else:
    pd_params = dict(n_estimators=500, max_depth=4, learning_rate=0.12,
                     subsample=0.95, colsample_bytree=0.90, min_child_weight=3)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
oof_pd = np.zeros(len(X_train_base))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_base, y_train), 1):
    Xtr, Xval = X_train_base.iloc[tr_idx], X_train_base.iloc[val_idx]
    ytr = y_train.iloc[tr_idx]
    est = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        n_jobs=-1, random_state=C.RANDOM_STATE, verbosity=0,
                        **pd_params)
    pipe = build_model_pipeline(est, include_grade=False, use_smote=True)
    cal  = CalibratedClassifierCV(pipe, cv="prefit")  # skip re-calibrating in CV
    pipe.fit(Xtr, ytr)
    oof_pd[val_idx] = pipe.predict_proba(Xval)[:, 1]
    print(f"  Fold {fold}/5 done  (mean OOF pd_score = {oof_pd[val_idx].mean():.4f})")

print(f"OOF pd_score — mean={oof_pd.mean():.4f}  std={oof_pd.std():.4f}")

# ── 3. Build augmented feature matrices ───────────────────────────────────────
X_train_aug = X_train_base.copy()
X_train_aug["pd_score_oof"] = oof_pd

# For test: use the pd_score already in the parquet (fitted on full training set)
test_parquet = pd.read_parquet(C.PROCESSED_DIR / "test_with_pd.parquet")
X_test_aug   = X_test_base.copy()
X_test_aug["pd_score_oof"] = test_parquet["pd_score"].values

# ── 4. Train ANR model on augmented features (matured loans only) ─────────────
matured_mask = R.is_matured(df_train_full)
df_mat       = df_train_full[matured_mask]
X_mat_aug    = X_train_aug.loc[df_mat.index]
anr_mat      = R.realized_anr(df_mat)
print(f"Matured training loans: {len(X_mat_aug):,} / {len(X_train_base):,}")

print("Optuna tuning ANR model with OOF pd_score feature (20 trials) …")
ret_params = tune_return_model(X_mat_aug, anr_mat, arch="lightgbm", n_trials=20)
print(f"Best params: {ret_params}")

print("Fitting final OOF-augmented ANR model …")
return_model_oof = fit_return_model(X_mat_aug, anr_mat,
                                    arch="lightgbm", params=ret_params)

# ── 5. Predict and save ───────────────────────────────────────────────────────
anr_pred_oof = return_model_oof.predict(X_test_aug)
print(f"OOF-augmented mean predicted ANR: {anr_pred_oof.mean():+.4f}")
print(f"Single-stage mean predicted ANR:  {test_parquet['anr_pred'].mean():+.4f}")

test_parquet["anr_pred_oof"] = anr_pred_oof
test_parquet.to_parquet(C.PROCESSED_DIR / "test_with_pd.parquet")
joblib.dump(return_model_oof, C.MODELS_DIR / "return_model_oof.pkl")

# ── 6. Portfolio comparison ───────────────────────────────────────────────────
test_parquet["_anr"] = R.realized_anr(test_parquet)
matured = R.is_matured(test_parquet)
book    = test_parquet[matured].copy()

BUDGETS = [0.25, 0.50, 0.75]
policies_ext = dict(POLICIES)
policies_ext["profit_scoring_oof"] = ("anr_pred_oof", False)

results = policy_returns(book, BUDGETS, anr_col="_anr", policies=policies_ext)

print("\nPortfolio comparison — single-stage vs OOF pd_score ANR model:")
print(results.pivot_table(index="policy", columns="budget", values="port_anr")
      .to_string(float_format=lambda v: f"{v:+.4f}"))
print("\nDone. Saved return_model_oof.pkl and updated test_with_pd.parquet.")
