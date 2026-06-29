"""Re-fit the ANR return model with LightGBM (Optuna-tuned) on the full book.

Skips the expensive PD model re-train — the pd_score, lgd_pred, and all other
columns in test_with_pd.parquet are already correct from the latest run.
Only anr_pred is updated.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/manoja/Documents/GitHub/Credit-Profit-Risk-Analysis/rebuild")

import joblib
import pandas as pd
from src import config as C, data as D, returns as R
from src.features import clean_frame, split_X_y
from src.train import tune_return_model, fit_return_model

print("Loading full dataset (~1.3M rows)...")
df = D.load_resolved(n=None)
df = clean_frame(df)
X, y = split_X_y(df, include_grade=False)

train_idx, test_idx = D.vintage_holdout_split(
    df, train_max_year=C.OOT_TRAIN_MAX_YEAR, test_year=C.OOT_TEST_YEAR)
X_train = X.loc[train_idx]
X_test  = X.loc[test_idx]
df_train_full = df.loc[train_idx]
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

print("Filtering matured training loans...")
matured_mask  = R.is_matured(df_train_full)
df_train_mat  = df_train_full[matured_mask]
X_train_mat   = X_train.loc[df_train_mat.index]
anr_train     = R.realized_anr(df_train_mat)
print(f"Matured training loans: {len(X_train_mat):,} / {len(X_train):,} "
      f"({matured_mask.mean():.1%})")

print("Optuna tuning LightGBM return model (20 trials) ...")
ret_params = tune_return_model(X_train_mat, anr_train,
                               arch="lightgbm", n_trials=20)
print(f"Best params: {ret_params}")

print("Fitting final LightGBM return model ...")
return_model = fit_return_model(X_train_mat, anr_train,
                                arch="lightgbm", params=ret_params)
joblib.dump(return_model, C.MODELS_DIR / "return_model.pkl")
meta = {"arch": "lightgbm", **ret_params}
(C.MODELS_DIR / "return_model_params.json").write_text(
    json.dumps({k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in meta.items()}, indent=2))

print("Updating test_with_pd.parquet with new anr_pred ...")
test = pd.read_parquet(C.PROCESSED_DIR / "test_with_pd.parquet")
anr_pred = return_model.predict(X_test)
test["anr_pred"] = anr_pred
anr_realized = R.realized_anr(df.loc[test_idx])
print(f"Mean predicted ANR: {anr_pred.mean():+.4f}  vs realized {anr_realized.mean():+.4f}")
test.to_parquet(C.PROCESSED_DIR / "test_with_pd.parquet")
print("Done. Run: cd rebuild && make returns && make portfolio")
