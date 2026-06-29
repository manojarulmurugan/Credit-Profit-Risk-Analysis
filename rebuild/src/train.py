"""Model training: model zoo, stacking, Optuna tuning, and probability calibration.

The zoo spans Logistic Regression, Decision Tree, Random Forest, SVM
(linear/poly/rbf/sigmoid), MLP, Naive Bayes, XGBoost, LightGBM, CatBoost, and a
Stacking ensemble. Every model trains inside the leakage-free pipeline
(preprocess -> corr drop -> MI select -> SMOTE -> clf) and is scored with
cross-validated AUC. Optuna runs hyper-parameter search on the training folds
only, and the final model's probabilities are isotonically calibrated so the PD
feeds the profit layer correctly.
"""

from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from . import config as C
from . import data as D
from . import evaluate as E
from .features import clean_frame, split_X_y
from .resampling import build_model_pipeline

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def base_estimators() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=9, random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=rs),
        # probability=False: the ROC-AUC scorer falls back to decision_function,
        # which is far faster (no internal calibration CV). SVMs are part of the
        # zoo comparison only; the headline models are XGBoost / stacking.
        "SVM (linear)": SVC(kernel="linear", random_state=rs),
        "SVM (poly)": SVC(kernel="poly", degree=3, random_state=rs),
        "SVM (rbf)": SVC(kernel="rbf", random_state=rs),
        "SVM (sigmoid)": SVC(kernel="sigmoid", random_state=rs),
        "MLP": MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500,
                             random_state=rs),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=rs),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=-1, num_leaves=31, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=rs,
            verbose=-1),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1, random_state=rs,
            verbose=False, allow_writing_files=False),
    }


def build_stacking() -> StackingClassifier:
    """Stacking ensemble with XGBoost meta-learner (the original headline model)."""
    rs = C.RANDOM_STATE
    level0 = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=rs)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=9, random_state=rs)),
        ("Random Forest", RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                                  random_state=rs)),
        ("Naive Bayes", GaussianNB()),
    ]
    meta = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                         eval_metric="logloss", n_jobs=-1, random_state=rs)
    return StackingClassifier(estimators=level0, final_estimator=meta,
                              stack_method="predict_proba", n_jobs=-1)


# --------------------------------------------------------------------------- #
# Cross-validated zoo comparison
# --------------------------------------------------------------------------- #
def run_zoo(X, y, include_grade: bool = False, k: int | str = "all",
            cv_folds: int = 3) -> pd.DataFrame:
    """Cross-validated ROC-AUC for each model in the zoo (+ stacking)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)
    rows = {}
    models = base_estimators()
    models["Stacking (XGB meta)"] = build_stacking()
    for name, est in models.items():
        pipe = build_model_pipeline(est, include_grade=include_grade, k=k)
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        rows[name] = {"cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()}
        print(f"  {name:24s} AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
    return pd.DataFrame(rows).T.sort_values("cv_auc_mean", ascending=False)


# --------------------------------------------------------------------------- #
# Optuna tuning for the headline XGBoost model (training folds only)
# --------------------------------------------------------------------------- #
def tune_xgboost(X, y, n_trials: int = 25, include_grade: bool = False,
                 cv_folds: int = 3) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        est = XGBClassifier(eval_metric="logloss", n_jobs=-1,
                            random_state=C.RANDOM_STATE, **params)
        pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=True)
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return study.best_params


# --------------------------------------------------------------------------- #
# Fit + calibrate the final production model
# --------------------------------------------------------------------------- #
def fit_calibrated_xgb(X, y, params: dict | None = None,
                       include_grade: bool = False) -> CalibratedClassifierCV:
    """Fit the leakage-free XGBoost pipeline, then calibrate its probabilities."""
    params = params or {}
    est = XGBClassifier(eval_metric="logloss", n_jobs=-1,
                        random_state=C.RANDOM_STATE, **params)
    pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=True)
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calibrated.fit(X, y)
    return calibrated


# --------------------------------------------------------------------------- #
# Profit-scoring head: architecture-aware zoo, tuning, and fit
# --------------------------------------------------------------------------- #

# Architecture keys used throughout. These are the canonical short names passed
# between zoo → tune → fit so callers never deal with raw class names.
RETURN_ARCHS = ("lightgbm", "xgboost", "catboost", "random_forest", "ridge", "elasticnet")
_RETURN_DISPLAY = {
    "lightgbm":     "LightGBM (Huber)",
    "xgboost":      "XGBoost",
    "catboost":     "CatBoost",
    "random_forest": "Random Forest",
    "ridge":        "Ridge",
    "elasticnet":   "ElasticNet",
}


def _make_return_estimator(arch: str, params: dict | None = None):
    """Construct a fresh (unfitted) regressor for the given architecture."""
    _p = dict(params) if params else {}
    rs = C.RANDOM_STATE
    if arch == "lightgbm":
        base = {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.9}
        base.update(_p)
        return LGBMRegressor(objective="huber", alpha=0.9, n_jobs=-1,
                             random_state=rs, verbose=-1, **base)
    elif arch == "xgboost":
        base = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.9}
        base.update(_p)
        return XGBRegressor(n_jobs=-1, random_state=rs, verbosity=0, **base)
    elif arch == "catboost":
        base = {"iterations": 400, "depth": 6, "learning_rate": 0.05}
        base.update(_p)
        return CatBoostRegressor(random_state=rs, verbose=False,
                                 allow_writing_files=False, **base)
    elif arch == "random_forest":
        base = {"n_estimators": 200}
        base.update(_p)
        return RandomForestRegressor(n_jobs=-1, random_state=rs, **base)
    elif arch == "ridge":
        base = {"alpha": 1.0}
        base.update(_p)
        return Ridge(**base)
    elif arch == "elasticnet":
        base = {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 2000}
        base.update(_p)
        return ElasticNet(**base)
    else:
        raise ValueError(f"Unknown return model architecture: '{arch}'. "
                         f"Choose from {RETURN_ARCHS}")


def return_model_zoo(X: pd.DataFrame, anr: pd.Series, include_grade: bool = False,
                     cv_folds: int = 3) -> pd.DataFrame:
    """Compare all return-model architectures at default params.

    Primary metric: Spearman rank correlation between predicted and realized ANR.
    Spearman is correct here because the model is used as a *ranker* in the
    portfolio backtest - ordinal correctness matters more than point accuracy.
    A fresh estimator is cloned for each fold to avoid state contamination.
    """
    from scipy.stats import spearmanr

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)
    rows = {}
    for arch in RETURN_ARCHS:
        est_template = _make_return_estimator(arch)
        spearmans, rmses = [], []
        ok = True
        for tr_idx, val_idx in cv.split(X):
            Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
            ytr, yval = anr.iloc[tr_idx], anr.iloc[val_idx]
            try:
                pipe = build_model_pipeline(clone(est_template),
                                            include_grade=include_grade, use_smote=False)
                pipe.fit(Xtr, ytr)
                preds = pipe.predict(Xval)
                sp, _ = spearmanr(yval.to_numpy(), preds)
                spearmans.append(float(sp))
                rmses.append(float(np.sqrt(np.mean((yval.to_numpy() - preds) ** 2))))
            except Exception as exc:
                print(f"    ! {_RETURN_DISPLAY[arch]} error: {exc}")
                ok = False
                break
        if ok and spearmans:
            rows[arch] = {
                "display_name":   _RETURN_DISPLAY[arch],
                "spearman_mean":  float(np.mean(spearmans)),
                "spearman_std":   float(np.std(spearmans)),
                "rmse_mean":      float(np.mean(rmses)),
                "rmse_std":       float(np.std(rmses)),
            }
            print(f"  {_RETURN_DISPLAY[arch]:24s}  "
                  f"Spearman={np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}  "
                  f"RMSE={np.mean(rmses):.5f}")

    df = pd.DataFrame(rows).T.sort_values("spearman_mean", ascending=False)
    return df


def tune_return_model(X, anr, arch: str = "lightgbm", n_trials: int = 20,
                      include_grade: bool = False, cv_folds: int = 3) -> dict:
    """Optuna search for the ANR regressor. Minimises cross-validated RMSE.

    Fresh estimator per fold (via _make_return_estimator inside the loop) so
    consecutive folds don't warm-start off each other's fitted weights.
    Ridge/ElasticNet have no meaningful Optuna search space and are skipped.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)

    def _trial_params(trial) -> dict:
        if arch == "lightgbm":
            return {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 800, step=100),
                "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
                "max_depth":         trial.suggest_int("max_depth", 3, 12),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }
        elif arch == "xgboost":
            return {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }
        elif arch == "catboost":
            return {
                "iterations":   trial.suggest_int("iterations", 200, 800, step=100),
                "depth":        trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        elif arch == "random_forest":
            return {
                "n_estimators":    trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth":       trial.suggest_int("max_depth", 4, 14),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
                "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        else:
            raise ValueError(f"arch '{arch}' has no Optuna search space.")

    def objective(trial):
        params = _trial_params(trial)
        rmses = []
        for tr_idx, val_idx in cv.split(X):
            Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
            ytr, yval = anr.iloc[tr_idx], anr.iloc[val_idx]
            est = _make_return_estimator(arch, params)  # fresh each fold
            pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=False)
            pipe.fit(Xtr, ytr)
            preds = pipe.predict(Xval)
            rmses.append(float(np.sqrt(np.mean((yval.to_numpy() - preds) ** 2))))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV RMSE ({_RETURN_DISPLAY.get(arch, arch)}): {study.best_value:.5f}")
    return study.best_params


def fit_return_model(X, anr, include_grade: bool = False,
                     arch: str = "lightgbm", params: dict | None = None):
    """Fit the final return model with the chosen architecture and tuned params."""
    est = _make_return_estimator(arch, params)
    pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=False)
    pipe.fit(X, anr)
    return pipe


# --------------------------------------------------------------------------- #
# Optuna tuning for Logistic Regression and Random Forest finalists
# --------------------------------------------------------------------------- #
def tune_logistic_regression(X, y, n_trials: int = 25, include_grade: bool = False,
                              cv_folds: int = 3) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
        # max_iter capped during tuning (saga is convergence-limited on the
        # SMOTE-expanded one-hot matrix); the chosen params are refit at higher
        # max_iter for the final model. The AUC ranking is stable under the cap.
        est = LogisticRegression(solver="saga", max_iter=400,
                                 random_state=C.RANDOM_STATE, **params)
        pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=True)
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return study.best_params


def tune_random_forest(X, y, n_trials: int = 25, include_grade: bool = False,
                       cv_folds: int = 3) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }
        est = RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, **params)
        pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=True)
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return study.best_params


def tune_lightgbm(X, y, n_trials: int = 25, include_grade: bool = False,
                  cv_folds: int = 3) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        est = LGBMClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, verbose=-1, **params)
        pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=True)
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return study.best_params


def tune_catboost(X, y, n_trials: int = 25, include_grade: bool = False,
                  cv_folds: int = 3) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=C.RANDOM_STATE)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        est = CatBoostClassifier(random_state=C.RANDOM_STATE, verbose=False,
                                 allow_writing_files=False, **params)
        pipe = build_model_pipeline(est, include_grade=include_grade, use_smote=True)
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return study.best_params


def fit_calibrated_model(X, y, clf, include_grade: bool = False) -> CalibratedClassifierCV:
    """Fit a leakage-free Pipeline for any classifier, then calibrate probabilities."""
    pipe = build_model_pipeline(clf, include_grade=include_grade, use_smote=True)
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calibrated.fit(X, y)
    return calibrated


# --------------------------------------------------------------------------- #
# Finalist comparison: Logistic Regression vs Random Forest vs XGBoost
# All three Optuna-tuned on identical data, calibrated, evaluated on same hold-out
# --------------------------------------------------------------------------- #
def run_finalist_comparison(X_train, X_test, y_train, y_test,
                            include_grade: bool = False, n_trials: int = 25) -> pd.DataFrame:
    """Tune LR, RF, and XGBoost with Optuna under identical conditions.

    All three share the same X_train / X_test split, the same Pipeline structure,
    and the same isotonic calibration, so the comparison is apples-to-apples.
    Results are saved to reports/finalist_comparison.json.
    """
    rows = []

    tuners = [
        ("Logistic Regression", LogisticRegression, tune_logistic_regression),
        ("Random Forest",       RandomForestClassifier, tune_random_forest),
        ("XGBoost",             XGBClassifier,          tune_xgboost),
        ("LightGBM",            LGBMClassifier,         tune_lightgbm),
        ("CatBoost",            CatBoostClassifier,     tune_catboost),
    ]

    models_out = {}
    for name, clf_class, tune_fn in tuners:
        print(f"\n  [{name}] Optuna tuning ({n_trials} trials) ...")
        best = tune_fn(X_train, y_train, n_trials=n_trials,
                       include_grade=include_grade)
        print(f"  [{name}] Best params: {best}")

        if clf_class is XGBClassifier:
            clf = XGBClassifier(eval_metric="logloss", n_jobs=-1,
                                random_state=C.RANDOM_STATE, **best)
        elif clf_class is LogisticRegression:
            # solver must be saga - supports both l1 and l2 penalties
            clf = LogisticRegression(solver="saga", max_iter=2000,
                                     random_state=C.RANDOM_STATE, **best)
        elif clf_class is LGBMClassifier:
            clf = LGBMClassifier(n_jobs=-1, random_state=C.RANDOM_STATE,
                                 verbose=-1, **best)
        elif clf_class is CatBoostClassifier:
            clf = CatBoostClassifier(random_state=C.RANDOM_STATE, verbose=False,
                                     allow_writing_files=False, **best)
        else:
            clf = RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, **best)

        print(f"  [{name}] Fitting calibrated model ...")
        model = fit_calibrated_model(X_train, y_train, clf,
                                     include_grade=include_grade)
        models_out[name] = model

        pd_scores = model.predict_proba(X_test)[:, 1]
        metrics = E.credit_metrics(y_test, pd_scores)
        rows.append({
            "Model": name,
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "KS":      round(metrics["ks"],      4),
            "Gini":    round(metrics["gini"],    4),
            "PR-AUC":  round(metrics["pr_auc"],  4),
            "Brier":   round(metrics["brier"],   4),
        })
        print(f"  [{name}] AUC={metrics['roc_auc']:.4f}  KS={metrics['ks']:.4f}"
              f"  Gini={metrics['gini']:.4f}  Brier={metrics['brier']:.4f}")

        suffix = name.lower().replace(" ", "_")
        joblib.dump(model, C.MODELS_DIR / f"finalist_{suffix}.pkl")
        best_to_save = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in best.items()}
        (C.MODELS_DIR / f"finalist_{suffix}_params.json").write_text(
            json.dumps(best_to_save, indent=2))

    df = pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
    (C.REPORTS_DIR / "finalist_comparison.json").write_text(
        df.reset_index().to_json(orient="records", indent=2))
    return df, models_out


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #
def main(zoo_sample: int | None = None, train_sample: int | None = None,
         include_grade: bool = False, n_trials: int = 25, run_full_zoo: bool = True,
         split_mode: str = None, with_return: bool = True, return_trials: int = 20,
         run_return_zoo: bool = False, return_arch: str | None = None,
         test_year: int | None = None):
    zoo_sample = zoo_sample or C.ZOO_SAMPLE
    # 0 is the CLI sentinel for "full book"; None falls back to config default.
    if train_sample is None:
        train_sample = C.TRAIN_SAMPLE
    elif train_sample == 0:
        train_sample = None  # load_resolved(n=None) uses the full dataset
    split_mode = split_mode or C.SPLIT_MODE

    sample_label = "full" if train_sample is None else f"{train_sample:,}"
    print(f"[1/5] Loading data (train_sample={sample_label}, split={split_mode}) ...")
    df = D.load_resolved(n=train_sample)
    df = clean_frame(df)
    X, y = split_X_y(df, include_grade=include_grade)
    print(f"      X shape={X.shape}, default rate={y.mean():.3f}")

    # Hold-out test set (split FIRST, before any fitting). Out-of-time (default):
    # train on older vintages, evaluate on the newest OOT_TEST_FRACTION; random:
    # a stratified random split kept as an ablation.
    if split_mode == "oot":
        _test_year = test_year or C.OOT_TEST_YEAR
        _train_max = _test_year - 1
        train_idx, test_idx = D.vintage_holdout_split(
            df, train_max_year=_train_max, test_year=_test_year)
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        tr_lo, tr_hi = df.loc[train_idx, "issue_date"].min(), df.loc[train_idx, "issue_date"].max()
        te_lo, te_hi = df.loc[test_idx, "issue_date"].min(), df.loc[test_idx, "issue_date"].max()
        print(f"      OOT train: issue_year <= {_train_max}  "
              f"({tr_lo:%Y-%m} .. {tr_hi:%Y-%m}, n={len(X_train):,})")
        print(f"      OOT test : issue_year == {_test_year}  "
              f"({te_lo:%Y-%m} .. {te_hi:%Y-%m}, n={len(X_test):,})")
    elif split_mode == "oot_fraction":
        train_idx, test_idx = D.time_ordered_split(df, test_fraction=C.OOT_TEST_FRACTION)
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        tr_lo, tr_hi = df.loc[train_idx, "issue_date"].min(), df.loc[train_idx, "issue_date"].max()
        te_lo, te_hi = df.loc[test_idx, "issue_date"].min(), df.loc[test_idx, "issue_date"].max()
        print(f"      OOT-fraction train: {tr_lo:%Y-%m} .. {tr_hi:%Y-%m}  (n={len(X_train):,})")
        print(f"      OOT-fraction test : {te_lo:%Y-%m} .. {te_hi:%Y-%m}  (n={len(X_test):,})")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)

    results = {}

    if run_full_zoo:
        print(f"[2/5] Cross-validated model zoo (sample={zoo_sample:,}) ...")
        Xz, yz = X_train, y_train
        if len(X_train) > zoo_sample:
            idx = (pd.Series(range(len(X_train)))
                   .groupby(y_train.to_numpy(), group_keys=False)
                   .apply(lambda s: s.sample(frac=zoo_sample / len(X_train),
                                             random_state=C.RANDOM_STATE)))
            Xz = X_train.iloc[idx.to_numpy()]
            yz = y_train.iloc[idx.to_numpy()]
        zoo = run_zoo(Xz, yz, include_grade=include_grade, cv_folds=3)
        zoo.to_csv(C.REPORTS_DIR / "model_zoo_cv.csv")
    else:
        print("[2/5] Skipping full zoo (run_full_zoo=False).")

    print(f"[3/5] Optuna tuning XGBoost ({n_trials} trials) ...")
    best_params = tune_xgboost(X_train, y_train, n_trials=n_trials,
                               include_grade=include_grade)
    print(f"      Best params: {best_params}")

    print("[4/5] Fitting calibrated XGBoost on the training set ...")
    model = fit_calibrated_xgb(X_train, y_train, params=best_params,
                               include_grade=include_grade)

    print("[5/5] Evaluating on the hold-out test set ...")
    pd_scores = model.predict_proba(X_test)[:, 1]
    metrics = E.credit_metrics(y_test, pd_scores, with_emp=True)
    metrics["split_mode"] = split_mode
    results["Calibrated XGBoost"] = metrics
    for kk, vv in metrics.items():
        print(f"      {kk:14s}: {vv:.4f}" if isinstance(vv, float) else f"      {kk:14s}: {vv}")

    # Per-vintage breakdown (OOT only): AUC stability across issue years on the
    # held-out cohort. A model that degrades sharply on the newest vintage is a
    # red flag for deployment.
    vintage_metrics = {}
    test_idx = X_test.index
    if split_mode in ("oot", "oot_fraction") and "issue_quarter" in df.columns:
        print("      Per-quarter AUC (OOT test):")
        qtr = df.loc[test_idx, "issue_quarter"]
        for q in sorted(qtr.dropna().unique()):
            mask = (qtr == q).to_numpy()
            if mask.sum() < 200 or y_test.to_numpy()[mask].sum() < 10:
                continue
            try:
                vm = E.credit_metrics(y_test.to_numpy()[mask], pd_scores[mask])
                vintage_metrics[str(q)] = {
                    "n": int(mask.sum()), "roc_auc": vm["roc_auc"],
                    "ks": vm["ks"], "default_rate": float(y_test.to_numpy()[mask].mean())}
                print(f"        {q}: n={mask.sum():>6,}  AUC={vm['roc_auc']:.4f}  "
                      f"KS={vm['ks']:.4f}  default_rate={y_test.to_numpy()[mask].mean():.3f}")
            except ValueError:
                continue

    # Persist artifacts - suffix encodes both grade flag and test vintage
    _vintage_tag = f"_{test_year}" if test_year and test_year != C.OOT_TEST_YEAR else ""
    suffix = ("_with_grade" if include_grade else "") + _vintage_tag
    joblib.dump(model, C.MODELS_DIR / f"pd_model{suffix}.pkl")
    (C.MODELS_DIR / f"best_params{suffix}.json").write_text(json.dumps(best_params, indent=2))
    (C.REPORTS_DIR / f"test_metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))
    if vintage_metrics:
        (C.REPORTS_DIR / f"vintage_metrics{suffix}.json").write_text(
            json.dumps(vintage_metrics, indent=2))

    # Persist the held-out test set (features + economics + target + PD + vintage)
    # for the profit-risk stage, so it reuses the EXACT same hold-out.
    econ = df.loc[test_idx, [c for c in C.ECONOMICS_COLUMNS if c in df.columns]]
    out = X_test.copy()
    out[C.TARGET] = y_test
    out["pd_score"] = pd_scores
    if "issue_year" in df.columns:
        out["issue_year"] = df.loc[test_idx, "issue_year"]
        out["issue_date"] = df.loc[test_idx, "issue_date"]
    # last_pymnt_date drives months-on-book for the ANR target; int_rate/sub_grade
    # are the LC grade benchmark for the portfolio backtest (NOT model features).
    if "last_pymnt_date" in df.columns:
        out["last_pymnt_date"] = df.loc[test_idx, "last_pymnt_date"]
    for gcol in ("int_rate", "sub_grade", "grade"):
        if gcol in df.columns and gcol not in out.columns:
            out[gcol] = df.loc[test_idx, gcol]
    out = out.join(econ)

    # Per-loan LGD model (two-stage): trained on training-partition defaults,
    # predicted for every test loan so the profit layer uses EL = PD x LGD x EAD
    # with a loan-specific LGD instead of a single portfolio constant.
    print("[LGD] Training two-stage LGD model on training defaults ...")
    from . import lgd as L
    df_train_full = df.loc[X_train.index]
    lgd_model = L.fit_lgd_model(df_train_full, include_grade=include_grade)
    lgd_eval = L.evaluate_lgd(lgd_model, df.loc[test_idx], include_grade=include_grade)
    out["lgd_pred"] = lgd_model.predict(X_test)
    joblib.dump(lgd_model, C.MODELS_DIR / f"lgd_model{suffix}.pkl")
    if lgd_eval:
        (C.REPORTS_DIR / f"lgd_metrics{suffix}.json").write_text(json.dumps(lgd_eval, indent=2))
        print(f"      LGD MAE={lgd_eval['mae']:.4f} vs constant {lgd_eval['portfolio_constant_mae']:.4f} "
              f"(mean realized {lgd_eval['mean_realized_lgd']:.3f})")

    # Profit-scoring head: architecture-aware return (ANR) regressor.
    # Pipeline: matured-only training → zoo (optional) → Optuna tune winner → fit.
    if with_return:
        print("[RET] Training return (ANR) model ...")
        from . import returns as R
        matured_train = R.is_matured(df_train_full)
        df_train_matured = df_train_full[matured_train]
        X_train_mat = X_train.loc[df_train_matured.index]
        anr_train = R.realized_anr(df_train_matured)
        print(f"      Matured training loans: {len(X_train_mat):,} / {len(X_train):,} "
              f"({matured_train.mean():.1%}) - training on these only")

        # Step 1 - architecture selection
        if return_arch:
            best_ret_arch = return_arch
            print(f"      Architecture forced: {_RETURN_DISPLAY.get(best_ret_arch, best_ret_arch)}")
        elif run_return_zoo:
            print("      Running return-model zoo (Spearman rank correlation) ...")
            zoo_ret = return_model_zoo(X_train_mat, anr_train, include_grade=include_grade)
            zoo_ret.to_csv(C.REPORTS_DIR / f"return_zoo_cv{suffix}.csv")
            best_ret_arch = str(zoo_ret.index[0])
            print(f"      Zoo winner: {_RETURN_DISPLAY.get(best_ret_arch, best_ret_arch)} "
                  f"(Spearman={zoo_ret.iloc[0]['spearman_mean']:.4f})")
        else:
            best_ret_arch = "lightgbm"

        # Step 2 - Optuna tuning (skip for linear models; no meaningful search space)
        if return_trials > 0 and best_ret_arch not in ("ridge", "elasticnet"):
            print(f"      Optuna tuning {_RETURN_DISPLAY.get(best_ret_arch, best_ret_arch)} "
                  f"({return_trials} trials) ...")
            ret_params = tune_return_model(X_train_mat, anr_train, arch=best_ret_arch,
                                           n_trials=return_trials, include_grade=include_grade)
            print(f"      Best params: {ret_params}")
        else:
            ret_params = None

        # Step 3 - final fit
        return_model = fit_return_model(X_train_mat, anr_train,
                                        include_grade=include_grade,
                                        arch=best_ret_arch, params=ret_params)
        out["anr_pred"] = return_model.predict(X_test)
        joblib.dump(return_model, C.MODELS_DIR / f"return_model{suffix}.pkl")
        meta = {"arch": best_ret_arch, **(ret_params or {})}
        (C.MODELS_DIR / f"return_model_params{suffix}.json").write_text(
            json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in meta.items()}, indent=2))
        anr_test_realized = R.realized_anr(df.loc[test_idx])
        print(f"      Return model: mean predicted ANR={out['anr_pred'].mean():+.4f} "
              f"vs realized {anr_test_realized.mean():+.4f}")

    out.to_parquet(C.PROCESSED_DIR / f"test_with_pd{suffix}.parquet")
    print(f"\nSaved model + metrics + scored test set to {C.MODELS_DIR} / {C.PROCESSED_DIR}")
    return model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Lending Club PD model.")
    ap.add_argument("--zoo-sample", type=int, default=None)
    ap.add_argument("--train-sample", type=int, default=None)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--include-grade", action="store_true",
                    help="Include LC's own grade/sub_grade/int_rate signals.")
    ap.add_argument("--no-zoo", action="store_true", help="Skip the slow full zoo.")
    ap.add_argument("--no-return", action="store_true",
                    help="Skip the profit-scoring (ANR) return-model head.")
    ap.add_argument("--return-trials", type=int, default=20,
                    help="Optuna trials for the ANR return model (0 = default params, no tuning).")
    ap.add_argument("--return-zoo", action="store_true",
                    help="Run multi-architecture zoo for the return model before tuning.")
    ap.add_argument("--return-arch", default=None, choices=list(RETURN_ARCHS),
                    help="Force a specific return-model architecture (skips the zoo).")
    ap.add_argument("--split-mode", choices=["oot", "oot_fraction", "random"], default=None,
                    help="Validation split: 'oot' (vintage holdout, default), "
                         "'oot_fraction', or 'random'.")
    ap.add_argument("--test-year", type=int, default=None,
                    help="OOT test vintage year (default: 2015). Train set = all years before this.")
    args = ap.parse_args()
    main(zoo_sample=args.zoo_sample, train_sample=args.train_sample,
         include_grade=args.include_grade, n_trials=args.trials,
         run_full_zoo=not args.no_zoo, split_mode=args.split_mode,
         with_return=not args.no_return, return_trials=args.return_trials,
         run_return_zoo=args.return_zoo, return_arch=args.return_arch,
         test_year=args.test_year)
