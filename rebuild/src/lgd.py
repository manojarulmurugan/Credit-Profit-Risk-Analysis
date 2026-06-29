"""Loss Given Default (LGD) modeling.

Replaces the single portfolio-average LGD constant with a per-loan LGD prediction
driven by origination-time features. Realized LGD on defaulted loans is bimodal
(a spike near total loss plus a spread of partial recoveries), so a single mean
is a poor summary. I model it with a two-stage mixture:

    Stage 1 (cure / total-loss classifier): P(near-total loss) for the loan.
    Stage 2 (severity regressor):           expected LGD given a partial recovery.

    predicted LGD = P(total) * 1.0 + (1 - P(total)) * severity

Both stages are trained ONLY on defaulted loans, using ONLY origination-time
features (the same allowlist as the PD model), so no post-origination economics
leak into the LGD predictors. The realized-LGD label itself is derived from
economics columns (that is the supervised target, not a feature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config as C
from . import data as D
from . import profit as P
from .features import build_preprocessor, clean_frame, split_X_y

TOTAL_LOSS_THRESHOLD = 0.95  # LGD at/above this counts as a near-total loss


def realized_lgd_target(df: pd.DataFrame) -> pd.Series:
    """Realized LGD label for defaulted loans, clipped to [0, 1]."""
    return P.realized_lgd(df).clip(0.0, 1.0)


class LGDModel:
    """Two-stage (cure + severity) per-loan LGD estimator."""

    def __init__(self, include_grade: bool = False,
                 total_loss_threshold: float = TOTAL_LOSS_THRESHOLD):
        self.include_grade = include_grade
        self.total_loss_threshold = total_loss_threshold
        self.preproc = None
        self.cure_clf = None
        self.sev_reg = None
        self.portfolio_mean_ = 0.6

    def fit(self, X_defaults: pd.DataFrame, lgd_target: pd.Series) -> "LGDModel":
        rs = C.RANDOM_STATE
        self.portfolio_mean_ = float(lgd_target.mean())

        self.preproc = build_preprocessor(include_grade=self.include_grade)
        Xt = self.preproc.fit_transform(X_defaults)

        # Stage 1: near-total-loss classifier
        is_total = (lgd_target >= self.total_loss_threshold).astype(int)
        if is_total.nunique() < 2:
            self.cure_clf = None  # degenerate: handled in predict
        else:
            self.cure_clf = LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                random_state=rs, verbose=-1).fit(Xt, is_total)

        # Stage 2: severity regressor on partial-loss loans
        partial = (lgd_target < self.total_loss_threshold).to_numpy()
        if partial.sum() >= 50:
            self.sev_reg = LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                random_state=rs, verbose=-1).fit(
                    Xt[partial], lgd_target.to_numpy()[partial])
        else:
            self.sev_reg = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Per-loan expected LGD in [0, 1]."""
        if self.preproc is None:
            return np.full(len(X), self.portfolio_mean_)
        Xt = self.preproc.transform(X)
        p_total = (self.cure_clf.predict_proba(Xt)[:, 1]
                   if self.cure_clf is not None else 0.0)
        if self.sev_reg is not None:
            severity = np.clip(self.sev_reg.predict(Xt), 0.0, self.total_loss_threshold)
        else:
            severity = self.portfolio_mean_
        lgd = p_total * 1.0 + (1.0 - p_total) * severity
        return np.clip(lgd, 0.0, 1.0)


def fit_lgd_model(df_train: pd.DataFrame, include_grade: bool = False) -> LGDModel:
    """Fit an LGDModel on the DEFAULTED loans within ``df_train``."""
    defaults = df_train[df_train[C.TARGET] == 1]
    X_def, _ = split_X_y(defaults, include_grade=include_grade)
    lgd_target = realized_lgd_target(defaults)
    return LGDModel(include_grade=include_grade).fit(X_def, lgd_target)


def evaluate_lgd(model: LGDModel, df_test: pd.DataFrame,
                 include_grade: bool = False) -> dict:
    """MAE / RMSE of predicted vs realized LGD on the test set's defaulted loans."""
    defaults = df_test[df_test[C.TARGET] == 1]
    if len(defaults) == 0:
        return {}
    X_def, _ = split_X_y(defaults, include_grade=include_grade)
    y_true = realized_lgd_target(defaults).to_numpy()
    y_pred = model.predict(X_def)
    # Honest baseline: the portfolio LGD constant a production system would use is
    # the TRAINING mean (learned at fit time), not the test set's own mean.
    constant = np.full_like(y_true, model.portfolio_mean_)
    return {
        "n_defaults": int(len(defaults)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_realized_lgd": float(y_true.mean()),
        "mean_predicted_lgd": float(y_pred.mean()),
        "train_portfolio_lgd": float(model.portfolio_mean_),
        "portfolio_constant_mae": float(mean_absolute_error(y_true, constant)),
    }


if __name__ == "__main__":
    import json

    import joblib

    print("Loading data for standalone LGD training ...")
    df = clean_frame(D.load_resolved(n=C.TRAIN_SAMPLE))
    train_idx, test_idx = D.vintage_holdout_split(df)
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]

    print(f"Training LGD model on {int((df_train[C.TARGET] == 1).sum()):,} defaults ...")
    model = fit_lgd_model(df_train)
    metrics = evaluate_lgd(model, df_test)
    print("LGD model evaluation (OOT defaults):")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v:,.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    improvement = metrics["portfolio_constant_mae"] - metrics["mae"]
    print(f"  MAE improvement vs constant: {improvement:+.4f}")

    joblib.dump(model, C.MODELS_DIR / "lgd_model.pkl")
    (C.REPORTS_DIR / "lgd_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved lgd_model.pkl + lgd_metrics.json")
