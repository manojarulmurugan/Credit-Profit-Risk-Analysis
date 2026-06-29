"""Feature engineering, cleaning, and the leakage-free preprocessing pipeline.

Preprocessing: median imputation, StandardScaler, Mutual Information feature
selection, Pearson correlation drop. Every fitted transform lives inside a
Pipeline so it is fit on the training fold only. Deterministic cleaning
(type coercions, binarization, rare-category collapsing) runs before the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


# --------------------------------------------------------------------------- #
# Cleaning / engineering (row-wise, no fitting -> safe before the split)
# --------------------------------------------------------------------------- #
_EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, non-fitted cleaning + feature engineering.

    These operations do not learn parameters from the data, so they are safe to
    apply before the train/test split. Anything that *learns* (imputation values,
    scaling stats, selection) happens later inside the modeling Pipeline.
    """
    df = df.copy()

    # term: " 36 months" -> 36
    if "term" in df:
        df["term_months"] = (
            df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        )

    # emp_length -> ordinal numeric
    if "emp_length" in df:
        df["emp_length_num"] = df["emp_length"].map(_EMP_LENGTH_MAP)

    # earliest_cr_line -> year
    if "earliest_cr_line" in df:
        df["earliest_cr_line_year"] = (
            pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce").dt.year
        )

    # Binarize derogatory counts (mirrors original + reference convention)
    if "pub_rec" in df:
        df["pub_rec_flag"] = (df["pub_rec"].fillna(0) > 0).astype(int)
    if "mort_acc" in df:
        df["mort_acc_flag"] = (df["mort_acc"].fillna(0) > 0).astype(int)
    if "pub_rec_bankruptcies" in df:
        df["pub_rec_bankruptcies_flag"] = (
            df["pub_rec_bankruptcies"].fillna(0) > 0
        ).astype(int)

    # home_ownership: collapse rare ANY/NONE into OTHER
    if "home_ownership" in df:
        df["home_ownership"] = df["home_ownership"].replace(
            {"ANY": "OTHER", "NONE": "OTHER"}
        )

    return df


def split_X_y(df: pd.DataFrame, include_grade: bool = False):
    """Return (X, y) using only the allowlisted feature columns."""
    numeric, categorical = C.feature_columns(include_grade=include_grade)
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df[C.TARGET].astype(int).copy()
    return X, y


def get_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Return the economics columns (for the profit layer) aligned to df.index."""
    cols = [c for c in C.ECONOMICS_COLUMNS if c in df.columns]
    return df[cols].copy()


# --------------------------------------------------------------------------- #
# Leakage-free transformers (fitted inside the Pipeline)
# --------------------------------------------------------------------------- #
class CorrelationThreshold(BaseEstimator, TransformerMixin):
    """Drop one of each pair of features with |Pearson r| above ``threshold``.

    Pearson correlation drop step, fit on the training fold
    only so no test information leaks in.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr)
        drop = set()
        for i in range(n_features):
            for j in range(i):
                if abs(corr[i, j]) > self.threshold and j not in drop:
                    drop.add(i)
        self.keep_mask_ = np.array([i not in drop for i in range(n_features)])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_mask_]


def build_preprocessor(include_grade: bool = False) -> ColumnTransformer:
    """ColumnTransformer: median-impute + scale numeric; impute + one-hot categorical."""
    numeric, categorical = C.feature_columns(include_grade=include_grade)

    numeric_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")


def build_feature_pipeline(include_grade: bool = False,
                           k: int | str = "all",
                           corr_threshold: float = 0.95) -> SkPipeline:
    """Preprocess -> Pearson correlation drop -> Mutual Information top-k select.

    The full leakage-free feature-selection stage, returned as a plain sklearn
    Pipeline so it can be slotted in front of any estimator (and any resampler)
    inside the modeling pipeline.

    When ``k == "all"`` the (expensive) Mutual Information step is skipped, since
    it would only recompute scores without dropping anything. Pass an integer
    ``k`` to actually perform Mutual Information top-k selection.
    """
    steps = [
        ("preprocess", build_preprocessor(include_grade=include_grade)),
        ("corr_drop", CorrelationThreshold(threshold=corr_threshold)),
    ]
    if k != "all":
        steps.append(("mi_select", SelectKBest(score_func=mutual_info_classif, k=k)))
    return SkPipeline(steps)
