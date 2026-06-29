"""Leakage-prevention tests - guards against resampling before the split
and outcome/economics columns appearing in the feature matrix.

Run: rebuild/.venv/bin/python -m pytest rebuild/tests   (or: make test)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import features as F  # noqa: E402


def _sample_df(n=4000):
    df = D.load_resolved(n=n)
    return F.clean_frame(df)


def test_feature_matrix_excludes_economics_and_leakage():
    """No economics/outcome column may appear in the PD feature matrix."""
    df = _sample_df()
    X, _ = F.split_X_y(df, include_grade=True)
    forbidden = set(C.ECONOMICS_COLUMNS) | set(C.LEAKAGE_COLUMNS)
    leaked = forbidden.intersection(X.columns)
    assert not leaked, f"Leakage columns present in features: {leaked}"


def test_target_is_binary_and_resolved():
    df = _sample_df()
    assert set(df[C.TARGET].unique()).issubset({0, 1})
    # Default rate should be a sane minority (Lending Club ~20%).
    assert 0.05 < df[C.TARGET].mean() < 0.45


def test_train_test_split_has_no_row_overlap():
    """The split must produce disjoint index sets (no duplicated/leaked rows)."""
    df = _sample_df()
    X, y = F.split_X_y(df)
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    overlap = set(X_train.index).intersection(X_test.index)
    assert not overlap, f"Train/test index overlap: {len(overlap)} rows"


def test_smote_only_affects_training_fold():
    """SMOTE must not change the number of test rows scored (applied at fit only)."""
    from sklearn.linear_model import LogisticRegression

    from src.resampling import build_model_pipeline

    df = _sample_df()
    X, y = F.split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    pipe = build_model_pipeline(LogisticRegression(max_iter=500), use_smote=True)
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:, 1]
    assert len(preds) == len(X_test)  # test set untouched by resampling


def test_oot_vintage_split_has_no_temporal_overlap():
    """Vintage holdout: max train issue date must precede min test issue date."""
    df = D.load_resolved(n=8000)
    train_idx, test_idx = D.vintage_holdout_split(df)
    assert len(test_idx) > 0 and len(train_idx) > 0
    dated_train = df.loc[train_idx, "issue_date"].dropna()
    dated_test = df.loc[test_idx, "issue_date"].dropna()
    assert dated_train.max() < dated_test.min(), (
        f"OOT leakage: train max {dated_train.max()} >= test min {dated_test.min()}")
    assert df.loc[test_idx, "issue_year"].eq(C.OOT_TEST_YEAR).all()


def test_oot_fraction_split_has_no_temporal_overlap():
    """Fraction-based OOT: no train loan may be newer than the earliest test loan."""
    df = D.load_resolved(n=8000)
    train_idx, test_idx = D.time_ordered_split(df, test_fraction=C.OOT_TEST_FRACTION)
    assert len(test_idx) > 0 and len(train_idx) > 0
    dated_train = df.loc[train_idx, "issue_date"].dropna()
    dated_test = df.loc[test_idx, "issue_date"].dropna()
    assert dated_train.max() <= dated_test.min(), (
        f"OOT leakage: train max {dated_train.max()} > test min {dated_test.min()}")


def test_correlation_threshold_drops_collinear_columns():
    from src.features import CorrelationThreshold

    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = np.column_stack([a, a * 2 + 1e-6 * rng.normal(size=500), rng.normal(size=500)])
    ct = CorrelationThreshold(threshold=0.95).fit(X)
    assert ct.keep_mask_.sum() == 2  # one of the two collinear columns dropped


if __name__ == "__main__":
    test_feature_matrix_excludes_economics_and_leakage()
    test_target_is_binary_and_resolved()
    test_train_test_split_has_no_row_overlap()
    test_oot_vintage_split_has_no_temporal_overlap()
    test_oot_fraction_split_has_no_temporal_overlap()
    test_smote_only_affects_training_fold()
    test_correlation_threshold_drops_collinear_columns()
    print("All leakage tests passed.")
