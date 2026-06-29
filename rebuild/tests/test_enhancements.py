"""Unit tests for the enhancement modules (LGD, survival, monitoring, fairness, API).

These are logic/shape tests that do not require a fully trained headline model.

Run: rebuild/.venv/bin/python tests/test_enhancements.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import monitoring as M  # noqa: E402
from src import survival as S  # noqa: E402
from src import lgd as L  # noqa: E402
from src import returns as RET  # noqa: E402
from src import fairness as FA  # noqa: E402
from src.features import clean_frame, split_X_y  # noqa: E402


def _df(n=6000):
    return clean_frame(D.load_resolved(n=n))


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert M.psi(x, x) < 1e-6
    assert M.psi_band(0.05) == "stable"
    assert M.psi_band(0.15) == "moderate shift"
    assert M.psi_band(0.40) == "significant shift"


def test_psi_detects_a_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(1.5, 1, 5000)  # shifted population
    assert M.psi(a, b) > 0.25


def test_survival_person_period_event_rate_is_low():
    """Monthly hazard event rate must be small (rare per-period default)."""
    df = _df()
    pp = S.build_person_period(df.loc[df.index])
    assert "event" in pp.columns
    assert 0 < pp["event"].mean() < 0.05
    # one event per defaulted loan at most
    assert pp["event"].sum() <= (df[C.TARGET] == 1).sum()


def test_survival_cumulative_pd_monotonic_and_bounded():
    df = _df()
    model = S.fit_hazard_model(S.build_person_period(df))
    curves = S.survival_curves(model, df.head(300).reset_index(drop=True))
    assert curves["cum_pd"].between(0, 1).all()
    # cumulative PD is non-decreasing within each loan
    diffs = curves.groupby("_loan")["cum_pd"].apply(lambda s: (s.diff().dropna() >= -1e-9).all())
    assert diffs.all()


def test_lgd_target_in_unit_interval():
    df = _df()
    defaults = df[df[C.TARGET] == 1]
    lgd = L.realized_lgd_target(defaults)
    assert lgd.between(0, 1).all()


def test_lgd_model_predicts_unit_interval():
    df = _df()
    train_idx, test_idx = D.time_ordered_split(df)
    model = L.fit_lgd_model(df.loc[train_idx])
    from src.features import split_X_y
    X_test, _ = split_X_y(df.loc[test_idx])
    preds = model.predict(X_test)
    assert ((preds >= 0) & (preds <= 1)).all()


def test_anr_target_finite_and_reasonable():
    """Realized ANR is finite, within the configured band, and defaulted loans
    earn a lower mean return than fully-paid loans (the target has signal)."""
    df = _df()
    anr = RET.realized_anr(df)
    assert np.isfinite(anr).all()
    assert anr.between(C.ANR_CLIP[0], C.ANR_CLIP[1]).all()
    assert anr[df[C.TARGET] == 1].mean() < anr[df[C.TARGET] == 0].mean()


def test_emp_in_valid_range():
    """EMP is non-negative and its optimal rejection fraction lies in [0, 1]."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=4000)
    # scores correlated with the label so the convex hull is non-trivial
    score = np.clip(0.5 * y + rng.normal(0, 0.3, size=4000), 0, 1)
    emp, frac = E.emp_credit_scoring(y, score)
    assert emp >= 0
    assert 0.0 <= frac <= 1.0


def test_return_model_predicts_finite():
    """The profit-scoring (ANR) regressor produces finite predictions on a held
    out slice and never sees economics columns as features."""
    from src.train import fit_return_model
    df = _df()
    train_idx, test_idx = D.time_ordered_split(df)
    X_train, _ = split_X_y(df.loc[train_idx])
    X_test, _ = split_X_y(df.loc[test_idx])
    # features must not contain any economics column
    assert not (set(C.ECONOMICS_COLUMNS) & set(X_train.columns))
    model = fit_return_model(X_train, RET.realized_anr(df.loc[train_idx]))
    preds = model.predict(X_test)
    assert np.isfinite(preds).all()


def test_fairness_base_feature_mapping():
    assert FA._base_feature("num__dti") == "dti"
    assert FA._base_feature("cat__purpose_small_business") == "purpose"
    assert FA._base_feature("cat__home_ownership_RENT") == "home_ownership"
    assert FA._base_feature("num__term_months") == "term_months"


def test_api_imports_and_health():
    from fastapi.testclient import TestClient
    from src.api import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


if __name__ == "__main__":
    test_psi_zero_for_identical_distributions()
    test_psi_detects_a_shift()
    test_survival_person_period_event_rate_is_low()
    test_survival_cumulative_pd_monotonic_and_bounded()
    test_lgd_target_in_unit_interval()
    test_lgd_model_predicts_unit_interval()
    test_anr_target_finite_and_reasonable()
    test_emp_in_valid_range()
    test_return_model_predicts_finite()
    test_fairness_base_feature_mapping()
    test_api_imports_and_health()
    print("All enhancement tests passed.")
