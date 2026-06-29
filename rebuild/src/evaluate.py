"""Credit-risk evaluation metrics.

Primary metrics: ROC-AUC, KS statistic, Gini coefficient, PR-AUC, and Brier
score (calibration). Accuracy is included as a secondary metric - on an
imbalanced problem (~20% default rate) it is uninformative as a standalone
number. ``text_report`` calls ``classification_report(y_true, y_pred)`` in the
correct argument order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config as C


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: max separation between good/bad cumulative distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini_coefficient(auc: float) -> float:
    """Gini = 2*AUC - 1 (standard credit-scoring discrimination measure)."""
    return float(2 * auc - 1)


def _roc_convex_hull(y_true, y_score):
    """ROC convex hull above the diagonal + class priors, for the EMP measure.

    Returns (pi0, pi1, F0, F1) where pi0/pi1 are the priors of the bad/good class,
    F0 is the hull's cumulative hit rate among bads (TPR) and F1 the cumulative
    false-alarm rate among goods (FPR), both sorted ascending. Mirrors the
    ``.empRocInfo`` helper of the EMP package (class label 1 = defaulter = "bad").
    """
    from scipy.spatial import ConvexHull

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_bad = int((y_true == 1).sum())
    total = len(y_true)
    pi0 = n_bad / total          # prior of bad (defaulters)
    pi1 = (total - n_bad) / total  # prior of good

    fpr, tpr, _ = roc_curve(y_true, y_score)
    pts = np.c_[fpr, tpr]
    hull = ConvexHull(pts)
    verts = [k for k in hull.vertices if pts[k, 1] >= pts[k, 0]]  # keep upper hull
    F1 = np.sort(pts[verts, 0])  # FPR (goods wrongly flagged)
    F0 = np.sort(pts[verts, 1])  # TPR (bads correctly flagged)
    return pi0, pi1, F0, F1


def emp_credit_scoring(y_true, y_score, p0: float = None, p1: float = None,
                       roi: float = None) -> tuple[float, float]:
    """Expected Maximum Profit for credit scoring (Verbraken et al., EJOR 2014).

    Faithful port of ``empCreditScoring`` (EMP R package / Banking-Analytics-Lab
    Python port): bimodal LGD with point masses ``p0`` at full recovery and ``p1``
    at total loss (uniform in between), constant ``roi`` per granted good loan.
    Returns ``(emp, emp_fraction)`` - expected profit per applicant and the
    fraction of applicants that would be rejected at the profit-optimal operation.
    """
    p0 = C.EMP_P0 if p0 is None else p0
    p1 = C.EMP_P1 if p1 is None else p1
    roi = C.EMP_ROI if roi is None else roi

    pi0, pi1, F0, F1 = _roc_convex_hull(y_true, y_score)
    if len(F0) < 2:
        return 0.0, 0.0
    alpha = 1.0 - p0 - p1

    lam = np.append(0.0, ((pi1 * roi) / pi0) * (np.diff(F1) / np.diff(F0)))
    lam = np.append(lam[lam < 1], 1.0)
    lam_ii, lam_ie = lam[:-1], lam[1:]
    F0c, F1c = F0[:len(lam_ii)], F1[:len(lam_ii)]

    emp = float(np.sum(alpha * (lam_ie - lam_ii) *
                       (pi0 * F0c * (lam_ie + lam_ii) / 2 - roi * F1c * pi1))
                + (pi0 * F0c[-1] - roi * pi1 * F1c[-1]) * p1)
    frac = float(np.sum(alpha * (lam_ie - lam_ii) * (pi0 * F0c + pi1 * F1c))
                 + p1 * (pi0 * F0c[-1] + pi1 * F1c[-1]))
    return emp, frac


def credit_metrics(y_true, y_score, threshold: float = 0.5,
                   with_emp: bool = False) -> dict:
    """Compute the full headline metric suite from probabilities of default.

    When ``with_emp`` is set, the profit-driven EMP measure and its optimal
    rejection fraction are added (skipped silently if the slice is degenerate).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    out = {
        "roc_auc": float(auc),
        "ks": ks_statistic(y_true, y_score),
        "gini": gini_coefficient(auc),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "recall_bad": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_bad": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_bad": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),  # secondary only
        "threshold": float(threshold),
    }
    if with_emp:
        try:
            emp, frac = emp_credit_scoring(y_true, y_score)
            out["emp"] = emp
            out["emp_fraction"] = frac
        except Exception:
            pass
    return out


def text_report(y_true, y_pred) -> str:
    """classification_report with the CORRECT (y_true, y_pred) argument order."""
    return classification_report(y_true, y_pred, digits=4)


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def metrics_frame(results: dict[str, dict]) -> pd.DataFrame:
    """Turn {model_name: metrics_dict} into a tidy, sorted DataFrame."""
    df = pd.DataFrame(results).T
    if "roc_auc" in df.columns:
        df = df.sort_values("roc_auc", ascending=False)
    return df
