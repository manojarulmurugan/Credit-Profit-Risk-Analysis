"""Assemble the leakage-free modeling pipeline.

SMOTE lives inside an imblearn Pipeline so it is applied to the training fold
ONLY during ``fit`` and never touches validation or test data.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from . import config as C
from .features import build_feature_pipeline


def build_model_pipeline(estimator,
                         include_grade: bool = False,
                         k: int | str = "all",
                         use_smote: bool = True,
                         corr_threshold: float = 0.95) -> ImbPipeline:
    """Return preprocess -> corr-drop -> MI-select -> [SMOTE] -> estimator.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
    use_smote : if False, rely on the estimator's own class weighting instead.
    """
    feature_pipe = build_feature_pipeline(
        include_grade=include_grade, k=k, corr_threshold=corr_threshold
    )
    steps = list(feature_pipe.steps)  # (name, transformer) tuples
    if use_smote:
        steps.append(("smote", SMOTE(random_state=C.RANDOM_STATE)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)
