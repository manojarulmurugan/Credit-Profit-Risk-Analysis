# Experiments Archive

Scripts in this directory are exploratory experiments that were run during development but were either ruled out or not adopted in the final pipeline. Each script is self-contained and runnable, but none are part of the production pipeline in `src/`.

For the full research narrative including why each experiment was tried and what it taught us, see [`RESEARCH_STORY.md`](../RESEARCH_STORY.md).

---

## `two_stage_model.py` — Two-Stage Return Model

**What:** Trains separate LightGBM models for paid loans and defaulted loans, then combines them at test time using the PD score as a mixing weight:
```
anr_combined = (1 - pd_score) × anr_paid + pd_score × anr_default
```

**Why we tried it:** Decomposing the return into a "what do you earn if paid?" and "what do you lose if it defaults?" sub-problems seemed structurally cleaner than predicting the combined ANR directly.

**Result:** Worse than single-stage at all budget levels.
- Profit scoring @25%: single-stage +3.04% vs two-stage +2.60%
- Profit scoring @50%: single-stage +2.36% vs two-stage +2.10%

**Why it failed:** Without `int_rate` in the feature set (grade-blind design), the paid-only sub-model cannot distinguish return variation among loans that don't default. A 5% ANR loan and a 12% ANR loan look identical from origination features alone — the interest rate is what differentiates them. The single-stage model implicitly learns a combined signal that generalizes better.

**Verdict:** Ruled out. Single-stage LightGBM on the full ANR target is the correct approach.

---

## `oof_pdscore_feature.py` — OOF PD Score as ANR Feature

**What:** Generates out-of-fold PD predictions via 5-fold cross-validation on the training set, then adds `pd_score_oof` as an extra feature when training the ANR model. Valid stacked generalization (Wolpert 1992) — no leakage.

**Why we tried it:** The PD model captures default risk from origination features. Perhaps giving the ANR model access to a distilled default-risk signal (the OOF pd_score) would let it learn the return/risk tradeoff more directly.

**Result:** Marginal. +5 bp at the 25% budget only, flat at all other budgets.

| Budget | Baseline | With OOF pd_score |
|---|---|---|
| @25% | +3.04% | +3.09% |
| @50% | +2.36% | +2.36% |

**Why it barely helped:** The ANR model already learns default risk implicitly from the same origination features the PD model uses. The OOF pd_score is a smoother version of information the ANR model already approximates. At the portfolio level, the marginal lift in ranking quality is negligible.

**Verdict:** Not adopted. The extra complexity (5-fold CV at training time, separate OOF predictions for test set) is not justified by +5 bp.

---

## `refit_lightgbm_return.py` — Targeted ANR Model Refit

**What:** A targeted script that re-fits only the ANR return model (skips the expensive PD model re-training). Loads the full 1.3M row dataset, reconstructs the train/test split, filters to matured training loans, runs Optuna tuning (20 trials), and updates `test_with_pd.parquet` with new `anr_pred` scores.

**Why it exists:** During iterative development, we often needed to re-tune the ANR model without re-running the 45-minute PD training step. This script decouples the two.

**Best params found:** `n_estimators=400, num_leaves=53, max_depth=9, learning_rate=0.029`

**Status:** Build artifact. Superseded by `src/train.py --return-arch lightgbm --return-trials 20` which handles the same refit as part of the full pipeline.

---

## `overnight_cross_vintage_runner.py` — Cross-Vintage Orchestration Script

**What:** Runs all 6 cross-vintage training jobs (3 vintages × 2 variants) sequentially overnight, with logging, error handling, and final report compilation. Produces `reports/overnight_log.txt` and `reports/cross_vintage_results.txt`.

**Why it exists:** Running 6 full training jobs (each 20–32 minutes) manually is error-prone. This script chains them with skip-on-failure logic, logging, and a final summary report.

**Status:** Build artifact. All cross-vintage results are already committed to `reports/`. Re-run only if you need to regenerate models from scratch. Individual runs can also be done via the CLI commands in `README.md`.

---

## Return Model Zoo Results (for reference)

The full zoo was run to select the best architecture for the ANR model. Results (5-fold CV on training set):

| Architecture | Train Spearman (mean ± std) | RMSE |
|---|---|---|
| Random Forest | +0.0148 ± 0.0025 | 0.2121 |
| XGBoost | −0.0308 ± 0.0005 | 0.2091 |
| **LightGBM (Huber)** | −0.0321 ± 0.0018 | **0.2091** |
| CatBoost | −0.0478 ± 0.0005 | **0.2089** |
| ElasticNet | −0.0654 ± 0.0043 | 0.2103 |
| Ridge | −0.1083 ± 0.0038 | 0.2097 |

**Key finding:** Training Spearman ≈ 0 for all architectures — ANR is hard to predict at the loan level due to idiosyncratic variance. Random Forest won by training Spearman but *lost* in OOT portfolio performance. LightGBM (Huber loss) was selected because Huber loss is robust to ANR outliers and LightGBM generalized best to the held-out vintage.

**Lesson:** For portfolio selection, training Spearman is an unreliable model-selection criterion. Evaluate on OOT portfolio ANR, not on point-prediction rank correlation.
