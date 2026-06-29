# Credit Risk + Profit Optimization (Lending Club)

An end-to-end, deployment-minded credit-risk system on the full Lending Club loan book (2.26M loans):

**out-of-time PD → calibration → per-loan LGD → expected loss (`PD × LGD × EAD`) → profit-optimized policy → lifetime PD / IFRS 9 ECL → ECOA fairness reason codes → PSI drift monitoring → FastAPI service**, with SHAP explainability and an interactive Streamlit demo.

> **Repository structure:** The active project lives in [`rebuild/`](rebuild/). The original undergraduate capstone is preserved untouched in [`legacy_capstone/`](legacy_capstone/) — see the [note at the bottom](#legacy-capstone) for context.

---

## Headline Results — Out-of-Time Validation

The model is trained on **vintages through 2014** and tested on **2015 originations** (~373K loans) — the standard next-vintage holdout for a model that scores future applicants. No Lending Club grade or interest-rate signals are used.

| Metric (OOT 2015 test) | Value |
|---|---|
| ROC-AUC | **0.708** |
| KS statistic | **0.301** |
| Gini coefficient | **0.416** |
| PR-AUC | **~0.37** |
| Brier score | **~0.16** |

Grade and interest rate are deliberately excluded so the model adds independent signal on top of Lending Club's own pricing.

### Profit Optimization (2015 OOT cohort)

| Policy | Total realized profit | Approval rate | Default rate (approved) |
|---|---|---|---|
| Approve everyone | +$191.2M | 100% | 20.2% |
| Profit-optimal (PD ≤ 0.26) | **+$257.9M** | 83.2% | 15.6% |
| **Value added by the model** | **+$66.7M** | | |

Ranking by PD and declining the riskiest ~17% of applicants adds **$66.7M** of realized profit on the 2015 origination book.

### Beyond the Headline Model

| Capability | Result |
|---|---|
| **Per-loan LGD** (two-stage cure + severity) | MAE 0.185 < 0.196 portfolio-constant baseline |
| **Lifetime PD** (discrete-time survival) | 12-month ECL $2.84M vs lifetime ECL $10.16M |
| **IFRS 9 staging** | Stage 1/2/3 allocation + 12-month vs lifetime ECL |
| **Fairness** | ECOA / Reg B adverse-action reason codes (SHAP-based) |
| **Monitoring** | PSI per feature + PD-score PSI across vintages |
| **Serving** | FastAPI `/score` + `/health`, Dockerized; `MODEL_CARD.md` |

---

## Methodology

| Step | Approach |
|---|---|
| Validation | **Out-of-time vintage holdout** — train ≤2014, test 2015 originations |
| Target definition | `loan_status`: Charged Off / Default → 1, Fully Paid → 0; unresolved dropped |
| Feature set | 22 origination-time features; grade/sub_grade/int_rate quarantined |
| Imbalance handling | SMOTE inside training folds only (via `imblearn.Pipeline`) |
| Feature selection | Mutual Information Gain + Pearson correlation drop |
| Model zoo | LR, Decision Tree, Random Forest, SVM (4 kernels), MLP, Naïve Bayes, XGBoost, **LightGBM**, **CatBoost**, stacking |
| Finalist tuning | Optuna (Bayesian) on LR, RF, XGBoost, LightGBM, CatBoost — identical OOT split |
| Calibration | Isotonic regression — calibrated probabilities feed the profit layer |
| LGD | Two-stage (cure classifier + severity regressor), per-loan |
| Lifetime PD | Discrete-time logistic hazard → survival → IFRS 9 staging / ECL |
| Explainability | SHAP global + per-applicant + ECOA adverse-action reason codes |
| Profit layer | `EL = PD × LGD × EAD` from real economics; profit-curve threshold optimization |
| Scorecard view | Weight of Evidence / Information Value (Basel / IFRS 9 framing) |
| Monitoring | Population Stability Index (PSI) |
| Serving | FastAPI + Docker |

---

## Project Layout

```
rebuild/
  src/
    config.py       # paths, feature allowlist, OOT/date/survival constants
    data.py         # load + target + date parsing + vintage_holdout_split (OOT)
    features.py     # cleaning, engineering, preprocessor, corr-drop + MI selection
    resampling.py   # imblearn pipeline (SMOTE inside training folds)
    train.py        # zoo (+LightGBM/CatBoost), Optuna finalists, calibration, OOT eval, LGD
    evaluate.py     # ROC-AUC / KS / Gini / PR-AUC / Brier
    explain.py      # SHAP global + per-applicant
    lgd.py          # two-stage per-loan Loss Given Default model
    survival.py     # discrete-time hazard → lifetime PD + IFRS 9 staging/ECL
    fairness.py     # ECOA adverse-action reason codes
    monitoring.py   # PSI drift (feature + PD score)
    profit.py       # EL = PD × LGD × EAD, profit curve, threshold, WoE/IV
    app.py          # Streamlit decision demo
    api.py          # FastAPI scoring service (/score, /health)
  notebooks/
    01_EDA_Preprocessing.ipynb   04_Survival_IFRS9.ipynb
    02_Modeling.ipynb            05_Fairness.ipynb
    03_Profit_Risk.ipynb         06_Monitoring.ipynb
  reports/
    01_EDA_Report.md … 06_Monitoring_Report.md   (+ regenerated *.json / *.csv)
  tests/
    test_no_leakage.py     # leakage + OOT no-temporal-overlap tests
    test_enhancements.py   # LGD / survival / monitoring / fairness / API tests
  MODEL_CARD.md   Dockerfile   .dockerignore   requirements.txt   Makefile
```

---

## Setup & Reproduction

The dataset is large and not committed to git. Download the full Lending Club accepted-loans file (`loan.csv`, ~2.26M rows × 145 columns) from Kaggle and place it at `Lending Club Dataset/loan.csv`.

```bash
cd rebuild
make install      # create .venv and install pinned dependencies
```

> **macOS note:** XGBoost / LightGBM require OpenMP. If you hit a `libomp.dylib` load error, run `brew install libomp`.

```bash
make test       # leakage / correctness / enhancement tests
make train      # zoo + Optuna + calibration + LGD → models/ + reports/  (OOT split)
make profit     # expected loss + profit-optimal threshold → reports/ + economics.json
make survival   # discrete-time survival → lifetime PD + IFRS 9 ECL
make monitor    # PSI drift report
make app        # Streamlit decision demo
make api        # FastAPI scoring service on :8000
```

---

## Key Design Decisions

**Why out-of-time validation?**  
A credit model is deployed to score *future* loans, so it must be tested on loans issued *after* those it trained on. A random split leaks future information through shared vintages and inflates metrics. OOT is the honest measure — and the per-vintage breakdown proves the model is stable, not just lucky.

**Why exclude grade and interest rate?**  
These columns directly encode Lending Club's internal risk model. Excluding them forces the model to learn independent signal from borrower fundamentals, making it additive to LC's pricing rather than a re-learn of it.

**Why model LGD per loan instead of a constant?**  
Realized loss severity is bimodal (a spike near total loss plus partial recoveries). A two-stage cure + severity model captures that structure and beats the portfolio-average constant (MAE 0.185 vs 0.196), feeding a more accurate `EL = PD × LGD × EAD`.

**Leakage prevention**  
All learned transforms (imputation, scaling, MI selection, SMOTE) live inside an `imblearn.Pipeline` fit only on training data within each fold. Origination vs economics columns are managed via an explicit allowlist in `src/config.py`. The test suite asserts no leakage *and* no temporal overlap in the OOT split on every run.

---

## Legacy Capstone

The original undergraduate capstone project is preserved untouched in [`legacy_capstone/`](legacy_capstone/). This rebuild keeps the same core methodology — median imputation, Mutual Information + Pearson feature selection, the same model-zoo philosophy, and a PD-to-expected-loss profit flow — and extends it with a large real-world dataset, out-of-time validation, probability calibration, a per-loan LGD model, lifetime PD / IFRS 9 ECL via survival analysis, ECOA fairness reason codes, PSI monitoring, SHAP explainability, and a containerized scoring API.
