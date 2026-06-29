# Model Card: Lending Club Credit Risk + Profit System

A model card summarizing the intended use, data, performance, fairness posture, and limitations of the models in this project, following the spirit of Mitchell et al. (2019) "Model Cards for Model Reporting."

---

## Model Overview

| | |
|---|---|
| **System** | Credit-risk decision system: PD model + LGD model + survival/IFRS 9 layer + profit optimization |
| **Headline model** | Isotonically-calibrated XGBoost classifier (probability of default) |
| **Supporting models** | Two-stage LightGBM LGD model; discrete-time logistic hazard (lifetime PD) |
| **Validation** | Out-of-time vintage holdout (train ≤2014, test 2015) |
| **Dataset** | Lending Club accepted loans (~1.3M resolved loans of 2.26M total) |
| **Version** | 1.0 |

---

## Intended Use

- **Primary use:** estimate probability of default and expected loss for consumer installment loans, and convert those into a profit-optimized approval policy.
- **Intended users:** credit risk analysts, model validators, and portfolio managers (illustrative / portfolio-demonstration context).
- **Out of scope:** real-money lending decisions without independent validation; populations materially different from US Lending Club borrowers; any use without fair-lending review on data that includes protected attributes.

---

## Data

- **Source:** Lending Club public accepted-loans file (`loan.csv`).
- **Target:** `loan_status` -> 1 for Charged Off / Default, 0 for Fully Paid; unresolved loans (Current, Late, Grace) dropped.
- **Features:** 22 origination-time features via an explicit allowlist (`src/config.py`). Lending Club's own `grade` / `sub_grade` / `int_rate` are quarantined from the headline model so it adds independent signal.
- **Economics columns** (`funded_amnt`, `total_pymnt`, recoveries, etc.) are used only by the LGD/profit layers, never as PD features.

---

## Performance (Out-of-Time Test)

PD model (no grade signals), calibrated XGBoost. Exact figures are regenerated into `reports/test_metrics.json` and `reports/vintage_metrics.json` on each run.

| Metric | Value (approx.) |
|---|---|
| ROC-AUC | ~0.71 |
| KS | ~0.30 |
| Gini | ~0.42 |
| Brier | ~0.16 |

A per-quarter AUC breakdown on the 2015 test vintage is reported to surface any within-year degradation.

**Finalist comparison:** Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost are all Optuna-tuned and calibrated under identical conditions; the winner is reported in `reports/finalist_comparison.json`.

**LGD model:** two-stage (cure classifier + severity regressor); per-loan LGD vs the portfolio-constant baseline reported in `reports/lgd_metrics.json`.

**Lifetime PD / IFRS 9:** 12-month vs lifetime PD and ECL, with Stage 1/2/3 allocation, in `reports/survival_summary.json`.

---

## Profit Layer

Expected Loss `EL = PD x LGD x EAD` (per-loan LGD), with an approval-threshold sweep that maximizes realized portfolio profit. Results in `reports/profit_curve.csv` and `models/economics.json`.

---

## Fairness

- **Adverse-action reason codes:** the system produces specific, ranked, model-faithful denial reasons (SHAP-based), satisfying ECOA / Regulation B and CFPB Circular 2022-03 on explainability. See `src/fairness.py` and `reports/05_Fairness_Report.md`.
- **Disparate-impact testing:** NOT performed. The public dataset contains no protected-class attributes (race, sex, age), so a meaningful disparate-impact analysis is not possible and is deliberately not fabricated. In production this would run on internal data with demographic information or proxies under a documented compliance-management system.
- **Geographic feature:** `addr_state` is included as a predictor; in a regulated deployment it would require fair-lending review as a potential proxy.

---

## Monitoring

Population Stability Index (PSI) per feature and on the PD score across vintages (`src/monitoring.py`, `reports/monitoring_psi.json`), with standard 0.10 / 0.25 action bands. PSI is the same divergence as the scorecard's Information Value.

---

## Limitations

- **Accepted-loans only:** no reject inference; the model is trained on funded loans, which carries selection bias relative to the full applicant pool.
- **Default timing** in the survival model is approximated from `last_pymnt_d`.
- **SICR** is a single-snapshot proxy, not a true origination-vs-current re-measurement.
- **No macroeconomic overlay:** point-in-time model; forward-looking macro scenarios are not yet incorporated.

---

## Ethical Considerations

Credit decisions materially affect people's lives. This system is built with leakage controls, honest out-of-time validation, calibrated probabilities, explainable denials, and drift monitoring — but it must not be deployed for real lending without protected-attribute fairness testing, independent model validation, and ongoing governance.
