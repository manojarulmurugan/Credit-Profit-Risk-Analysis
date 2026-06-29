# EDA Report - Lending Club Credit Risk

**Source:** `rebuild/notebooks/01_EDA_Preprocessing.ipynb`  
**Sample:** 60,000-row stratified sample drawn from 1.31M resolved loans  
**Date:** June 2026

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total rows in file | 2,260,668 |
| Resolved loans (used for modeling) | 1,306,387 |
| Fully Paid (Good, target = 0) | 1,043,940 (79.9%) |
| Charged Off / Default (Bad, target = 1) | 262,447 (20.1%) |
| Portfolio default rate | **20.1%** |
| Features in final model matrix | 22 |

**Why are rows dropped?** Only "Fully Paid" and "Charged Off / Default" loans have known outcomes. 919K "Current", 21K "Late", and 9K "In Grace Period" loans are excluded because their final state is unknown - including them would introduce noise into the target variable.

**Why ROC-AUC over accuracy:** Credit default is a minority-class problem (~20% bad). A model that predicts every applicant as "good" already scores ~80% accuracy, making accuracy uninformative for this task. ROC-AUC, KS statistic, and Gini are used as primary metrics.

---

## 2. Class Balance

The dataset has a ~4:1 imbalance (80% good, 20% bad). This is addressed with SMOTE applied **inside training folds only** - never on the full dataset before splitting - ensuring the resampling step cannot influence test set evaluation.

---

## 3. Per-Feature EDA

### 3.1 Loan Grade (Strongest Predictor)

| Grade | Default Rate |
|---|---|
| A | ~5.3% |
| B | ~11.1% |
| C | ~18.4% |
| D | ~25.7% |
| E | ~32.1% |
| F | ~38.4% |
| G | ~42.8% |

**Finding:** Grade G defaults at 8× Grade A. This is by far the single strongest predictor.  
**Modeling decision:** The main PD model is built **without** `grade`, `sub_grade`, or `int_rate`. These columns encode Lending Club's own internal risk model. Excluding them ensures the model learns independent signal from borrower fundamentals, making it additive to LC's existing risk pricing rather than a re-derivation of it.

---

### 3.2 Loan Term

| Term | Default Rate |
|---|---|
| 36 months | ~15.2% |
| 60 months | ~27.4% |

**Finding:** 60-month loans default at nearly 2× the rate of 36-month loans.  
**Feature treatment:** `term` is engineered to `term_months` (integer 36 or 60).

---

### 3.3 Home Ownership

| Status | Default Rate |
|---|---|
| OTHER | ~25.4% |
| RENT | ~22.1% |
| OWN | ~18.7% |
| MORTGAGE | ~16.3% |

**Finding:** Renters default more than mortgage holders. Rare categories `ANY` and `NONE` are collapsed into `OTHER` during cleaning.

---

### 3.4 Loan Purpose

| Purpose | Default Rate |
|---|---|
| Small Business | ~27.3% |
| Renewable Energy | ~24.6% |
| Educational | ~22.1% |
| Moving | ~20.8% |
| Debt Consolidation | ~20.1% |
| Medical | ~17.2% |
| Credit Card | ~15.4% |
| Home Improvement | ~14.9% |
| Major Purchase | ~13.8% |
| Car | ~9.7% |

**Finding:** Small business loans carry the highest per-unit default risk. Car and major purchase loans are the safest by purpose.

---

### 3a. Additional Categoricals

**Verification Status - Counter-intuitive finding:**

| Status | Default Rate |
|---|---|
| Verified | ~22.6% |
| Source Verified | ~20.3% |
| Not Verified | ~17.9% |

"Verified" applicants default *more* than "Not Verified." This reflects a **selection-bias dynamic** - Lending Club verifies income specifically for applicants it flags as higher-risk. The feature is retained in the model, where it captures LC's risk-flagging behavior rather than raw income quality. Worth noting when presenting results to a non-technical audience.

**Employment Length - Weak predictor:**  
Default rates range only from ~21.4% (<1 year) to ~18.1% (10+ years) - a 3pp spread across all categories. Mutual Information will rank this feature near the bottom; it will likely be dropped by SelectKBest in the pipeline.

**Application Type:** Joint applicants default less than individual applicants, likely because two incomes reduce repayment risk.

---

### 3b. Numeric Feature Distributions

KDE plots (default vs non-default population) show degree of separation for each feature:

| Feature | Defaulter skew | Signal strength |
|---|---|---|
| `int_rate` | Higher rates → more defaults | Strong (excluded from model - encodes LC's risk) |
| `dti` | Higher DTI → more defaults | Strong |
| `revol_util` | Higher utilization → more defaults | Moderate-Strong |
| `annual_inc` | Lower income → more defaults | Moderate |
| `loan_amnt` | Larger loans → more defaults | Moderate |
| `installment` | Higher installment → more defaults | Moderate (collinear with loan_amnt) |

---

### 3c. Continuous Feature Binning

Binning exposes non-linear relationships that Pearson correlation misses.

**DTI buckets:**
| DTI Range | Default Rate |
|---|---|
| 0–5 | ~13.2% |
| 5–10 | ~16.7% |
| 10–15 | ~19.4% |
| 15–20 | ~22.3% |
| 20–30 | ~25.8% |
| 30–50 | ~30.1% |

**Revolving utilization:**
| Util Range | Default Rate |
|---|---|
| 0–20% | ~13.9% |
| 20–40% | ~18.6% |
| 40–60% | ~21.8% |
| 60–80% | ~24.7% |
| 80–100% | ~28.3% |
| >100% | ~33.1% |

Both show clean monotonic step-ups - these are strong, well-behaved features for any classifier.

---

### 3d. Pearson Correlation Heatmap

Key findings:
- **`loan_amnt` ↔ `installment`**: |r| ≈ 0.95 - near-perfect collinearity. The `CorrelationThreshold` transformer inside the pipeline drops one automatically at fit-time.
- **`open_acc` ↔ `total_acc`**: |r| ≈ 0.68 - related but not redundant (total includes open). Both kept.
- **Most other pairs**: |r| < 0.3 - the feature set is largely orthogonal, which is good for both linear and tree-based models.

---

### 3e. Cross-Feature Interaction Heatmaps

**Grade × Term:**  
The 60-month penalty is *not* uniform - it amplifies with risk tier.

| Grade | 36-month default rate | 60-month default rate | Δ |
|---|---|---|---|
| A | ~4.1% | ~8.6% | +4.5pp |
| D | ~22.8% | ~30.5% | +7.7pp |
| G | ~38.7% | ~50.6% | +11.9pp |

A Grade-G 60-month loan is essentially a coin flip. Tree-based models capture this interaction naturally.

**Grade × Home Ownership:** Mortgage ownership reduces default risk across all grades, but the reduction is proportionally larger in lower grades.

---

### 3f. Income-Based Segmentation

| Income Quartile | Default Rate |
|---|---|
| Q1 (Low) | ~24.8% |
| Q2 | ~21.3% |
| Q3 | ~18.7% |
| Q4 (High) | ~15.1% |

Clear gradient - Q1 borrowers default 10pp above Q4. Within grade tiers, income still independently modulates risk, justifying `annual_inc` as a standalone model feature.

---

### 3g. Geographic EDA (U.S. State)

**Highest default-rate states:** NV (~25.3%), AK (~24.8%), MS (~24.1%), FL (~23.7%), NM (~23.4%)  
**Lowest default-rate states:** ME (~14.2%), VT (~14.8%), ND (~15.1%), SD (~15.4%), NH (~15.8%)

~10pp spread between best and worst states. Geography proxies local economic conditions, unemployment cycles, and state-level bankruptcy law differences. `addr_state` is included as a one-hot encoded categorical with `min_frequency=0.01` to handle rare states.

---

### 3h. Classic Credit Risk Signals

**Delinquencies (past 2 years):**
| Count | Default Rate |
|---|---|
| 0 | ~18.8% |
| 1 | ~26.4% |
| 2 | ~30.7% |
| 3+ | ~35.2% |

A single recent delinquency raises default probability by ~8pp. Currently binarized as `pub_rec_flag` (any vs. none) in the feature set.

**Credit inquiries (last 6 months):**
| Count | Default Rate |
|---|---|
| 0 | ~16.3% |
| 1 | ~20.4% |
| 2 | ~23.9% |
| 3 | ~27.1% |
| 4 | ~29.8% |
| 5+ | ~33.4% |

More inquiries = more financial stress. 5+ inquiries → 33% default, 2× baseline.

---

### 3i. Bivariate Scatter - int_rate vs DTI

A 2D view of the two strongest continuous signals. Defaulted loans cluster toward the **top-right** (high interest rate AND high DTI). The bottom-left quadrant (low rate, low DTI) is the safest lending zone. The overlap in the middle explains why these features alone achieve moderate but not perfect discrimination.

---

### 3j. Installment-to-Income Ratio

`installment / (annual_inc / 12)` = monthly payment as a share of monthly income.

| Burden Bucket | Default Rate |
|---|---|
| <2% | ~14.1% |
| 2–4% | ~18.3% |
| 4–6% | ~21.7% |
| 6–8% | ~25.2% |
| 8–10% | ~28.6% |
| >10% | ~33.4% |

**Key finding:** This engineered ratio is a stronger monotonic predictor than either `installment` or `annual_inc` alone. It directly captures **affordability burden** - whether the borrower can realistically service the loan from current income.  
**Recommendation:** Add this as an explicit feature in `features.py` for the next model iteration.

---

### 3k. Credit Age Analysis

`credit_age = 2018 − earliest_cr_line_year`

| Credit Age | Default Rate |
|---|---|
| 0–5 years | ~27.3% |
| 5–10 years | ~22.8% |
| 10–15 years | ~19.4% |
| 15–20 years | ~17.1% |
| 20–25 years | ~15.6% |
| 25+ years | ~13.2% |

**Finding:** Older credit histories default at half the rate of new ones. A borrower with 25+ years of history is a fundamentally different risk than one with <5 years.  
**Current treatment:** `earliest_cr_line_year` is already in the feature set. A direct `credit_age` column would be a stronger, more interpretable signal and is recommended for the next iteration.

---

### 3l. Defaulter Composition by Purpose and Grade

"Who defaults the most?" vs "who drives the *volume* of defaults?" are different questions.

| Purpose | Share of all loans | Share of all defaults | Over/under-represented? |
|---|---|---|---|
| Debt Consolidation | 48.3% | 49.1% | Proportionate |
| Credit Card | 22.6% | 17.4% | Under-represented |
| Home Improvement | 7.1% | 5.3% | Under-represented |
| Small Business | 3.1% | 4.2% | **Over-represented** |
| Major Purchase | 4.6% | 3.2% | Under-represented |
| Other | 14.3% | 20.8% | **Over-represented** |

**Finding:** Debt consolidation dominates default *volume* (~49%) simply because it dominates loan volume. Its *rate* is near average. Small business and "Other" are over-represented in defaults relative to their share of loans - that is where elevated rate risk concentrates.

---

## 4. Modeling Implications Summary

| EDA Finding | Decision | Impact |
|---|---|---|
| Grade near-perfectly predictive | Excluded from model - builds independent borrower signals | High |
| Verification status reversal | Kept; documents it captures LC's risk-flagging behavior | Medium |
| loan_amnt ↔ installment collinear | CorrelationThreshold in Pipeline auto-drops one | Medium |
| Employment length near-flat | MI ranking will drop it; keep for now | Low |
| Installment-to-income ratio monotonic | **Add as engineered feature in next iteration** | High |
| Credit age inverse relationship | Add direct `credit_age` column in next iteration | Medium |
| Delinquency / inquiry non-linear | Currently binarized; capped-count version worth testing | Medium |
| ~10pp geographic spread | addr_state one-hot encoded (min_frequency=0.01) | Low-Medium |

---

## 5. Leakage Control

All analysis above uses only **origination-time columns**. The following economics columns are quarantined for the profit layer in `03_Profit_Risk.ipynb` and are never seen by the PD model:

- `funded_amnt`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`

The leakage assertion in `tests/test_no_leakage.py` verifies this programmatically on every training run.
