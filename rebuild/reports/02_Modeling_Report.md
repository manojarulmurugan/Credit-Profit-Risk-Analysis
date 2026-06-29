# Credit Risk Modeling Report
## Lending Club Probability of Default — End-to-End Process

---

## Executive Summary

We built a **Probability of Default (PD) model** on 1.3 million resolved Lending Club loans to determine which loan applications to approve and at what threshold to maximize portfolio profit. The model is validated **out-of-time** — trained on vintages through **2014**, tested on **2015 originations** — the standard next-vintage holdout for a model that scores future applicants.

**The headline result (out-of-time test, 85,452 scored 2015 loans from the 300K development sample):**

| Metric | Value |
|---|---|
| Model | Optuna-tuned, isotonic-calibrated XGBoost |
| Test ROC-AUC | **0.704** |
| KS Statistic | **0.297** |
| Gini Coefficient | **0.408** |
| PR-AUC | **0.371** |
| Brier | **0.148** |
| Optimal approval threshold | **PD ≤ 0.26** |
| Value added vs approve-all (profit) | **+$13.4M** (see Profit-Risk report) |

> The gradient-boosted finalists (CatBoost, XGBoost, LightGBM) finish in a statistical tie (§5); XGBoost is chosen for its mature SHAP TreeExplainer support (needed for exact adverse-action reason codes) and ubiquity in credit risk.

---

## 1. The Problem

### 1.1 What We're Predicting

Given a loan application at the moment of origination — before any payments are made — predict whether this loan will end in **Charge-Off (default)** or **Full Repayment**.

This is a binary classification problem:
- **Target = 1**: Charged Off / Defaulted
- **Target = 0**: Fully Paid

### 1.2 Why This Is Hard

The dataset has a severe **class imbalance**: only ~20% of loans default. This creates a trap:

> A model that labels every loan as "will pay" achieves **80% accuracy** while being completely useless — it misses every single default.

Our reported model accuracy is **77.7%** — essentially the baseline a dummy classifier would achieve on this cohort. This is why accuracy was treated as a **secondary/decorative metric** and all real decisions were made on ROC-AUC, KS, and Brier.

### 1.3 What Data We Used

We loaded only **origination-time columns** — information known at the moment a loan application is submitted:

| Column group | Examples |
|---|---|
| Loan terms | `loan_amnt`, `installment`, `term_months` |
| Borrower financials | `annual_inc`, `dti`, `revol_util`, `revol_bal` |
| Credit history | `delinq_2yrs`, `inq_last_6mths`, `earliest_cr_line_year` |
| Employment | `emp_length_num` |
| Categorical | `home_ownership`, `purpose`, `verification_status`, `addr_state` |

**Strictly excluded** from features (would cause data leakage):
- `total_pymnt`, `total_rec_prncp`, `total_rec_int` — only known after the loan resolves
- `recoveries`, `collection_recovery_fee` — only known post-default
- `funded_amnt` — post-origination economics

---

## 2. The Pipeline Architecture

Every model was trained inside a **leakage-free sklearn/imblearn Pipeline** — a sequential chain where all fitting happens on training data only and the test set only ever passes through pre-fitted transformers.

```
Input: raw feature matrix X
    │
    ▼
┌─────────────────────────────────────────────┐
│  [1] Median Imputation                      │  ← fitted on train, applied to test
│      Fill missing values with training median│
├─────────────────────────────────────────────┤
│  [2] StandardScaler                         │  ← fitted on train, applied to test
│      Mean=0, Std=1 for all numeric features  │
├─────────────────────────────────────────────┤
│  [3] OneHotEncoder                          │  ← fitted on train, applied to test
│      Categorical → binary columns           │
│      Rare categories (<1%) → "unknown"      │
├─────────────────────────────────────────────┤
│  [4] CorrelationThreshold(0.95)             │  ← fitted on train, applied to test
│      Drop one of each pair with |r| > 0.95  │
├─────────────────────────────────────────────┤
│  [5] SMOTE                                  │  ← TRAINING ONLY (never on test)
│      Synthetic minority oversampling        │
│      Creates synthetic default examples     │
├─────────────────────────────────────────────┤
│  [6] Classifier                             │
│      XGBoost / Logistic Regression / etc.   │
└─────────────────────────────────────────────┘
    │
    ▼
Output: pd_score ∈ [0.0, 1.0]
```

### Why SMOTE Must Live Inside the Pipeline

SMOTE creates synthetic defaulted loans by interpolating between real defaulted loans. If applied **before** the train/test split, synthetic loans derived from training samples would contaminate the test set, making the model look far better than it really is.

Inside the pipeline, SMOTE only fires during `.fit()` on training data. The test set is never oversampled.

---

## 3. Feature Importance — Mutual Information Ranking

Before any model was trained, we ranked all features by **Mutual Information (MI)** — a measure of how much knowing a feature's value reduces uncertainty about whether a loan will default.

Unlike Pearson correlation (which only captures linear relationships), MI detects any relationship — linear, nonlinear, threshold effects.

**How it was computed:**
- 30,000 training samples (never the test set)
- After full preprocessing (imputed, scaled, encoded)
- Computed with `sklearn.feature_selection.mutual_info_classif`

**Expected top features and why:**

| Feature | Why High MI |
|---|---|
| `int_rate` / `sub_grade` | LendingClub's own risk grade — directly encodes expected default probability |
| `dti` (Debt-to-Income) | Most direct measure of repayment burden — high DTI leaves no buffer for shocks |
| `revol_util` | Credit card utilization — being maxed out is a classic financial distress signal |
| `annual_inc` | Raw repayment capacity |
| `inq_last_6mths` | Desperate credit-seeking often precedes default |
| `term_months` | 60-month loans attract riskier borrower profiles |

**Important caveat:** MI ranks features in isolation. A low-MI feature may still be valuable in combination with others. The model itself captures interactions; MI is only a diagnostic guide.

---

## 4. The Model Zoo

We benchmarked 13 model families under identical conditions on 20,000 training samples to select which families to take to the final tuning stage.

### The Models Tested

| Model | Core Mechanism |
|---|---|
| Logistic Regression | Linear decision boundary in probability space |
| Decision Tree | Recursive feature-threshold splitting |
| Random Forest | 200 trees, each on random subsets → averaged |
| SVM (linear) | Maximum-margin linear hyperplane |
| SVM (rbf/poly/sigmoid) | Nonlinear kernel tricks |
| MLP (3×10 layers) | Neural network, 3 hidden layers of 10 neurons |
| Naive Bayes | Probabilistic, assumes feature independence |
| XGBoost | Gradient-boosted trees, state-of-the-art tabular |
| LightGBM | Leaf-wise gradient boosting, fast on large data |
| CatBoost | Ordered boosting, strong defaults |
| Stacking (XGB meta) | LR + DT + RF + NB → XGBoost learns from outputs |

### Zoo Results (Cross-Validated ROC-AUC, default hyperparameters)

```
SVM (linear)         0.684 ± 0.001  ← Top of the untuned zoo
Logistic Regression  0.684 ± 0.002  ← Statistical tie
CatBoost             0.673 ± 0.003
XGBoost              0.673 ± 0.005
Random Forest        0.666 ± 0.005
LightGBM             0.660 ± 0.002
SVM (rbf)            0.642 ± 0.004
SVM (poly)           0.639 ± 0.009
Stacking (XGB meta)  0.637 ± 0.009
Decision Tree        0.627 ± 0.004
MLP                  0.626 ± 0.015
Naive Bayes          0.567 ± 0.005
SVM (sigmoid)        0.564 ± 0.012
```

### Interpreting the Zoo

**Surprise result:** Linear models (SVM, LR) edge out the gradient-boosted models at default settings.

This is well-documented in credit scoring literature. After proper preprocessing (scaling, encoding, correlation removal), the relationship between features and default probability is largely **linear**. Untuned tree nonlinearity doesn't provide a big edge over a strong linear model — CatBoost and XGBoost (both 0.673) lead the boosting pack only narrowly.

**Why the gradient-boosted models were still taken to the finalist round:**
1. With Optuna tuning they close the gap with the linear leaders (and the finalist confirms they pass them).
2. XGBoost provides exact SHAP values via TreeExplainer (legally relevant for adverse-action reasons).
3. Gradient boosting is the industry standard for tabular credit risk at major lenders.

**The stacking ensemble underperformed** — the 4 base models (LR, DT, RF, NB) make similar errors, so the XGBoost meta-learner gains little from combining their outputs at default settings.

**The error bars** show cross-validation standard deviation across folds. Narrow bars mean the model performs consistently regardless of which data it sees — a sign of robustness; wide bars (MLP: ±0.015) mean high variance.

---

## 5. Finalist Comparison — Optuna-Tuned LR, RF, XGBoost, LightGBM, CatBoost

### Why the Zoo Wasn't Enough

The zoo's top result (LR ≈ SVM linear at 0.684, boosters ~0.673) used **default hyperparameters on 20K rows** — neither the best LR nor the best XGBoost. Picking a model from the zoo alone would have been based on convention, not evidence. This section gives the five top contenders identical treatment: the same out-of-time split, the same Pipeline, the same Optuna tuning budget, and the same hold-out test set.

### What Is Optuna?

Optuna is a **Bayesian hyperparameter optimization** framework. It is dramatically more efficient than exhaustive search:

| Method | How it explores | Trials needed |
|---|---|---|
| Grid Search | Exhaustively tries all combinations | Exponential in # of hyperparameters |
| Random Search | Tries random combinations with no memory | High — no learning between trials |
| **Optuna (TPE)** | Builds a surrogate model of "good regions", samples intelligently | 25 trials ≈ hundreds of grid-search points |

All Optuna trials use **3-fold cross-validation on training data only**. The hold-out test set is untouched until the very final evaluation.

### Hyperparameter Spaces Searched

**XGBoost:** `n_estimators` (200–600), `max_depth` (3–8), `learning_rate` (0.01–0.3 log), `subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0), `min_child_weight` (1–10)

**Logistic Regression:** `C` (0.001–10 log), `penalty` (l1 or l2), solver fixed to `saga` (supports both penalties)

**Random Forest:** `n_estimators` (100–500), `max_depth` (4–14), `min_samples_leaf` (2–30), `max_features` (sqrt / log2 / 0.3)
**LightGBM:** `n_estimators`, `num_leaves`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_samples`
**CatBoost:** `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`

### Actual Results — All Five Optuna-Tuned on the Out-of-Time Split

| Model | ROC-AUC | KS | Gini | PR-AUC | Brier |
|---|---|---|---|---|---|
| **CatBoost** | **0.684** | **0.272** | **0.368** | 0.373 | **0.164** |
| **XGBoost** | 0.682 | 0.264 | 0.364 | **0.374** | 0.164 |
| LightGBM | 0.681 | 0.262 | 0.362 | 0.372 | 0.164 |
| Logistic Regression | 0.672 | 0.255 | 0.344 | 0.363 | 0.166 |
| Random Forest | 0.656 | 0.228 | 0.313 | 0.346 | 0.167 |

### Interpreting the Results

**The three gradient-boosted models finish in a statistical tie** (AUC 0.681–0.684 — well inside the sampling noise of a ~20K out-of-time test). CatBoost has the slimmest AUC/KS/Gini lead; XGBoost is fractionally ahead on PR-AUC. Logistic Regression is a respectable fourth (0.672), and Random Forest trails (0.656).

**Why XGBoost is the headline model despite CatBoost's hairline lead:**
1. The 0.002 AUC gap is not statistically meaningful at this sample size — there is no true winner among the boosters.
2. XGBoost has the most mature **SHAP TreeExplainer** support, which the adverse-action reason-code module (Notebook 05) depends on for exact attributions.
3. It is the de-facto industry standard for tabular credit risk, simplifying model-risk-management review.

**Why the boosted models pass the linear leaders after tuning:** with proper depth/regularisation they capture **non-linear interactions** — the DTI risk curve's diminishing-returns shape, the interaction between `inq_last_6mths` and `revol_util` — that a hyperplane cannot form regardless of how `C` is tuned. That LR still lands within ~1pp AUC confirms the relationship is *mostly* linear, an honest and useful finding in its own right.

**Random Forest trails the boosters.** Bagging smooths error but doesn't correct it; gradient boosting sequentially corrects residuals, giving it the structural edge on tabular problems.

### Probability Calibration — Why It's Mandatory

Raw XGBoost outputs scores in [0,1] but they are not guaranteed to match observed default rates. If XGBoost says "35% default probability" but those loans actually default at 52%, the profit formula `EL = PD × LGD × EAD` uses the wrong PD and the optimal threshold is placed incorrectly.

**The fix: Isotonic Calibration** — `CalibratedClassifierCV(method='isotonic', cv=3)` fits a monotonic mapping from raw scores to observed rates on held-out folds.

> "When the calibrated model says 30%, about 30% of those loans actually defaulted."

Our headline Brier score of **0.148** sits under the OOT base-rate baseline (~0.16), confirming the calibrated probabilities are trustworthy inputs to the profit layer even under out-of-time drift.

**Verdict: XGBoost is the headline model** — it is in a statistical dead heat with CatBoost and LightGBM at the top of the finalist, and is selected for its mature SHAP TreeExplainer support and industry ubiquity rather than a meaningful metric edge.

---

## 6. Final Model Evaluation — Hold-Out Test Set

The final calibrated XGBoost was evaluated on the **untouched out-of-time hold-out** — all **2015 originations in the development sample** (85,452 loans), issued after every training loan (no temporal overlap, asserted by the test suite).

### Full Metric Results

| Metric | Score | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.704** | 70.4% chance of correctly ranking a bad loan above a good one on the 2015 OOT test |
| **KS Statistic** | **0.297** | Peak separation of 29.7% between good/bad score distributions. Industry pass (≥0.20) ✓ |
| **Gini Coefficient** | **0.408** | 40.8% of maximum possible discrimination power |
| **PR-AUC** | **0.371** | Better than random baseline (~0.20) at detecting defaults |
| **Brier Score** | **0.148** | Below the OOT base-rate baseline; calibrated ✓ |

### Metric Deep Dives

**ROC-AUC = 0.704**

Pick one defaulted loan and one non-defaulted loan at random from the 2015 test set. There is a **70.4% chance** the model assigned a higher risk score to the defaulted loan — on a true out-of-time vintage holdout. For context, bureau-based models (with FICO scores) typically achieve 0.75–0.85; without bureau data, **0.70 OOT is a solid result**.

**KS = 0.297**

Sort all test loans from lowest to highest predicted PD. Walk down the list tracking two cumulative curves: "% of all defaults captured" (should rise fast early) and "% of all good loans captured" (should rise slowly early). The KS is the maximum vertical gap = **29.7%**. Industry classification:

```
< 0.20  → Poor: reject the model
0.20–0.40 → Acceptable ✓ (we are here)
0.40–0.60 → Good
> 0.60  → Excellent (often indicates data leakage)
```

**Gini = 0.408**

Simply: `Gini = 2 × AUC − 1 = 2 × 0.704 − 1 = 0.408`. On a scale from 0 (random) to 1 (perfect), the model has **40.8% of the maximum possible discrimination power**.

**PR-AUC = 0.371**

On an imbalanced dataset, ROC-AUC can be misleadingly optimistic because it gives equal weight to the majority (good) class. PR-AUC focuses on minority class performance. A random classifier scores PR-AUC ≈ **0.20** (the OOT base rate). Our **0.371** is **~1.9× better than random** — meaningful signal for catching defaults.

**Brier = 0.148**

Mean squared error of the predicted probabilities:
- Perfect model: 0.0
- **Our model: 0.148**
- Naive baseline (always predict ~20%): `0.20 × 0.80² + 0.80 × 0.20² ≈ 0.16`
- Worst case: 0.25

Our model edges under the naive baseline, confirming the PD scores carry real information beyond the base rate even under OOT drift.

**The Precision/Recall @ 0.5 Threshold Problem**

| Metric | Score | Why it looks bad |
|---|---|---|
| Recall (Bad) | 2.4% | Almost no loans score > 0.50 — threshold is wrong |
| Precision (Bad) | 54.4% | When it does flag, it's usually right |
| F1 (Bad) | 4.5% | Collapsed by terrible recall |

The 0.5 threshold is the sklearn default — it is **not the operational threshold**. Notebook 03 finds the profit-optimal threshold at **PD ≤ 0.26** on the 2015 OOT cohort (~83% approval).

---

## 7. Business Impact — Profit Optimization

This is where the model earns its keep. The profit layer in Notebook 03 uses the calibrated PD scores to find the approval policy that maximizes portfolio profit.

### The Profit Formula

$$\text{Per-loan profit} = \text{total payments received} - \text{amount funded}$$

- **Fully paid loan**: payments > principal → positive profit (you earned interest)
- **Defaulted loan**: payments < principal → negative profit (you lost principal net of recoveries)

### The Approval Threshold Sweep

We tested every possible threshold from 0 to 1 (in 101 steps):

> "Approve all loans with predicted PD ≤ threshold. What is the total realized profit of that approved cohort?"

This is a pure **backtesting exercise** on actual historical outcomes — not a simulation.

### Results (2015 OOT cohort, 85,452 scored loans)

| Policy | Threshold | Approval Rate | Default Rate | Total Profit |
|---|---|---|---|---|
| Approve nobody | 0.00 | 0% | — | $0 |
| **Optimal (model)** | **0.26** | **82.6%** | **15.7%** | **+$58.1M** |
| Moderate | 0.20 | 69.6% | 13.6% | +$54.3M |
| Approve everyone | 1.00 | 100% | 20.1% | +$44.7M |

### The Key Business Numbers

```
Profit with model policy (PD ≤ 0.26):  +$58,124,096
Profit with no model (approve all):    +$44,697,769
                                       ──────────────
Value added by the model:              +$ 13,426,326

Default rate (model policy):  15.7%
Default rate (no model):      20.1%
                             ─────────────
Reduction in default rate:    -4.4 percentage points
```

**On the 2015 out-of-time cohort the model adds $13.4M of value** versus approving everyone, while still lending to ~83% of applicants — selective enough to matter, permissive enough to capture the profitable majority of the book.

### Why the Optimal Threshold Sits Around 0.26

The profit curve peaks where the marginal rejected loan's expected loss exceeds its contribution to portfolio profit. On the 2015 book that balance lands at **PD ≤ 0.26 (~83% approval)** — rejecting the riskiest ~17% adds $13.4M without over-tightening policy.

---

## 8. Explainability — SHAP Values

### Why Explainability Is a Legal Requirement

In the United States:
- **Equal Credit Opportunity Act (ECOA)**: Lenders must provide specific reasons for adverse actions (credit denials)
- **Fair Credit Reporting Act (FCRA)**: Similar requirements when credit reports are involved

"The algorithm said no" is not a legally acceptable explanation. SHAP provides the per-loan, per-feature breakdown required for adverse action notices.

### How SHAP Works

SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each individual prediction, based on game theory's Shapley values:

> "Among all possible orderings in which features could enter the model, what is the average marginal contribution of this feature to the prediction?"

For XGBoost, we use **TreeExplainer** — an exact algorithm that computes Shapley values in polynomial time without approximation.

**Example adverse action explanation:**
```
Loan #X declined (PD score = 0.38, threshold = 0.26)

Feature contributions:
  dti = 35.2          → +0.09  (increased risk: high debt burden)
  revol_util = 87%    → +0.07  (increased risk: near-maxed credit cards)
  annual_inc = 42,000 → +0.04  (increased risk: below-median income)
  inq_last_6mths = 3  → +0.03  (increased risk: recent credit-seeking)
  mort_acc_flag = 1   → -0.02  (decreased risk: homeowner with mortgage)
  emp_length = 10+yr  → -0.01  (decreased risk: stable employment)
```

This is exactly what goes in the adverse action notice: "Your application was declined primarily due to your debt-to-income ratio and high revolving credit utilization."

### Global Feature Importance — Actual SHAP Rankings

The global SHAP chart shows `mean(|SHAP value|)` across all test loans — the average absolute contribution of each feature to the model's output, regardless of direction.

**Actual rankings from the trained out-of-time model:**

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `term_months` | 0.374 | The single largest driver. 60-month vs 36-month is the most impactful binary split — consistent with EDA showing 60-month loans default at ~2× the rate. |
| 2 | `open_acc` | 0.231 | Number of open credit accounts — can signal creditworthiness or overextension depending on context. |
| 3 | `revol_util` | 0.196 | Revolving credit utilisation — being near-maxed on cards is a classic financial stress signal. |
| 4 | `dti` | 0.159 | Debt-to-income — monotonic risk step-up confirmed in EDA (13% at DTI<5, 30% at DTI 30–50). |
| 5 | `revol_bal` | 0.153 | Absolute revolving balance carried. |
| 6 | `inq_last_6mths` | 0.150 | Recent credit inquiries proxy financial urgency. 5+ inquiries → 33% default rate in EDA. |
| 7 | `annual_inc` | 0.144 | Higher income → lower risk. Fundamental affordability signal. |
| 8 | `emp_length_num` | 0.136 | Despite a flat univariate signal in EDA, employment length contributes in combination with other features. |
| 9 | `initial_list_status_f` | 0.123 | Fractional listing status proxies investor demand at origination. |
| 10 | `loan_amnt` | 0.123 | Larger loans default more — higher repayment burden. |

**Key insight:** `term_months` ranks #1 by a wide margin — above `dti`, `revol_util`, and `annual_inc`. The model's single biggest discriminating factor is the loan duration the borrower chose. Even holding all other characteristics constant, a 60-month borrower looks fundamentally different to this model than a 36-month borrower. This independently agrees with the Information Value ranking in the Profit-Risk scorecard.

---

## 9. Model Limitations & Caveats

| Limitation | Detail |
|---|---|
| **No bureau data** | We have no FICO scores, no payment history outside LC — bureau models typically score 0.75+ AUC vs our ~0.70 OOT |
| **Historical data** | Trained on 2007–2014 vintages; economic conditions change (COVID, rate cycles not represented) |
| **LGD** | Modeled per loan (two-stage cure + severity); MAE beats portfolio constant baseline |
| **Fairness** | Adverse-action reason codes implemented (Notebook 05); formal disparate-impact testing impossible without protected attributes |
| **Static threshold** | PD ≤ 0.26 is optimal for the 2015 OOT cohort; should be recalibrated periodically and monitored via PSI (Notebook 06) |

---

## 10. Summary

```
Problem:    Binary default prediction on 1.3M LendingClub loans
Validation: Out-of-time vintage holdout (train ≤2014, test 2015)
Features:   16 numeric + 6 categorical origination-time features
Pipeline:   Impute → Scale → OHE → CorrelationDrop(0.95) → SMOTE → XGBoost
Tuning:     Optuna Bayesian search, training folds only (LR, RF, XGB, LightGBM, CatBoost)
Calibration: Isotonic regression (3-fold CV on training data)

Final XGBoost OOT test metrics (2015 hold-out, 85,452 scored loans):
  ROC-AUC : 0.704
  KS      : 0.297  (industry pass: acceptable range 0.20–0.40 ✓)
  Gini    : 0.408

Top SHAP feature: term_months — 60-month term is the single largest risk driver

Business impact (2015 OOT hold-out):
  Optimal threshold    : PD ≤ 0.26
  Optimal approval rate: 82.6%   (default rate 15.7% vs 20.1% approve-all)
  Value added vs approve-all: +$13.4M
```
