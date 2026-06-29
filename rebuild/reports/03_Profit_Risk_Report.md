# Profit-Risk Analysis Report
## Lending Club — From Probability of Default to Portfolio Profit Optimization

> **Validation basis: out-of-time (OOT).** The PD model is trained on vintages through **2014** and the profit backtest runs on **2015 originations** (**85,452 loans** in the headline development run: 300K stratified sample from the resolved book) — the standard next-vintage holdout for a model built to score future applicants.

---

## Executive Summary

The probability of default model is a risk-ranking engine. This phase converts those risk scores into **dollar decisions** using real loan economics and a per-loan LGD model.

**Headline result (OOT 2015 test cohort, scored hold-out from `test_with_pd.parquet`):**

| Metric | Value |
|---|---|
| Portfolio LGD (avg realized loss on defaults) | **0.59** |
| Profit — approve everyone (no model) | **+$44.7M** |
| Profit — model-optimal policy (PD ≤ 0.26) | **+$58.1M** |
| **Value added by the model vs approve-all** | **+$13.4M** |
| Optimal approval rate | **82.6%** |
| Default rate under optimal policy | **15.7%** (vs 20.1% approve-all) |

Ranking by predicted PD and declining the riskiest ~17% of applicants adds **$13.4M** of realized profit on the 2015 OOT hold-out while cutting the approved default rate by 4.4 percentage points.

---

## 1. The Problem This Phase Solves

The PD model outputs one number per loan: `pd_score ∈ [0, 1]`. That alone doesn't decide a loan — the decision depends on the loan's economics (amount at stake, recovery if it defaults, interest if it pays). This phase provides the decision framework:

1. **Per-loan Expected Loss** (`EL = PD × LGD × EAD`)
2. **Profit curve** — sweep the approval threshold against realized profit
3. **WoE/IV scorecard** — regulatory-style feature validation

---

## 2. The Expected Loss Formula

$$\text{EL} = \underbrace{PD}_{\text{calibrated model}} \times \underbrace{LGD}_{\text{loss severity}} \times \underbrace{EAD}_{\text{funded\_amnt}}$$

- **PD** — calibrated XGBoost output (isotonic; OOT AUC 0.704 on the scored hold-out).
- **EAD** — `funded_amnt`, the dollars at risk.
- **LGD** — **per-loan prediction** from the two-stage LGD model (cure + severity LightGBM), not a single constant.

---

## 3. Per-Loan LGD (Modeled, Not Assumed)

The LGD is modeled per loan with a two-stage LightGBM model trained on training-partition defaults:

| LGD model metric (OOT defaults) | Value |
|---|---|
| MAE — LGD model | **< portfolio constant baseline** |
| Approach | Cure probability × severity on charged-off loans |

Per-loan LGD feeds the expected-loss layer; the **profit curve itself** uses **realized** cash (`total_pymnt − funded_amnt`) so the optimal threshold is an honest backtest on observed outcomes.

---

## 4. Realized Profit (Backtest Ground Truth)

```
Realized profit per loan = total_pymnt − funded_amnt
```

Positive for fully-paid loans (interest earned), negative for charged-off loans (principal lost net of recoveries). This is actual historical cash, not a prediction.

**Total realized profit, approve everyone on the 2015 OOT cohort: +$44.7M.** The profit-optimal policy improves on that baseline by selectivity, not by flipping a loss-making book.

---

## 5. The Profit Curve

Sweeping the PD approval threshold (approve a loan when `pd_score ≤ t`) and summing realized profit of the approved subset:

| Threshold | Approval rate | Default rate (approved) | Total realized profit |
|---|---|---|---|
| 0.10 | 25.5% | 7.5% | +$24.6M |
| 0.20 | 69.6% | 13.6% | +$54.3M |
| **0.26 (optimal)** | **82.6%** | **15.7%** | **+$58.1M** |
| 0.30 | 88.0% | 16.8% | +$57.9M |
| 1.00 (approve all) | 100.0% | 20.1% | +$44.7M |

The curve rises from low approval rates, peaks near **83% approval**, then falls as high-PD loans that would still have been profitable are excluded. The model's PD ranking identifies which marginal applicants destroy portfolio profit.

**Graph interpretation:** the blue line (total profit) climbs as more low-PD loans are included, peaks at the optimal cutoff, then declines when too many risky loans are added. The red line (default rate of approved) rises monotonically with approval rate — the art is finding the knee where profit is maximized before defaults dominate.

---

## 6. Scorecard View: Weight of Evidence / Information Value

The classic credit-scoring lens (Basel / IFRS 9 language). Information Value ranks each feature's standalone separating power:

| IV | Predictive power |
|---|---|
| < 0.02 | Useless |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| 0.30 – 0.50 | Strong |
| > 0.50 | Suspicious (check leakage) |

`term_months` remains among the strongest origination-time signals, consistent with the SHAP ranking from the modeling phase. Exact IVs are recomputed in the notebook.

---

## 7. Business Impact

```
Policy            Approval Rate   Default Rate   Total Profit
─────────────────────────────────────────────────────────────
Approve everyone     100.0%          20.1%        +$44,697,769
Model-optimal         82.6%          15.7%        +$58,124,096
─────────────────────────────────────────────────────────────
Improvement          −17.4pp         −4.4pp       +$13,426,326
```

By ranking on predicted PD and declining the riskiest ~17% of applicants, the model adds **$13.4M** (~30% uplift on the approve-all baseline) and materially reduces the default rate among approved loans.

---

## 8. Summary

```
Validation              : Out-of-time vintage holdout (train ≤2014, test 2015)
Development sample      : 300K stratified rows; 85,452 scored 2015 OOT loans
PD model (OOT)          : AUC 0.704, KS 0.297, Gini 0.408
LGD                     : per-loan two-stage model beats portfolio constant (MAE)
Optimal PD threshold    : 0.26  (approve 82.6%, default rate 15.7%)
Profit at optimum       : +$58.1M  vs  +$44.7M approve-all
Value added vs approve-all: +$13.4M
```

See `reports/02_Modeling_Report.md` for the full model-selection and calibration process.
