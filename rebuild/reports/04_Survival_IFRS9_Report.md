# Lifetime PD Report
## Discrete-Time Survival Analysis & IFRS 9 Expected Credit Loss

---

## Executive Summary

The headline PD model produces a single static probability ("will this loan ever default?"). IFRS 9 provisioning requires a **term structure** of default risk - the probability of default in each future month - to separate **12-month** from **lifetime** Expected Credit Loss (ECL). This phase builds that term structure with discrete-time survival analysis and uses it for IFRS 9 staging and ECL.

| Output | Value (OOT scoring sample) |
|---|---|
| Mean 12-month PD | ~0.07 |
| Mean lifetime PD | ~0.29 (vs realized default rate ~0.24) |
| 12-month ECL as share of lifetime ECL | ~29% |
| Stage allocation | Stage 1 majority; Stage 2 (SICR) small; Stage 3 = defaulted |

The key result: **12-month ECL is roughly one-third of lifetime ECL**, which is precisely the provisioning cliff IFRS 9 introduces when a loan migrates from Stage 1 to Stage 2.

---

## 1. The Problem: Static PD Is Not Enough for IFRS 9

The Notebook 02 model answers a binary lifetime question. IFRS 9 needs two distinct loss allowances:

- **12-month ECL** - losses from defaults expected in the next 12 months (Stage 1).
- **Lifetime ECL** - losses over the entire remaining life (Stage 2 and Stage 3).

Both require knowing *when* defaults happen, not just *whether*. That is a survival-analysis question.

---

## 2. Method: Discrete-Time Survival (Pooled Logistic Hazard)

### Time-to-event construction

Each loan is expanded into the **person-period** (long) format - one row per month observed:

- Observed duration = months from `issue_d` to `last_pymnt_d`, capped at the contractual term.
- A charged-off loan contributes `event = 0` for every survived month and `event = 1` in its default month.
- A fully-paid loan is **censored**: `event = 0` for all months, then it exits the risk set.

Dates are used only to build the label; no post-origination column enters the feature set.

### The hazard model

A logistic regression models the monthly hazard:

$$h(t) = P(\text{default in month } t \mid \text{survived to } t) = \sigma(\beta_0 + \beta_1 t + \beta_2 t^2 + \beta_3 \ln t + \mathbf{x}^\top \boldsymbol{\gamma})$$

The time terms (`period`, `period²`, `log period`) capture the **baseline hazard** shape; the origination features `x` shift each loan's hazard up or down. Crucially, **no class rebalancing** is applied - the hazard must reflect the true, low (~0.8%) monthly default probability. Rebalancing would recalibrate the rare per-period event toward 50% and saturate cumulative PD toward 1.0.

### From hazard to term structure

| Quantity | Formula |
|---|---|
| Survival | `S(t) = ∏_{s≤t} (1 − h(s))` |
| Cumulative PD | `F(t) = 1 − S(t)` |
| Marginal PD (default exactly in month t) | `S(t−1) × h(t)` |
| 12-month PD | `F(12)` |
| Lifetime PD | `F(term)` |

---

## 3. The Baseline Hazard Is Humped

Empirical monthly default risk is low at origination, rises to a peak around months 8–18, then declines as surviving borrowers prove their creditworthiness. This humped shape is why *timing* matters: two loans with the same lifetime PD can have very different 12-month PDs depending on where they sit on this curve. A static classifier cannot represent this.

---

## 4. 12-Month PD vs Lifetime PD

- Mean lifetime PD (~0.29) lands close to the realized default rate (~0.24) - a basic calibration sanity check.
- Mean 12-month PD (~0.07) is far lower: most defaults occur after the first year, so the 12-month window captures only a fraction of lifetime risk.
- Lifetime PD ≥ 12-month PD for every loan by construction (more time, more default opportunity).

The gap between the two is the economic content IFRS 9 monetizes through staging.

---

## 5. IFRS 9 Staging

| Stage | Condition | Allowance |
|---|---|---|
| **Stage 1** | Performing, no significant increase in credit risk | 12-month ECL |
| **Stage 2** | Significant Increase in Credit Risk (SICR) since origination | Lifetime ECL |
| **Stage 3** | Credit-impaired (defaulted) | Lifetime ECL, PD = 1 |

**SICR rule used here (simplified proxy):** a loan is flagged Stage 2 when its lifetime PD is both high in absolute terms (≥ 0.15) and well above the portfolio median (≥ 3× median). A production SICR test re-measures each loan's lifetime PD over time against its origination expectation using behavioural data; with a single snapshot I approximate that with a relative + absolute threshold and label it explicitly as a proxy.

---

## 6. Expected Credit Loss

$$\text{ECL} = \sum_t \text{marginalPD}(t) \times \text{LGD} \times \text{EAD} \times \frac{1}{(1+r)^{t/12}}$$

- **EAD** = `funded_amnt`.
- **LGD** = portfolio assumption (~0.62); can be swapped for the per-loan LGD model from Phase 3.
- **r** = the loan's own interest rate (discounting future losses to present value).
- Capping the sum at 12 months gives 12-month ECL; summing over the full term gives lifetime ECL.

**Result: 12-month ECL ≈ 29% of lifetime ECL.** When a loan migrates Stage 1 → Stage 2, the lender must immediately recognize the *full lifetime* allowance - a step change of roughly 3.5× in provisions for that exposure. This staging cliff is the central mechanic IFRS 9 introduced, and it is only computable with the survival term structure.

---

## 7. Why This Matters for the Portfolio

This phase elevates the project from a generic binary classifier to a credit-risk system that speaks the language of bank accounting and regulation:

- Produces a **PD term structure**, not a single number.
- Implements **IFRS 9 staging** with an explicit, documented SICR proxy.
- Computes **12-month and lifetime ECL** with discounting - the actual numbers a provisioning team books.

---

## 8. Limitations

- **Default timing** is approximated by `last_pymnt_d`; the true charge-off date can lag.
- **SICR** is a single-snapshot proxy, not a true origination-vs-current re-measurement.
- The hazard model is intentionally a transparent logistic regression; a gradient-boosted discrete-time hazard could lift discrimination at the cost of interpretability.
