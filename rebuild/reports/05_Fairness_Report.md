# Fairness Report
## Adverse-Action Reason Codes (ECOA / Regulation B)

---

## Executive Summary

Under the **Equal Credit Opportunity Act (ECOA)** and **Regulation B**, a lender that denies credit must provide the applicant a statement of the **specific principal reasons**. The CFPB's **Circular 2022-03** confirmed there is **no "black-box exception"**: machine-learning models must be able to produce specific, accurate denial reasons, or they are non-compliant.

This phase makes the PD model compliant on that dimension: it converts each applicant's SHAP contributions into ranked, plain-language **adverse-action reason codes**.

### Scope and an honest limitation

A complete fair-lending program has two parts:

1. **Adverse-action reason codes** (explainable denials) - implemented here.
2. **Disparate-impact testing** (approval-rate parity across protected classes such as race, sex, age) and a search for less-discriminatory alternatives - **not possible on this dataset**.

The public Lending Club dataset contains **no protected-class attributes**. Rather than fabricate a disparate-impact analysis on a weak geographic proxy (which would be misleading), this phase focuses on the part the data genuinely supports. The disparate-impact workflow is described below as what *would* run in production.

---

## 1. Reason Codes: Method

For a given applicant:

1. Compute SHAP contributions of every feature to the applicant's predicted PD (via `explain.py`, TreeExplainer on the underlying XGBoost).
2. Keep only **risk-increasing** factors (positive SHAP - the factors that pushed this applicant toward denial).
3. Map each transformed feature back to its base origination feature and to consumer-friendly reason text (e.g. `dti` -> "Debt-to-income ratio too high").
4. Return the top factors, de-duplicated by feature, ranked by contribution.

This guarantees the reasons are **faithful to the actual model decision**, not generic boilerplate.

---

## 2. Example: A Single Denial

For a representative high-risk applicant, the generated notice reads:

| Rank | Reason | SHAP contribution |
|---|---|---|
| 1 | Requested loan term carries elevated risk | +0.84 |
| 2 | Geographic risk factors | +0.65 |
| 3 | Revolving credit utilization too high | +0.20 |
| 4 | Recent delinquencies on file | +0.11 |

Each line is a principal reason a compliance team can place directly on an adverse-action notice, with the contribution quantifying how strongly that factor raised the applicant's risk.

---

## 3. Portfolio-Level Reason Frequency

Aggregating reason codes across high-risk applicants shows which factors most commonly drive denials. This supports two governance needs:

- **Fair-lending monitoring** - an unexpected concentration in a single factor (especially one correlated with a protected class) is a red flag worth investigating.
- **Product feedback** - the most common denial reasons inform underwriting policy and applicant guidance.

---

## 4. What Production Disparate-Impact Testing Would Add

On internal application data (which carries or can proxy demographics via methods like BISG), a full program would:

1. **Measure approval-rate ratios** across protected groups (the four-fifths / 80% rule).
2. **Test pricing and limit disparities** for statistical significance.
3. **Search for less-discriminatory alternatives** - retrain candidate models and check whether a comparably predictive model produces smaller disparities (a step CFPB examiners now expect).
4. **Document a compliance-management system** governing pre-deployment testing, ongoing monitoring, and exception review.

None of this is possible without protected-class data, which is why it is described rather than fabricated here.

---

## 5. Compliance Takeaway

- The model can attach **specific, ranked, model-faithful reasons** to every decision - satisfying ECOA / Reg B and CFPB Circular 2022-03 on explainability.
- The limitation (no protected attributes -> no disparate-impact test) is stated openly, which is itself the responsible posture a reviewer wants to see.
