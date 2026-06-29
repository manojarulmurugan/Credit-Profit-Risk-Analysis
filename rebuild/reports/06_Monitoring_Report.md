# Model Monitoring Report
## Population Stability Index (PSI) Drift Detection

---

## Executive Summary

A credit model is trained on one population but scores future applicants whose characteristics drift over time. **Population Stability Index (PSI)** is the banking industry's standard early-warning metric for that drift. This phase computes per-feature PSI and PD-score PSI across loan vintages (earliest vintages = reference, newest = current), giving an objective, regulator-recognized trigger for model review.

| Signal | Purpose |
|---|---|
| Per-feature PSI | Localizes *which* inputs drifted |
| PD-score PSI | Tells you whether drift actually moved the model's risk ranking |

---

## 1. What PSI Measures

PSI compares a variable's distribution in a reference window against a current window:

$$\text{PSI} = \sum_i (\text{actual}_i - \text{expected}_i) \times \ln\frac{\text{actual}_i}{\text{expected}_i}$$

Reference bin edges are quantiles (each reference bin holds ~10% of mass); the current sample is bucketed into those same edges.

**Industry bands:**

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | None |
| 0.10 – 0.25 | Moderate shift | Investigate |
| > 0.25 | Significant shift | Recalibrate / retrain |

**Connection to the scorecard:** PSI is mathematically the same divergence as the Information Value (IV) used for feature selection in the profit-risk scorecard - IV computes it between good/bad outcomes, PSI computes it between two time windows. The same binning code powers both.

---

## 2. Reference vs Current Windows

Loans are split by issue vintage: the earliest issue years form the **reference** population (what the model effectively learned), and the latest years form the **current** production population. This is the same temporal logic as the out-of-time validation, applied to drift monitoring.

---

## 3. Per-Feature PSI

Each origination feature is scored and color-banded:

- **Green (PSI < 0.10)** - stable, no action.
- **Orange (0.10–0.25)** - moderate shift, investigate.
- **Red (> 0.25)** - significant shift, candidate for recalibration.

Features tied to credit appetite and loan structure (utilization, loan amount, term mix) tend to drift most across vintages, reflecting Lending Club's evolving product and underwriting over time. The exact PSI values are regenerated into `reports/monitoring_psi.json` on each run.

---

## 4. PD-Score PSI - The Direct Signal

The most actionable single number is drift in the **PD score distribution** itself. Even when no individual feature looks alarming, a shift in the score distribution changes the approval mix and expected losses. Score PSI is computed by scoring both the reference and current populations with the deployed model and comparing the two PD distributions.

If score PSI crosses 0.10, the monitoring system flags the model for review; above 0.25, recalibration or retraining is warranted.

---

## 5. Monitoring Takeaway

PSI converts "is the model still valid?" into an objective, scheduled check:

- **Per-feature PSI** localizes the source of any drift.
- **Score PSI** measures whether that drift actually moved the model's output.
- Because PSI = IV mathematically, the monitoring reuses the scorecard machinery already built - a clean, auditable, low-overhead control.

In production this would run monthly or quarterly, with PSI thresholds wired to automated alerts and a documented model-risk-management response (investigate -> recalibrate -> retrain).
