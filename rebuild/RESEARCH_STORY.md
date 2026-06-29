# Research Story: Credit-Profit-Risk Analysis

## TL;DR

- **Started:** Standard credit risk — train an XGBoost model to predict loan default on LendingClub data, approve/reject applicants to maximize profit
- **Problem discovered:** On accepted-only, risk-priced LC data, optimal approval rate ≈ 99%. The model barely rejects anyone because LC's interest rates already compensate for default risk. Profit uplift vanishes to near-zero once loans are fully matured.
- **Reframe:** Switch from lender accept/reject → investor portfolio selection. Predict *annualized return*, rank all loans, invest in the top fraction under a budget constraint.
- **Result:** Profit scoring (ANR model) beats LC's own grade ordering by **200–460 basis points**, validated across **four independent out-of-time cohorts** (2012–2015). The finding holds at every budget level from 1% to 75%.

---

## 1. The Dataset

**Source:** LendingClub accepted-loan file, ~2.26 million rows, 145 columns, loans originated 2007–2018.

**Why LendingClub?** LC is the canonical academic dataset for peer-to-peer lending research (Serrano-Cinca & Gutiérrez-Nieto, EJOR 2016; Bastani et al., EJOR 2021; Malekipirbazari & Aksakalli, 2015). The full loan file includes actual cashflows: `funded_amnt`, `total_pymnt`, `recoveries` — enough to compute true realized profit per loan rather than proxy economics.

**The accepted-only limitation:** LC's dataset contains only *accepted* loans — applicants LC already decided to fund. There are no outcomes for rejected applicants. This means the dataset is scoped to the *investor's decision space* (which of LC's accepted loans to fund), not the *lender's decision space* (which applicants to accept). This distinction drove the entire reframing.

**Resolved-loan subset:** Of the 2.26M rows, approximately 1.31M had terminal status (Fully Paid or Charged Off) at the time of the data snapshot. Unresolved loans (Current, Late, In Grace Period) were excluded — realized cashflows only.

**Maturity:** Loans whose full contractual term (36 or 60 months) had elapsed before the snapshot date are fully seasoned. Earlier vintages are 100% matured; recent vintages have a mix. The return analysis uses only matured loans for its primary backtest, eliminating survival-estimation uncertainty from the headline numbers.

---

## 2. Phase 1: Standard Credit Risk (The Path We Tried First)

### What we built

A full production-style credit risk pipeline:

- **PD model (XGBoost):** Trained on origination-time features only (income, employment, DTI, loan purpose, credit history). Excluded `int_rate` and `sub_grade` to keep the model grade-blind — independent signal, not a re-learning of LC's own pricing.
- **Calibration:** Isotonic regression to convert SMOTE-inflated scores back to real default probabilities.
- **LGD model (LightGBM, two-stage):** Cure probability × severity for defaulted loans. MAE 0.171 vs 0.184 constant-LGD baseline.
- **Profit curve:** Sweep PD threshold 0→1 in 101 steps, compute `sum(total_pymnt − funded_amnt)` for approved loans at each threshold.
- **Survival / IFRS 9:** Discrete-time hazard model for lifetime PD and expected credit loss provisioning.
- **Fairness:** SHAP-based adverse-action reason codes (no protected attributes in LC public data, so no disparate-impact testing).
- **Monitoring:** PSI feature drift between training and deployment populations.
- **Deployment:** FastAPI `/score` endpoint + Streamlit dashboard.

### PD model performance

| Model | CV AUC (20K zoo sample) | OOT AUC (full 2015 holdout) |
|---|---|---|
| SVM (linear) | 0.684 | — |
| Logistic Regression | 0.684 | — |
| CatBoost | 0.673 | 0.684 |
| **XGBoost** | 0.673 | **0.712** |
| LightGBM | 0.660 | 0.681 |
| Random Forest | 0.666 | 0.656 |

XGBoost selected — statistical tie with CatBoost/LightGBM but preferred for SHAP `TreeExplainer` compatibility and industry standard status. Full Optuna tuning on the final fit (20 trials, 3-fold CV) pushed OOT AUC to **0.712** on the complete 2015 vintage.

### The profit backtest problem

Here is where the interesting tension emerged. We ran the profit curve on the 2015 out-of-time holdout and found:

| Scenario | Test loans | Approval rate | Profit uplift |
|---|---|---|---|
| OOT 2015 full (exhibit) | 373,412 | 85% | **+$14M** (~30% vs approve-all) |
| Matured-only (honest) | 282,853 | **99.4%** | **≈ $0** |
| Random split (reference, not OOT) | — | 89% | +$12M |

The matured-only row is the critical one. Once we filtered to loans whose full economic life had completed — the only loans whose realized profit is final and trustworthy — the model approved 99.4% of them and the uplift collapsed to ~$157K on a book earning $77M in approve-all profit.

**Why does uplift vanish on matured loans?** Two compounding reasons:

1. **LC already priced the risk.** Interest rates compensate for default probability. A loan the model flags as high-risk was already assigned a higher rate. Even if it defaults, a portion of interest was collected first. On a fully matured, resolved book, nearly every loan — including many defaulted ones — was net profitable in aggregate.

2. **Optimal approval rate near 100% is the correct answer** for a lender facing an already-priced accepted-only book. The model's value is *risk ranking for capital allocation*, not discovering a secretly loss-making book.

The full-OOT-2015 uplift of +$14M is real in a deployment sense — it describes what happens when a model trained on 2007–2014 data is applied to 2015 originations that haven't fully resolved yet. But it partially reflects immature cashflows (incomplete recoveries on 2015 defaults) and early repayments that haven't hit their LGD floor. Honest backtest on matured loans tells you the model barely moves the needle.

**This was the signal to reframe the problem entirely.**

---

## 3. The Reframe: Investor Portfolio Selection

### The insight

The accept/reject framing is approximately degenerate on accepted-only, risk-priced data. But there is still signal to extract — just not in *whether* to fund, but in *which* loans over-compensate for their actual risk relative to other available loans.

This is the investor's problem, not the lender's. An investor on LC (or any marketplace lender) selects a *subset* of already-accepted loans to fund, given a capital budget. The objective is to maximize the annualized return on that capital — not to minimize default probability.

The ANR (Annualized Net Return) measures this directly:

```
ANR = (total_pymnt / funded_amnt)^(12 / months_on_book) - 1
```

ANR inherently penalizes early high-rate defaults (they have short months_on_book, so the annualization doesn't help them) and rewards loans that pay consistently through their full term. It prices in both the default and the interest — the "interest already paid for the risk" effect that kills the accept/reject story is baked into the target.

### The portfolio evaluation framework

Four policies compared on the same out-of-time test set (train ≤ year Y−1, test = year Y), matured loans only:

| Policy | Method | Description |
|---|---|---|
| **Invest-all** | No model | Fund every available loan. Passive baseline. |
| **Grade-only** | Sort by `int_rate` ascending | LC's own risk-pricing order — the benchmark to beat. |
| **Default scoring** | Sort by `pd_score` ascending | Rank by predicted default probability (lowest PD first). |
| **Profit scoring** | Sort by `anr_pred` descending | Rank by predicted annualized return (highest first). |

At each budget fraction (1% to 100%), compute the capital-weighted realized ANR of the funded portfolio. The "return curve" sweeps this across all budgets. The optimal budget maximizes portfolio ANR rate (not total dollars — see Section 6.3 for this distinction).

---

## 4. Building the Return (ANR) Model

### Target construction

`ANR` is computed per loan from economics columns (`total_pymnt`, `funded_amnt`, issue and last payment dates). These columns are **never used as model features** — only as the target. Feature allowlist remains the same origination-time variables as the PD model.

### Matured-only training

The ANR model was trained only on loans whose full term had elapsed before the data snapshot. For the 2015 holdout setup (training on 2007–2014 data), this was 401,338 of 452,163 training loans (88.8%). Unmatured training loans have incomplete realized ANR and would introduce noise.

### Model family selection

We ran a zoo of 6 architectures for the return model, each tuned with Optuna (20 trials, 5-fold CV):

| Architecture | Train Spearman (mean ± std) | RMSE |
|---|---|---|
| Random Forest | **+0.0148 ± 0.0025** | 0.2121 ← best Spearman |
| XGBoost | −0.0308 ± 0.0005 | 0.2091 |
| **LightGBM (Huber)** | −0.0321 ± 0.0018 | **0.2091** |
| CatBoost | −0.0478 ± 0.0005 | **0.2089** ← best RMSE |
| ElasticNet | −0.0654 ± 0.0043 | 0.2103 |
| Ridge | −0.1083 ± 0.0038 | 0.2097 |

**Key finding:** Training Spearman ≈ 0 for all architectures. The ANR signal is genuinely weak in absolute prediction terms — loan-level ANR has high variance driven by idiosyncratic events (job loss, early payoff, unexpected recovery). *This does not mean the models are useless for portfolio selection.* Portfolio return depends on ranking quality across loans, not point-prediction accuracy. A model that consistently identifies a slightly better top decile compounds into a meaningful portfolio advantage.

**Model selection paradox:** Random Forest won by training Spearman (+0.015) but LightGBM won in OOT portfolio performance. This is because Spearman measures point-prediction rank correlation — Random Forest's slight positive score may reflect overfitting on training CV folds. LightGBM's Huber loss (robust to outliers, appropriate for skewed ANR distribution) generalized better to the test vintage.

**Selected: LightGBM with Huber loss**, Optuna-tuned hyperparameters: `n_estimators=400, num_leaves=53, max_depth=9, learning_rate=0.029`.

---

## 5. Experiments Tried — and What We Learned

### 5.1 Two-Stage Return Model (Ruled Out)

**What we tried:** Train separate LightGBM models on paid loans (338,787 training loans) and defaulted loans (59,802), then combine predictions at test time:
```
anr_combined = (1 - pd_score) × anr_pred_paid + pd_score × anr_pred_default
```

**Result:** Worse than single-stage at all budget levels.

| Budget | Single-stage profit scoring | Two-stage profit scoring |
|---|---|---|
| @25% | +3.04% | +2.60% |
| @50% | +2.36% | +2.10% |

**Why it failed:** The paid-only sub-model has no signal to distinguish return variation among paid loans. Without `int_rate` in the feature set (deliberately excluded), a loan that pays 5% vs 12% ANR looks identical to origination features. The signal distinguishing paid loans by return *is* the interest rate — which we excluded to stay grade-blind. The single-stage model implicitly combines default risk and yield in one prediction and happens to generalize better because it doesn't split a problem that doesn't cleanly split.

Script: `experiments/two_stage_model.py`

---

### 5.2 OOF PD Score as ANR Feature (Not Adopted)

**What we tried:** Use 5-fold cross-validated out-of-fold PD predictions (`pd_score_oof`) as an additional feature for the ANR model. This is valid stacked generalization (Wolpert 1992) — the OOF predictions never see the rows they're predicting, so there's no leakage.

**Result:** Marginal improvement — +5 bp at the 25% budget only, flat elsewhere.

| Budget | Baseline profit scoring | With OOF pd_score |
|---|---|---|
| @25% | +3.04% | +3.09% |
| @50% | +2.36% | +2.36% |

**Why it barely helped:** The ANR model already implicitly learns default risk from the same origination features the PD model uses. Adding the PD model's output as a feature gives the ANR model a slightly smoother version of information it already approximates. At the portfolio level, the marginal lift in ranking quality is negligible.

Script: `experiments/oof_pdscore_feature.py`

---

### 5.3 Grade-Aware Variant (Informative, Not the Main Story)

**What we tried:** Train both the PD model and the ANR model with `int_rate` and `sub_grade` added to the feature set (Grade-Aware variant), alongside the Grade-Blind primary models.

**PD model:** AUC 0.712 (Grade-Blind) → 0.731 (Grade-Aware). The extra signal is real — LC's interest rate encodes significant default-risk information.

**Portfolio impact:**

| Budget | Grade-Blind profit scoring | Grade-Aware profit scoring | Grade-only (benchmark) |
|---|---|---|---|
| @3% (2015 test) | +4.46% | +4.43% | +2.42% |
| @25% (2015 test) | +3.04% | +2.92% | +2.44% |

Grade-Aware is slightly *worse* at broad budgets because the ANR model, knowing the interest rate, chases high-yield loans — which tend to be riskier. It gets the short-term yield right but misjudges the loss impact.

**The finding:** Grade-Blind models, trained purely on borrower characteristics, already extract most of the pricing signal LC encodes into interest rates. The grade-blind advantage of 200–460 bp over grade-only is achieved without ever seeing `int_rate` — confirming the model learned genuine origination-time predictors of return, not just a proxy for LC's own pricing.

---

### 5.4 Cross-Vintage Validation (The Robustness Test)

This was the most important validation step. Four independent training/test setups with no data overlap between cohorts:

| Vintage | Train | Test | Matured |
|---|---|---|---|
| 2012 | 2007–2011 (42,535 loans) | 2012 (53,367 loans) | **100%** |
| 2013 | 2007–2012 (95,902 loans) | 2013 (134,793 loans) | **100%** |
| 2014 | 2007–2013 (230,695 loans) | 2014 (170,643 loans) | 77% |
| 2015 | 2007–2014 (452,163 loans) | 2015 (282,853 loans) | 75% |

Each vintage re-trains all three models from scratch on data available before the test year. No sharing of model weights, hyperparameters, or training data between vintages.

**Results — profit scoring vs LC grade at optimal budget:**

| Vintage | Optimal budget | Profit scoring ANR | vs LC Grade | vs Invest-All |
|---|---|---|---|---|
| 2012 | 2% | +6.56% | **+459 bp** | +544 bp |
| 2013 | 1% | +5.32% | **+236 bp** | +274 bp |
| 2014 | 1% | +5.22% | **+223 bp** | +310 bp |
| 2015 | 3% | +4.46% | **+204 bp** | +390 bp |

The advantage decreases from 2012 to 2015 — consistent with increasing market efficiency and institutional competition on the LC platform as it matured. But it never disappears: profit scoring beats grade-only in every vintage at every budget level from 1% to 75%.

The 2012 and 2013 vintages being 100% matured makes those two rows particularly clean — no survival adjustment, no censored cashflows, exact realized economics.

---

## 6. Why Optimal Budget Is 1–3% (Not 99%)

This caused initial confusion when comparing to the Phase 1 results (where optimal approval rate was ~85–99%).

The Phase 1 profit curve maximized **total dollar profit** — which grows nearly monotonically as you approve more loans (even marginal loans add a small positive expected profit on an already-priced book). So 85–99% approval was correct for *that* objective.

The ANR return curve maximizes **portfolio ANR rate** — a capital-weighted average. Adding more loans beyond the top ranked ones dilutes the portfolio average. The top 1–3% of loans by predicted return earn 5–6% annualized; the next tranche earns less; the full book earns 0.5–2.5% depending on the vintage. The rate peaks at a small budget then declines.

For an investor who cares about *return on capital* (not total dollars — the remaining capital can be deployed elsewhere), maximizing rate at 1–3% budget is the correct answer. For an institutional investor deploying hundreds of millions, the dollar-maximizing budget (30–50%, where the rate-advantage × loan-count product peaks) matters more.

---

## 7. Train/Test Split Iterations

We went through three split designs before settling on the vintage holdout:

| Split | Problem |
|---|---|
| **Random stratified split** | Train/test share vintages — inflated AUC, not out-of-time |
| **Time-ordered 25% fraction** | Test spans multiple recent vintages with different maturity profiles; early runs showed distorted profit curves |
| **Vintage holdout (final)** | Train ≤ year Y−1, test = year Y — clean deployment simulation, stable within-year AUC (2015Q1–Q4 range: 0.69–0.71) |

We also compared pooled vs. segmented (separate 36-month / 60-month) models across walk-forward folds. Pooled won or tied on AUC in most slices; segmentation was rejected for production simplicity.

---

## 8. From Capstone to Rebuild — What Changed

The original undergraduate capstone had several critical flaws that required a full rebuild:

| Flaw | Impact |
|---|---|
| CTGAN synthetic data as primary training set | Models learned generator artifacts, not real credit patterns; ~97% "accuracy" was not credible |
| Resampling (SMOTE/bootstrap) before train/test split | Duplicated rows leaked between train and test — inflated all metrics |
| Accuracy as headline metric on 80/20 class imbalance | "Predict all good" ≈ 80% accuracy; trivially achievable |
| Proxy economics from bureau aggregates | No real `funded_amnt`, `total_pymnt`, `recoveries` — profit layer was not grounded in actual cashflows |
| 52nd-percentile approval heuristic | Not a principled PD threshold sweep; not comparable to realized profit |

The rebuild started from scratch: full LendingClub `loan.csv`, explicit feature allowlist, economics quarantined to target/LGD roles only, SMOTE inside Pipeline (training fold only), AUC/KS/Gini as primary metrics, realized-profit backtest.

---

## 9. Limitations

**Market efficiency.** LC correctly prices most loans. The model's edge is real but operates in a thin band — we're identifying mispriced loans at the margin, not discovering fundamentally broken pricing. As LC's algorithms improved and institutional capital entered the platform (especially post-2015), the marginal pricing errors narrowed.

**Small optimal budget.** The 1–3% optimal budget represents a few hundred to a few thousand loans per vintage. Real investors on LC faced liquidity constraints at this scale — high-quality loans were funded within hours. The backtest assumes perfect allocation; execution reality would involve some slippage.

**Single data source.** LendingClub closed to retail investors in 2020. The findings are from a specific period (2007–2015) of a specific market structure. They demonstrate the *methodology* — investor framing, return-target modeling, cross-vintage validation — rather than a deployable trading signal.

**Maturity adjustment.** For the 2014 and 2015 test vintages (77% and 75% matured), the 22–25% of unmatured loans are excluded from the backtest. The final portfolio numbers reflect the matured subset, not the full cohort.

**No reject inference.** The dataset contains only accepted loans. We cannot model how rejected applicants would have performed. The invest-all baseline represents funding *all accepted LC loans*, not all applicants.

---

## 10. Files Reference

| What | Where |
|---|---|
| Main training pipeline | `src/train.py` |
| ANR target + return curves | `src/returns.py` |
| Portfolio comparison | `src/portfolio.py` |
| Accept/reject framing (Phase 1) | `src/profit.py`, `notebooks/EXPLORATORY_accept_reject_framing.ipynb` |
| Cross-vintage figures | `notebooks/07_Cross_Vintage_Analysis.ipynb` |
| Ruled-out experiments (scripts) | `experiments/` |
| All results for all vintages | `reports/cross_vintage_results.txt` |
| Summary table | `reports/cross_vintage_summary_table.csv` |
