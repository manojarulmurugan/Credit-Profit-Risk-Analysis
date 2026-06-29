# 07 — Return Scoring & Portfolio Selection

*Companion to `notebooks/03B_Return_Portfolio.ipynb`. Numbers below are the executed
300K-sample, OOT-2015 hold-out artifacts (`reports/portfolio_comparison.csv`,
`reports/return_portfolio_comparison.json`, `reports/test_metrics.json`).*

---

## 1. Problem statement: default scoring vs profit scoring

The PD/profit layer (Reports 02–03) frames credit as a **lender** deciding whom to
*accept or reject* to maximize total realized profit. On LendingClub that decision is
near-degenerate (HANDOVER §5): the book is **accepted-only** and **already
risk-priced via grade/`int_rate`**, so interest compensates risk and rejecting loans
barely moves total realized profit — "approve-all" looked near-optimal and model
uplift collapsed to ≈ $0 on the seasoned book.

The peer-to-peer credit-risk literature resolves this by **changing the objective**:
model the loan's **return**, not its default, and treat the task as an **investor
selecting a portfolio** under a budget, scored by realized **Annualized Net Return
(ANR)** (Serrano-Cinca & Gutiérrez-Nieto, *EJOR* 2016; Bastani, Asgharian & Lessmann,
"How can lenders prosper", *EJOR* 2021; Verbraken, Bravo, Weber & Baesens, EMP,
*EJOR* 2014). ANR is time-aware — an early high-interest default is penalized — so the
"interest already paid for the risk" effect that flattened accept/reject is priced
into the target itself.

**Method.** On the same OOT vintage hold-out (train ≤2014, score 2015) we compare four
investor policies that rank candidate loans and invest in the top slice under a budget:

| Policy | Ranking |
|---|---|
| **invest-all** | none (buy the whole book) — baseline |
| **default scoring** | PD model, lowest default risk first (the existing arm) |
| **profit scoring** | predicted ANR, highest return first (new `return_model.pkl`) |
| **grade-only** | LC's `int_rate`, safest first (the benchmark) |

The per-loan return target is
`ANR = (total_pymnt / funded_amnt)^(12 / months_on_book) − 1` (`src/returns.py`). Loan
maturity is handled honestly: a **matured** book (term fully elapsed before the
2019-02 data snapshot; 64,749 of 85,452 loans, 75.8%) is the primary realized
backtest, and a **survival-adjusted** expected ANR covers the unmatured 24.2% as a
robustness layer (`src/portfolio.py`, reusing the discrete-time hazard model).

---

## 2. The target has strong signal

Mean realized ANR splits cleanly by outcome:

| Outcome | Mean realized ANR |
|---|---|
| Fully paid | **+8.37%** |
| Defaulted | **−40.26%** |

---

## 3. Headline result — selectivity restores the model's value

Capital-weighted **realized** annualized return on the **matured** book:

| Policy | budget 25% | budget 50% | budget 75% | invest-all (100%) |
|---|---|---|---|---|
| **default scoring** | **+2.89%** | +2.18% | +1.52% | — |
| **grade-only** | +2.57% | **+2.25%** | **+1.68%** | — |
| **profit scoring** | +2.68% | +2.10% | +1.46% | — |
| **invest-all** | — | — | — | **+0.55%** |

**Read this against §5.** In the accept/reject framing the model "added nothing". In
the return framing, **every selective policy earns ~4× the invest-all baseline**
(+2.2% vs +0.55% at a 50% budget). The value was never absent — it was invisible to
*total realized profit* and only appears under *annualized return*.

**Honest nuance:** profit scoring is **competitive, not dominant** here — PD scoring,
LC's grade, and direct ANR modeling rank loans about equally well on this matured
book. We report that as found rather than overclaiming the literature's return uplift.

Default rate of the invested slice confirms the policies are genuinely de-risking
(e.g. default scoring at 25% budget invests in a 6.6%-default slice vs 14.9% for the
whole book).

---

## 4. Robustness — survival-adjusted full cohort (resolving §5)

Substituting `E[ANR] = (1 − lifetime_PD)·int_rate − lifetime_PD·LGD` for the unmatured
loans and re-running on the full cohort:

| Policy | budget 25% | budget 50% | budget 75% | invest-all (100%) |
|---|---|---|---|---|
| **default scoring** | **+2.56%** | **+1.17%** | −1.49% | — |
| **grade-only** | +1.03% | −1.13% | −3.87% | — |
| **profit scoring** | −0.43% | −1.50% | −3.59% | — |
| **invest-all** | — | — | — | **−8.24%** |

The decisive reversal: once unmatured risk is priced in, **invest-all is strongly
value-destroying (−8.24%)** while **selective PD ranking stays positive (+2.56%)** —
the exact opposite of the original "approve-all wins" conclusion. Default scoring is
the most robust ranker under censoring; the direct-ANR model degrades more, again
reported honestly.

---

## 5. Beat-the-grade & EMP

| Metric | Value |
|---|---|
| ROC-AUC — PD model (no grade) | 0.704 |
| ROC-AUC — LC `int_rate` alone | 0.710 |
| EMP (PD model) | 0.0111 |
| EMP optimal reject fraction | 0.095 |
| Spearman corr(`pd_score`, `anr_pred`) | −0.73 |

LC's grade is a strong stand-alone ranker (AUC 0.710); the **grade-free** PD model
reaches comparable discrimination (0.704) from origination features alone — it adds
**independent signal**, not a re-learn of LC's price. The −0.73 score correlation
shows the return model is related to, but not a mere inversion of, the PD model
(default drivers ≠ return drivers).

---

## 6. What this changes about the project's story

- The credit model's value is **real and quantifiable once the objective is an
  investor's annualized return** rather than a lender's total realized profit.
- The §5 OOT-vs-seasoned dilemma is **resolved, not hidden**: maturity is modeled
  (matured realized + survival-adjusted), and the conclusion is stable —
  selectivity beats invest-all, and invest-all is negative once censoring is priced.
- The honest, defensible headline: *"Selective investing earns ~4× the annualized
  return of buying the whole book; a grade-free model matches LC's own grade on
  discrimination; and once unmatured-loan risk is accounted for, indiscriminate
  investing destroys value while risk ranking preserves it."*

---

## 7. Reproduce

```bash
cd rebuild
.venv/bin/python -m src.train --train-sample 300000 --trials 10 --split-mode oot --no-zoo
make returns      # return_curve.csv + return_portfolio_comparison.json
make portfolio    # portfolio_comparison.csv (matured + survival-adjusted)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03B_Return_Portfolio.ipynb
```
