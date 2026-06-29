"""Central configuration: paths, column groups, and pipeline constants.

The feature set is defined as an explicit ALLOWLIST. This is the single most
important leakage-control decision in the project: by only ever selecting known
origination-time columns, no post-origination / outcome column can silently leak
into the model, regardless of how the raw file changes.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUILD_ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = PROJECT_ROOT / "Lending Club Dataset" / "loan.csv"
DATA_DICT = PROJECT_ROOT / "Lending Club Dataset" / "LCDataDictionary.xlsx"

DATA_DIR = REBUILD_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REBUILD_ROOT / "models"
REPORTS_DIR = REBUILD_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility / sampling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.25  # 75/25 train/test split

# Validation strategy. "oot" (headline): train on vintages through OOT_TRAIN_MAX_YEAR,
# hold out a single future origination year (OOT_TEST_YEAR) — e.g. train ≤2014,
# test 2015 — mirroring deployment on the next year's book. "oot_fraction" keeps
# the newest OOT_TEST_FRACTION by issue date as an ablation. "random" is stratified.
SPLIT_MODE = "oot"
OOT_TRAIN_MAX_YEAR = 2014   # training vintages: issue_year <= this value
OOT_TEST_YEAR = 2015        # OOT test vintage: issue_year == this value
OOT_TEST_FRACTION = 0.25    # used only when split_mode == "oot_fraction"

# The full resolved dataset is ~1.3M rows. For tractable iteration we develop on
# a stratified sample; set to None to use everything. These can be overridden via
# environment-style edits or function arguments.
ZOO_SAMPLE = 20_000      # sample for the full slow model-zoo CV comparison
TRAIN_SAMPLE = 300_000    # stratified sample for headline training; set None for full book (~1.3M)
SURVIVAL_SAMPLE = 50_000  # sample for the (memory-heavy) person-period expansion
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Profit-scoring / return arm (returns.py, portfolio.py, EMP metric)
# --------------------------------------------------------------------------- #
# Data snapshot used to decide loan maturity (a loan is "matured" once its
# contractual term has fully elapsed before the snapshot). None -> derive it from
# the data as max(last_pymnt_date), so the cut tracks whatever file is loaded.
SNAPSHOT_DATE = None
# Realized Annualized Net Return is clipped to a sane consumer-loan band so that
# very short months-on-book (a near-immediate payoff) cannot blow up the
# annualization. [-100%, +100%].
ANR_CLIP = (-1.0, 1.0)
# Expected Maximum Profit for credit scoring (Verbraken et al., 2014). Bimodal
# LGD with point masses p0 at full recovery (lambda=0) and p1 at total loss
# (lambda=1), uniform in between; ROI is the return per granted (good) loan.
# Defaults match the EMP R package / Banking-Analytics-Lab port.
EMP_P0 = 0.55
EMP_P1 = 0.10
EMP_ROI = 0.2644

# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #
TARGET = "target"  # 1 = Bad (default / charged off), 0 = Good (fully paid)
TARGET_SOURCE = "loan_status"

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
# Everything else (Current, Late, In Grace Period) is unresolved -> dropped.

# --------------------------------------------------------------------------- #
# Feature groups (ALLOWLIST)
# --------------------------------------------------------------------------- #
# Numeric features available at origination.
NUMERIC_FEATURES = [
    "loan_amnt",
    "installment",
    "annual_inc",
    "dti",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "emp_length_num",        # engineered from emp_length
    "term_months",           # engineered from term
    "earliest_cr_line_year",  # engineered from earliest_cr_line
    "pub_rec_flag",          # binarized pub_rec
    "mort_acc_flag",         # binarized mort_acc
    "pub_rec_bankruptcies_flag",  # binarized
]

# Categorical features available at origination.
CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
    "addr_state",
]

# Lender-assigned risk signals: these encode Lending Club's OWN risk model.
# The main PD model is built WITHOUT them; a side comparison includes them.
GRADE_NUMERIC = ["int_rate"]
GRADE_CATEGORICAL = ["sub_grade"]

# --------------------------------------------------------------------------- #
# Date columns. Used to derive loan vintage (OOT split) and the time-to-event
# label for survival analysis. They are NOT model features (the allowlist above
# excludes them, and they are listed in LEAKAGE_COLUMNS below so the leakage test
# asserts they never reach the feature matrix).
# --------------------------------------------------------------------------- #
DATE_COLUMNS = ["issue_d", "last_pymnt_d"]

# --------------------------------------------------------------------------- #
# Economics columns (used by the PROFIT layer ONLY - never as model features)
# --------------------------------------------------------------------------- #
ECONOMICS_COLUMNS = [
    "funded_amnt",        # EAD - exposure at default
    "total_pymnt",        # total cash received (principal + interest + recoveries)
    "total_rec_prncp",    # principal repaid
    "total_rec_int",      # interest received (realized revenue)
    "recoveries",         # post charge-off gross recovery
    "collection_recovery_fee",
]

# --------------------------------------------------------------------------- #
# Known leakage / outcome columns. NOT used as features (the allowlist already
# excludes them); kept here so tests can assert they never appear in X.
# --------------------------------------------------------------------------- #
LEAKAGE_COLUMNS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "funded_amnt", "funded_amnt_inv", "issue_d",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term", "chargeoff_within_12_mths", "collections_12_mths_ex_med",
    "pymnt_plan", "policy_code", "loan_status",
] + [c for c in [
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
] ]

# Raw columns we must read from the CSV (keeps memory low vs. all 145 cols).
# "grade" is included for EDA/visualization only (NOT a model feature - the
# allowlist in feature_columns() controls what reaches the model).
RAW_USECOLS = sorted(set(
    [TARGET_SOURCE]
    + ["loan_amnt", "installment", "annual_inc", "dti", "open_acc", "revol_bal",
       "revol_util", "total_acc", "delinq_2yrs", "inq_last_6mths",
       "emp_length", "term", "earliest_cr_line", "pub_rec", "mort_acc",
       "pub_rec_bankruptcies", "grade"]
    + CATEGORICAL_FEATURES
    + GRADE_NUMERIC + GRADE_CATEGORICAL
    + ECONOMICS_COLUMNS
    + DATE_COLUMNS
))


def feature_columns(include_grade: bool = False) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for the chosen variant."""
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    if include_grade:
        numeric = numeric + GRADE_NUMERIC
        categorical = categorical + GRADE_CATEGORICAL
    return numeric, categorical
