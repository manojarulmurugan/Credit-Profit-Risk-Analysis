"""Data loading and target definition for the Lending Club rebuild.

Responsibilities:
- Read only the columns I need (memory-friendly on the 1.19 GB file).
- Define the binary target (1 = Bad / default, 0 = Good / fully paid).
- Drop unresolved loans (Current / Late / In Grace Period).
- Provide a stratified sampler for fast iteration.
"""

from __future__ import annotations

import pandas as pd

from . import config as C


def load_raw(usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw Lending Club CSV with only the needed columns."""
    if usecols is None:
        usecols = C.RAW_USECOLS
    # Some declared columns may be absent in certain dataset versions; intersect.
    available = pd.read_csv(C.RAW_CSV, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in available]
    df = pd.read_csv(C.RAW_CSV, usecols=usecols, nrows=nrows, low_memory=False)
    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map loan_status to a binary target and keep only resolved loans."""
    df = df.copy()
    status = df[C.TARGET_SOURCE]
    df[C.TARGET] = pd.NA
    df.loc[status.isin(C.BAD_STATUSES), C.TARGET] = 1
    df.loc[status.isin(C.GOOD_STATUSES), C.TARGET] = 0
    df = df[df[C.TARGET].notna()].copy()
    df[C.TARGET] = df[C.TARGET].astype(int)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Lending Club month-year date columns into usable datetime/vintage.

    `issue_d` (origination month) drives the loan vintage used for the OOT split
    and for survival analysis; `last_pymnt_d` (last payment month) is used by the
    survival module to derive the time-to-event. Both are 'Mon-YYYY' strings.
    """
    df = df.copy()
    if "issue_d" in df:
        df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["issue_year"] = df["issue_date"].dt.year
        df["issue_quarter"] = df["issue_date"].dt.to_period("Q").astype(str)
    if "last_pymnt_d" in df:
        df["last_pymnt_date"] = pd.to_datetime(
            df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    return df


def vintage_holdout_split(df: pd.DataFrame,
                          train_max_year: int = C.OOT_TRAIN_MAX_YEAR,
                          test_year: int = C.OOT_TEST_YEAR):
    """Return (train_idx, test_idx) for a single-vintage out-of-time holdout.

    Training loans are those with ``issue_year <= train_max_year``; the test set
    is all loans originated in ``test_year``. Rows with a missing issue year are
    kept in the training partition. This mirrors scoring the next origination
    vintage after fitting on all prior history (e.g. train ≤2014, test 2015).
    """
    if "issue_year" not in df:
        raise ValueError("vintage_holdout_split requires parse_dates() to have run first.")
    train_mask = df["issue_year"].le(train_max_year) | df["issue_year"].isna()
    test_mask = df["issue_year"].eq(test_year)
    return df.index[train_mask], df.index[test_mask]


def time_ordered_split(df: pd.DataFrame, test_fraction: float = C.OOT_TEST_FRACTION):
    """Return (train_idx, test_idx) split out-of-time by loan vintage.

    Loans are ordered by `issue_date`; the newest `test_fraction` form the
    out-of-time test set. This mirrors deployment (a model is fit on what has
    matured and scored on newer originations) and guarantees no temporal overlap.
    Rows with a missing issue date are kept in the training partition.
    """
    if "issue_date" not in df:
        raise ValueError("time_ordered_split requires parse_dates() to have run first.")
    dated = df[df["issue_date"].notna()].sort_values("issue_date")
    undated = df[df["issue_date"].isna()]
    n_test = int(len(dated) * test_fraction)
    test_idx = dated.index[-n_test:] if n_test > 0 else dated.index[:0]
    train_idx = dated.index[:-n_test] if n_test > 0 else dated.index
    train_idx = train_idx.union(undated.index)
    return train_idx, test_idx


def stratified_sample(df: pd.DataFrame, n: int | None,
                      random_state: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Return a class-stratified sample of size ~n (or the full frame if n is None)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    out = (
        df.groupby(C.TARGET, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
    )
    return out.reset_index(drop=True)


def load_resolved(n: int | None = None,
                  random_state: int = C.RANDOM_STATE,
                  with_dates: bool = True) -> pd.DataFrame:
    """Convenience: load raw -> target -> optional stratified sample -> dates."""
    df = load_raw()
    df = make_target(df)
    df = stratified_sample(df, n, random_state=random_state)
    if with_dates:
        df = parse_dates(df)
    return df


if __name__ == "__main__":
    frame = load_resolved(n=5000)
    print(f"Loaded {len(frame):,} resolved loans, {frame.shape[1]} columns")
    print(frame[C.TARGET].value_counts(normalize=True).rename("rate"))
