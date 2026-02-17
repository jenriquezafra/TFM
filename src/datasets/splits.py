# To compute the splits of the input dataset with the percentages fixed on config

import numpy as np
import pandas as pd

# creamos una funcion que tome un dataset y saca 3


def _validate_split_pcts(pct_train, pct_val, pct_test):
    pcts = np.array([pct_train, pct_val, pct_test], dtype=float)
    if np.any(pcts < 0.0) or np.any(pcts > 1.0):
        raise ValueError("Split percentages must be in [0, 1]")
    if not np.isclose(pcts.sum(), 1.0, atol=1.0e-12):
        raise ValueError(
            f"Split percentages must sum to 1.0; got {float(pcts.sum()):.12f}"
        )


def dataframes_splits(df, pct_train, pct_val, pct_test, seed):
    _validate_split_pcts(pct_train, pct_val, pct_test)
    N = len(df)
    rng = np.random.default_rng(seed)

    # permutation
    index = rng.permutation(N)

    n_train = int(N * pct_train)
    n_val = int(N * pct_val)
    n_test = N - n_train - n_val

    idx_train = index[:n_train]
    idx_val = index[n_train:n_train+n_val]
    idx_test = index[n_train + n_val:]

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    return df_train, df_val, df_test


def dataframes_splits_stratified_quantiles(
    df,
    pct_train,
    pct_val,
    pct_test,
    seed,
    target_col="IV",
    n_bins=20,
):
    """
    Stratified split based on target quantile bins.
    Useful when target tails are rare and plain random split can unbalance them.
    """
    _validate_split_pcts(pct_train, pct_val, pct_test)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    target = df[target_col]
    if target.isna().any():
        raise ValueError(f"Target column '{target_col}' contains NaNs")

    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    n_unique = int(target.nunique(dropna=True))
    effective_bins = min(n_bins, max(1, n_unique))

    if effective_bins == 1:
        strata = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)
    else:
        try:
            strata = pd.qcut(target, q=effective_bins, labels=False, duplicates="drop")
            strata = strata.astype(np.int64)
        except ValueError:
            strata = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)

    df_aux = df.copy()
    df_aux["_iv_stratum"] = strata

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df_aux.groupby("_iv_stratum", sort=False):
        idx = rng.permutation(len(group))
        n_group = len(group)
        n_train = int(n_group * pct_train)
        n_val = int(n_group * pct_val)

        idx_train = idx[:n_train]
        idx_val = idx[n_train:n_train + n_val]
        idx_test = idx[n_train + n_val:]

        train_parts.append(group.iloc[idx_train])
        val_parts.append(group.iloc[idx_val])
        test_parts.append(group.iloc[idx_test])

    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    val_df = pd.concat(val_parts, axis=0, ignore_index=True)
    test_df = pd.concat(test_parts, axis=0, ignore_index=True)

    # Shuffle each split after concatenating all strata.
    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)
    val_df = val_df.iloc[rng.permutation(len(val_df))].reset_index(drop=True)
    test_df = test_df.iloc[rng.permutation(len(test_df))].reset_index(drop=True)

    train_df = train_df.drop(columns=["_iv_stratum"])
    val_df = val_df.drop(columns=["_iv_stratum"])
    test_df = test_df.drop(columns=["_iv_stratum"])

    return train_df, val_df, test_df
