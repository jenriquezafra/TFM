# To compute the splits of the input dataset with the percentages fixed on config

import numpy as np
import pandas as pd

# creamos una funcion que tome un dataset y saca 3

def dataframes_splits(df, pct_train, pct_val, pct_test, seed):
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

# luego esta funcion la añadimos a gen_synth