import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm

PATH = Path('MFCC-files/')

def get_n_seconds(num, n):
    df = pd.read_csv(PATH / f'{num:02d}-MFCC.csv', index_col=False, header=None)
    # df = df.transpose()
    # print(df.head())
    COLS = len(df.columns)
    df = df.iloc[:, :n]
    print(COLS)

    df2 = pd.DataFrame()
    # Calculate mean and variance
    df2['mean'] = df.mean(axis=1)
    df2['var'] = df.var(axis=1)

    # Filter out columns with nearly identical values (variance close to zero)
    variance_threshold = 1e-8
    valid_columns = df.columns[df.var() > variance_threshold]
    
    # Calculate skewness and kurtosis only for valid columns
    df2['skew'] = df[valid_columns].apply(skew, axis=1)
    df2['kurtosis'] = df[valid_columns].apply(kurtosis, axis=1)

    # Calculate Delta (first-order differences)
    delta_df = df.diff().fillna(0)
    df2['delta_mean'] = delta_df.mean(axis=1)
    df2['delta_var'] = delta_df.var(axis=1)
    
    # Calculate Delta-Delta (second-order differences)
    delta2_df = delta_df.diff().fillna(0)
    df2['delta2_mean'] = delta2_df.mean(axis=1)
    df2['delta2_var'] = delta2_df.var(axis=1)
    
    # Normalize each column to a 0-1 range
    # df2 = (df2 - df2.min()) / (df2.max() - df2.min())

    # Save to CSV
    df2.to_csv(f'start-mean-var-15/{num:02d}-mean_var.csv', index=False)

for i in range(1, 117):
    get_n_seconds(i, 430*3)

# get_n_seconds(1, 430*3)
