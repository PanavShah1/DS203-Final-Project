import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm

PATH = Path('cleaned-MFCC/')

def mean_var(num):
    df = pd.read_csv(PATH / f'{num:02d}-MFCC.csv')
    df = df.transpose()
    COLS = len(df.columns)
    print(COLS)

    df2 = pd.DataFrame()
    
    # Calculate mean and variance
    df2['mean'] = df.mean(axis=1)
    df2['var'] = df.var(axis=1)

    # Calculate skewness and kurtosis
    df2['skew'] = df.apply(skew, axis=1)
    df2['kurtosis'] = df.apply(kurtosis, axis=1)

    # Calculate Delta (first-order differences)
    delta_df = df.diff().fillna(0)  # Filling NaN from differencing
    df2['delta_mean'] = delta_df.mean(axis=1)
    df2['delta_var'] = delta_df.var(axis=1)
    
    # Calculate Delta-Delta (second-order differences)
    delta2_df = delta_df.diff().fillna(0)
    df2['delta2_mean'] = delta2_df.mean(axis=1)
    df2['delta2_var'] = delta2_df.var(axis=1)
    
    # Save to CSV
    df2.to_csv(f'mean-var/{num:02d}-mean_var.csv', index=False)

# Process each file
for i in tqdm(range(1, 118)):
    mean_var(i)
