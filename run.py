import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# num = 5
MFCC_DIR = Path('MFCC-files/')



def run(num):
    df = pd.read_csv(MFCC_DIR / f'{num:02d}-MFCC.csv', header=None)
    # df.describe()
    df = df.transpose()

    COLS = len(df.columns)
    print(COLS)

    threshold = 3
    window_size = 500

    def clean_column(n):
        # Check if column `n` exists
        if n not in df.columns:
            print(f"Column {n} does not exist in DataFrame.")
            return

        mean = df[n].mean()
        std = df[n].std()

        # Calculate z-scores
        if std == 0:
            df[f'{n}_z_score'] = np.zeros(len(df)) 
        else:
            df[f'{n}_z_score'] = (df[n] - mean) / std

        cleaned_moving_avg = []
        without_outliers = []

        for i in range(len(df)):
            if np.abs(df[f'{n}_z_score'][i]) > threshold:
                cleaned_moving_avg.append(np.nan)
            else:
                cleaned_moving_avg.append(df[n][i])
                without_outliers.append(df[n][i])

        # Calculate moving average using non-outlier values
        df[f'{n}_moving_avg'] = df[n].rolling(window=window_size, center=True, min_periods=1).mean()

        cleaned_column = []
        for i in range(len(df)):
            if np.abs(df[f'{n}_z_score'][i]) > threshold:
                # Check if there are non-outlier values available
                if without_outliers:
                    # Use the moving average or the mean of non-outliers
                    cleaned_column.append(df[f'{n}_moving_avg'][i])
                else:
                    cleaned_column.append(np.nan)
            else:
                cleaned_column.append(df[n][i])

        df[f'{n}_cleaned'] = cleaned_column

    for i in tqdm(range(20)):
        clean_column(i)

    # def calculate_variance(n):
    #     return df[f'{n}_cleaned'].var()/df[f'{n}_cleaned'].mean()**2

    df2 = df.copy() # Just the cleaned columns
    df2.columns = df2.columns.astype(str)
    cleaned_columns = [col for col in df2.columns if col.endswith('_cleaned')]
    df2 = df2[cleaned_columns]
    df2.columns = [col[:-8] for col in df2.columns]  
    df2.reset_index(drop=True, inplace=True)
    df2.columns = df2.columns.astype(int)



    df3 = df2.copy()

    def standardization_column(n):
        mean = df2[n].mean()
        std = df2[n].std()
        df3[n] = (df2[n] - mean) / std

    for i in range(0):
        standardization_column(i)

    df4 = df3.copy()
    df4 = df4.transpose()
    df4.head()

    df5 = df4.copy()
    df5 = df5.transpose()

    cov_matrix = df5.cov()
    cov_matrix.to_csv(f"cov-matrix/{num:02d}-cov.csv")

for i in range(78, 116):
    run(i)