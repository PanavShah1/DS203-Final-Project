import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

PATH = Path('MFCC-files/')
REQUIRED_TIME = 60
required_cols = 60 * 85

def contract_file(n):
    df = pd.read_csv(PATH / f'{n:02d}-MFCC.csv', index_col=False, header=None)
    df = df.transpose()
    no_cols = len(df)
    cols_to_remove = no_cols - required_cols

    interval = math.ceil(no_cols / cols_to_remove)

    keep_rows = [i for i in range(no_cols) if i % interval != 0]
    df_reduced = df.iloc[keep_rows]

    if len(df_reduced) > required_cols:
        df_reduced = df_reduced.iloc[:required_cols]
    
    print(f"Original rows: {no_cols}, Reduced rows: {len(df_reduced)}")
    df_reduced.to_csv(f'MFCC-len-60-sec/{n:02d}-MFCC.csv', index=False)

for i in range(1, 117):
    contract_file(i)
