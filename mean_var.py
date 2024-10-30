import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

PATH = Path('cleaned-MFCC/')

def mean_var(num):
    df = pd.read_csv(PATH / f'{num:02d}-MFCC.csv')
    df = df.transpose()
    COLS = len(df.columns)
    print(COLS)

    df2 = pd.DataFrame()
    mean = df.mean(axis=1)
    var = df.var(axis=1)

    df2['mean'] = mean
    df2['var'] = var
    
    df2.to_csv(f'mean-var/{num:02d}-mean_var.csv', index=False)

for i in tqdm(range(1, 118)):
    mean_var(i)
