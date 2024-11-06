import numpy as np
import pandas as pd
from pathlib import Path

# Paths to the folders
PATH = Path('MFCC-files/')
PATH_SAVE = Path('MFCC-shortened/')

def shorten_MFCC(num, n):
    # Read the CSV file
    df = pd.read_csv(PATH / f'{num:02d}-MFCC.csv', index_col=False, header=None)
    df = df.transpose()  # Transpose for easier manipulation if necessary
    # Calculate the number of full segments based on 5-second chunks
    length_segment = n * 86
    num_segments = len(df) // length_segment
    print(f"Number of 5-second segments: {num_segments}")

    # DataFrame to store the mean of each segment
    df2 = pd.DataFrame()

    # Loop over each segment and compute the mean
    for i in range(num_segments):
        start = i * length_segment
        end = start + length_segment
        df_segment = df.iloc[start:end]
        
        # Calculate mean of each segment (average over rows for each column)
        segment_mean = df_segment.mean(axis=0)
        
        # Append the mean values as a new row in df2
        df2 = pd.concat([df2, pd.DataFrame([segment_mean])], ignore_index=True)
    
    # Save the result to a new CSV file without header or index
    df2.to_csv(PATH_SAVE / f'{num:02d}-MFCC-shortened.csv', index=False, header=False)
    print(f"Shortened file saved for MFCC-{num}.")

# Run the function for example file 2 and segment length of 5 seconds
for i in range(1, 117):
    shorten_MFCC(i, 1)
