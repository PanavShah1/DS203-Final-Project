import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# Paths to the input and output directories
PATH = Path('MFCC-shortened/')
PATH_SAVE = Path('MFCC-PCA/')
PATH_SAVE.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

# Set the desired number of principal components
n_components = 20  # Adjust as needed based on your data

def apply_pca_to_file(index, n_components):
    # Load the file based on the index and transpose
    filename = f'{index:02d}-MFCC-shortened.csv'
    df = pd.read_csv(PATH / filename, header=None, index_col=False)
    df = df.transpose()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    
    # Convert PCA result to DataFrame and save
    df_pca = pd.DataFrame(df_pca)
    output_filename = f'{index:02d}-PCA.csv'
    df_pca.to_csv(PATH_SAVE / output_filename, index=False, header=False)
    print(f"PCA reduced data saved for file {index:02d} with {n_components} components.")

# Loop through files by index
for i in range(1, 117):  # Adjust the range as needed for the number of files
    apply_pca_to_file(i, n_components)
