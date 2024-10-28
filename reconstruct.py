import pandas as pd
import numpy as np
import librosa
import librosa.display
from scipy.io.wavfile import write
from time import time
from tqdm import tqdm
from pathlib import Path

start_no = 35
end_no = 35

for i in tqdm(range(start_no, end_no+1)):

    t0 = time()

    # Step 1: Read the MFCC data from the CSV file
    csv_file = f'MFCC-files/{i:02}-MFCC.csv'  # Path to your CSV file
    mfcc_data = pd.read_csv(csv_file, header=None).values  # Assuming MFCC data

    # Step 2: Use librosa to reconstruct audio from MFCCs
    # You may need to adjust the parameters according to your data
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(mfcc_data)

    # Step 3: Write the reconstructed audio to a WAV file
    sampling_rate = 44100  # Standard for librosa, adjust if necessary
    write(f'WAV-files/{i:02}-WAV.wav', sampling_rate, reconstructed_audio)

    # Step 4: Get file duration and size
    duration = librosa.get_duration(y=reconstructed_audio, sr=sampling_rate)
    size = Path(f'MFCC-files/{i:02}-MFCC.csv').stat().st_size


    with open("time_audio_files.csv", "a") as f:
        f.write(f"{i},{time()-t0},{duration},{size}\n")
