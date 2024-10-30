import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import os

def load_mfcc(file_path):
    """Load MFCC features from a CSV file."""
    return pd.read_csv(file_path).values

def compress_mfcc(mfcc, target_shape):
    """Compress MFCC to the target shape using mean."""
    # Average over the time axis to get a mean representation
    if mfcc.shape[0] > target_shape[0]:
        # If candidate has more rows than the target, average them
        return np.mean(mfcc[:target_shape[0]], axis=0)
    elif mfcc.shape[0] < target_shape[0]:
        # If candidate has fewer rows, pad with zeros
        padding = np.zeros((target_shape[0] - mfcc.shape[0], mfcc.shape[1]))
        return np.vstack((mfcc, padding))
    else:
        return mfcc

def is_similar_mfcc(candidate_mfcc, reference_mfcc, similarity_threshold):
    """Check if candidate MFCC is similar to reference MFCC."""
    # Calculate mean of the MFCCs for comparison
    candidate_mean_compressed = np.mean(candidate_mfcc, axis=0)
    reference_mean = np.mean(reference_mfcc, axis=0)
    
    # Check if the shapes match before calculating similarity
    if candidate_mean_compressed.shape != reference_mean.shape:
        print(f"Shape mismatch: Candidate {candidate_mean_compressed.shape}, Reference {reference_mean.shape}")
        return False

    similarity = 1 - cosine(candidate_mean_compressed, reference_mean)
    print(f"Similarity: {similarity:.4f}")
    
    return similarity >= similarity_threshold

def main(candidate_folder, reference_file, similarity_threshold=0.8):
    """Main function to check for anthem similarity."""
    # Load reference MFCC
    known_anthem_mfcc = load_mfcc(reference_file)
    
    # Print shapes for debugging
    print(f"Reference shape: {known_anthem_mfcc.shape}")

    # Iterate through candidate files
    detected_anthems = []
    for candidate_file in os.listdir(candidate_folder):
        if candidate_file.endswith('.csv'):
            candidate_path = os.path.join(candidate_folder, candidate_file)
            candidate_mfcc = load_mfcc(candidate_path)
            print(f"Candidate shape: {candidate_mfcc.shape}")

            # Compress candidate MFCC to match reference shape
            compressed_candidate_mfcc = compress_mfcc(candidate_mfcc, known_anthem_mfcc.shape)
            
            # Check similarity
            if is_similar_mfcc(compressed_candidate_mfcc, known_anthem_mfcc, similarity_threshold):
                detected_anthems.append(candidate_file)
                print(f"Detected anthem: {candidate_file}")

    print(f"Detected anthem files: {detected_anthems}")

if __name__ == "__main__":
    candidate_folder = 'MFCC-files'  # Set this to your folder path
    reference_file = 'MFCC-files/117-MFCC.csv'  # Set this to your reference file
    main(candidate_folder, reference_file)
