import numpy as np
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, squareform
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from check_accuracy import check_accuracy, format_accuracy
from sklearn.metrics import silhouette_score

# Step 1: Load and Preprocess Data
data = []
for i in tqdm(range(1, 117)):
    # Load the data (transposed so each sequence is a column)
    my_data = genfromtxt(f'MFCC-len-60-sec/{i:02d}-MFCC.csv', delimiter=',')
    data.append(my_data.T)  # Transpose to have shape (features, timesteps)

# Step 2: Normalize the Data (Standardization)
scaler = StandardScaler()
for i in tqdm(range(len(data))):
    data[i] = scaler.fit_transform(data[i].T).T  # Standardize each sequence

# Step 3: Dimensionality Reduction for Each Timestep
reduced_data = []
n_components = 20  # Reduce to 20 components per timestep
for sequence in tqdm(data):
    pca = PCA(n_components=n_components, random_state=42)
    reduced_sequence = pca.fit_transform(sequence)  # Shape becomes (timesteps, 20)
    reduced_data.append(reduced_sequence)

# Step 4: Compute DTW Distances Between Sequences
n_sequences = len(reduced_data)
dtw_distances = np.zeros((n_sequences, n_sequences))

for i in tqdm(range(n_sequences)):
    for j in range(i + 1, n_sequences):
        dist, _ = fastdtw(reduced_data[i], reduced_data[j], dist=euclidean)  # Using euclidean function here
        dtw_distances[i, j] = dist
        dtw_distances[j, i] = dist  # Make it symmetric

# Step 5: Clustering using K-medoids with the DTW Distance Matrix (6 clusters)
n_clusters = 6  # Set the number of clusters to 6
kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
clusters = kmedoids.fit_predict(dtw_distances)

# Step 6: Compute Silhouette Score
sil_score = silhouette_score(dtw_distances, clusters, metric="precomputed")

# Step 7: Output the Clustering Results and Silhouette Score
print("Cluster assignments for each sequence:", clusters)
print("Silhouette Score:", sil_score)
print("Accuracy of the clustering:")
print(format_accuracy(check_accuracy(clusters)))
