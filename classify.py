import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from check_accuracy import check_accuracy, format_accuracy
from sklearn.metrics import silhouette_score

# mean-var, mean_var, 160
# start-mean-var-5, mean_var, 160
# start-mean-var-10, mean_var, 160
# start-mean-var-15, mean_var, 160
# start-mean-var-20, mean_var, 160
# mean-var-60-sec, mean_var, 160
# MFCC-shortened, MFCC-reduced, 400
# MFCC-PCA, PCA, 400  ** - main clustering
# MFCC-TSNE, MFCC-TSNE-Clustered, 400

PATH = Path('MFCC-PCA/')
TYPE = 'PCA'
SIZE = 400
N_CLUSTERS = 6
HEADER = None

X = []
y = []

for i in range(1, 117):
    df = pd.read_csv(PATH / f'{i:02d}-{TYPE}.csv', index_col=False, header=None)
    print(len(df))
    df.fillna(0, inplace=True)
    df_np = df.to_numpy()
    if df_np.size == SIZE:
        df_reshape = df_np.reshape((1, SIZE))
        X.append(df_reshape)
    else:
        print(f"Unexpected size: {df_np.size}. Cannot reshape to (1, {SIZE}).")

X = np.array(X)
X = X.squeeze(axis=1)


# Sort data based on the first column in descending order
# X_sorted = X[np.argsort(X[:, 0])]


pca = PCA(n_components=3)
reduced_data = pca.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X)
print("\n".join([f"{i+1} : {list(kmeans.labels_)[i]}" for i in range(len(kmeans.labels_))]))
print([len(kmeans.labels_[kmeans.labels_ == i]) for i in range(N_CLUSTERS)])

# Create a figure with three subplots
fig = plt.figure(figsize=(25, 6))

# 3D Scatter Plot
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                       alpha=0.7, c=kmeans.labels_, s=50, cmap='viridis')
ax1.set_title('PCA - 3D Visualization of Clusters')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')

# 2D Scatter Plot
ax2 = fig.add_subplot(132)
ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], 
            alpha=0.7, c=kmeans.labels_, s=50, cmap='viridis')
ax2.set_title('PCA - 2D Visualization of Clusters')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')

# 1D Plot of PC 1 vs 0
ax3 = fig.add_subplot(133)
ax3.scatter(reduced_data[:, 0], [0 for i in range(len(reduced_data[:, 0]))], 
            alpha=0.7, c=kmeans.labels_, s=50, cmap='viridis')
ax3.set_title('PCA - 1D Visualization of Clusters')
ax3.set_xlabel('PC 1')

# Legend for clusters
handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label+1),
                       markersize=10, markerfacecolor=scatter1.cmap(scatter1.norm(label))) for label in range(N_CLUSTERS)]
fig.legend(handles=handles, title='Cluster Labels', loc='upper right')

# Calculate accuracy
accuracy_output = check_accuracy(kmeans.labels_)
print(format_accuracy(accuracy_output))

# Silhouette Score
sil_score = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {sil_score}")
# Show the plot
plt.tight_layout()
plt.show()


