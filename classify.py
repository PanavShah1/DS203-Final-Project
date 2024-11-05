import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
<<<<<<< HEAD
=======
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from check_accuracy import check_accuracy, format_accuracy
>>>>>>> e3606a0 (removed wav)


PATH = Path('mean-var/')
TYPE = 'mean_var'
SIZE = 160
N_CLUSTERS = 7

X = []
y = []

for i in range(1, 117):
    df = pd.read_csv(PATH / f'{i:02d}-{TYPE}.csv', header=0)
    df.fillna(0, inplace=True)
    df_np = df.to_numpy()
    if df_np.size == SIZE:  # Adjust this number based on your data
        df_reshape = df_np.reshape((1, SIZE))
        print("Reshaped array:", df_reshape)
        X.append(df_reshape)
    else:
        print(f"Unexpected size: {df_np.size}. Cannot reshape to (1, {SIZE}).")

X = np.array(X)
X = X.squeeze(axis=1)
print(X.shape)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
reduced_data = pca.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X)
print("\n".join([f"{i+1} : {list(kmeans.labels_)[i]}" for i in range(len(kmeans.labels_))]))
print()
print([len(kmeans.labels_[kmeans.labels_ == i]) for i in range(N_CLUSTERS)])

# Create a figure with three subplots
fig = plt.figure(figsize=(25, 6))

# 3D Scatter Plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
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
ax2 = fig.add_subplot(133)
ax2.scatter(reduced_data[:, 0], [0 for i in range(len(reduced_data[:, 0]))], 
            alpha=0.7, c=kmeans.labels_, s=50, cmap='viridis')
<<<<<<< HEAD
ax2.set_title('PCA - 1D Visualization of Clusters')
ax2.set_xlabel('PC 1')
=======
ax3.set_title('PCA - 1D Visualization of Clusters')
ax3.set_xlabel('PC 1')

handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label),
                       markersize=10, markerfacecolor=scatter1.cmap(scatter1.norm(label))) for label in range(N_CLUSTERS)]
fig.legend(handles=handles, title='Cluster Labels', loc='upper right')


accuracy_output = check_accuracy(output)

print(format_accuracy(accuracy_output))

>>>>>>> e3606a0 (removed wav)

# Show the plot
plt.tight_layout()
plt.show()

