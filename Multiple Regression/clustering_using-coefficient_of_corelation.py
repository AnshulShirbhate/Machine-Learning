# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.preprocessing import StandardScaler

# # Step 1: Generate a Sample Dataset
# np.random.seed(42)
# n_samples = 100

# # Generate random data for parameters
# data = pd.DataFrame({
#     'Age': np.random.randint(20, 70, size=n_samples),
#     'BMI': np.random.uniform(18, 35, size=n_samples),
#     'Weight': np.random.uniform(50, 100, size=n_samples),
#     'BP': np.random.uniform(80, 180, size=n_samples),
#     'Height': np.random.uniform(150, 200, size=n_samples)
# })

# print("Sample Data:")
# print(data.head())

# # Step 2: Compute the Correlation Matrix
# correlation_matrix = data.corr()
# print("\nCorrelation Matrix:")
# print(correlation_matrix)

# # Step 3: Perform Clustering Based on Correlation Matrix
# # We use hierarchical clustering with the correlation matrix
# # Convert the correlation matrix to a distance matrix (1 - correlation)
# distance_matrix = 1 - correlation_matrix

# # Perform hierarchical clustering
# linked = linkage(distance_matrix, 'complete')

# # Step 4: Visualize the Clusters with a Dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#            orientation='top',
#            labels=correlation_matrix.columns,
#            distance_sort='descending',
#            show_leaf_counts=True)
# plt.title('Hierarchical Clustering Dendrogram (based on correlation)')
# plt.xlabel('Feature')
# plt.ylabel('Distance')
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 1: Generate a Sample Dataset
np.random.seed(42)
n_samples = 100

# Generate random data for parameters
data = pd.DataFrame({
    'Age': np.random.randint(20, 70, size=n_samples),
    'BMI': np.random.uniform(18, 35, size=n_samples),
    'Weight': np.random.uniform(50, 100, size=n_samples),
    'BP': np.random.uniform(80, 180, size=n_samples),
    'Height': np.random.uniform(150, 200, size=n_samples)
})

print("Sample Data:")
print(data.head())

# Step 2: Compute the Correlation Matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Step 3: Visualize with a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Heatmap of Feature Correlations')
plt.show()

# Step 4: Visualize with a Pair Plot
sns.pairplot(data)
plt.suptitle('Pair Plot of Features', y=1.02)
plt.show()
