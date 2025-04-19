#CLUSTER ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\arist\Downloads\ICT_Subdimension_Dataset new.csv")
df = df.drop_duplicates(subset=["City"]).reset_index(drop=True)
features = df.select_dtypes(include=[np.number])

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o", linestyle="dashed")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal K")
plt.show()


optimal_k = 3 #based on the elbow plot 

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Print cluster assignments (to check if cities are correctly assigned)
print("\nCluster Assignments:")
print(df[["City", "Cluster"]].sort_values(by="Cluster"))


# Get cities properly grouped by their assigned clusters
print("\nCities in Each Cluster:")
for cluster_id in sorted(df["Cluster"].unique()):
    cities_in_cluster = df[df["Cluster"] == cluster_id]["City"].tolist()
    print(f"Cluster {cluster_id}: {cities_in_cluster}")

# Compute the best cluster based on numeric feature averages
numeric_columns = df.select_dtypes(include=["number"]).columns
best_cluster = df.groupby("Cluster")[numeric_columns].mean().sum(axis=1).idxmax()

print("\nBest Cluster:", best_cluster)

# Print cities in the best cluster correctly
best_cities = df[df["Cluster"] == best_cluster]["City"]
print("\nCities in the Best Cluster:")
print(best_cities.to_string(index=False))



