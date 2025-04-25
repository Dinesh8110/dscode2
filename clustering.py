import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch



df = pd.read_csv("bot_detection_data.csv")

features = ['Retweet Count','Follower Count','Mention Count']
df_cluster = df[features].dropna()

#normalize
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_cluster)

# kmeans clustering
kmeans = KMeans(n_clusters=3,random_state=42)
df['KMean_Clusters'] = kmeans.fit_predict(x_scaled)

# DBSCAN clustering
dbscan = DBSCAN(eps=1.2,min_samples=3)
df['dbscan_Clusters'] = dbscan.fit_predict(x_scaled)

#visualizations
#Kmeans clustering
plt.figure(figsize=(8,4))
plt.title("KMeans clustering visualization ")
sns.scatterplot(x=df_cluster['Retweet Count'],y=df_cluster['Follower Count'],hue=df['KMean_Clusters'],palette='viridis')
plt.show()

plt.figure(figsize=(8,4))
plt.title("DBSCAN clustering visualization ")
sns.scatterplot(x=df_cluster['Retweet Count'],y=df_cluster['Follower Count'],hue=df['dbscan_Clusters'],palette='Set1')
plt.show()


#Silhouette sccore
print("SilHouette Score for kMeans :",silhouette_score(x_scaled,df['KMean_Clusters']))



















# # Load dataset
# file_path = "bot_detection_data.csv"  # Update path if needed
# df = pd.read_csv(file_path)

# # Select numerical features for clustering
# features = ['Retweet Count', 'Mention Count', 'Follower Count']
# df_cluster = df[features].dropna()

# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df_cluster)

# # ---- K-Means Clustering ----
# kmeans = KMeans(n_clusters=3, random_state=42)
# df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# # ---- Agglomerative Clustering (with subsample and timer) ----
# agglo_sample_limit = 1000  # Max rows to cluster
# X_agglo = X_scaled[:agglo_sample_limit]
# df_agglo = df_cluster.iloc[:agglo_sample_limit].copy()

# start = time.time()
# agglo = AgglomerativeClustering(n_clusters=3, linkage='average')
# df_agglo['Agglo_Cluster'] = agglo.fit_predict(X_agglo)
# print("Agglomerative Clustering completed in", round(time.time() - start, 2), "seconds")

# # ---- DBSCAN Clustering ----
# dbscan = DBSCAN(eps=1.2, min_samples=3)
# df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# # ---- Visualization ----
# plt.figure(figsize=(18, 5))

# # K-Means Plot
# plt.subplot(1, 3, 1)
# sns.scatterplot(x=df_cluster['Retweet Count'], y=df_cluster['Follower Count'], hue=df['KMeans_Cluster'], palette="viridis")
# plt.title("K-Means Clustering")

# # Agglomerative Plot (subsample only)
# plt.subplot(1, 3, 2)
# sns.scatterplot(x=df_agglo['Retweet Count'], y=df_agglo['Follower Count'], hue=df_agglo['Agglo_Cluster'], palette="coolwarm")
# plt.title("Agglomerative Clustering (Subset)")

# # DBSCAN Plot
# plt.subplot(1, 3, 3)
# sns.scatterplot(x=df_cluster['Retweet Count'], y=df_cluster['Follower Count'], hue=df['DBSCAN_Cluster'], palette="Set1")
# plt.title("DBSCAN Clustering")

# plt.tight_layout()
# plt.show()

# # ---- Silhouette Scores (for models with clusters > 1) ----
# print("Silhouette Score (K-Means):", silhouette_score(X_scaled, df['KMeans_Cluster']))
# print("Silhouette Score (Agglomerative - subset):", silhouette_score(X_agglo, df_agglo['Agglo_Cluster']))

# # ---- Dendrogram ----
# plt.figure(figsize=(10, 5))
# sch.dendrogram(sch.linkage(X_agglo, method='ward'))
# plt.title("Hierarchical Clustering Dendrogram (Subset)")
# plt.show()
