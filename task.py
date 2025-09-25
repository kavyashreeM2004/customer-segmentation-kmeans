import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('Mall_Customers.csv')

# Fix column name issues (strip spaces)
df.columns = df.columns.str.strip()

# Select correct features with exact column names
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Handle missing values by imputing with mean
features = features.fillna(features.mean())

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Find optimal K using elbow method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal K')
plt.show()

# Fit KMeans with chosen K (example k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to original dataframe
df['Cluster'] = clusters

# Visualize clusters and centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=scaled_features[:, 0], y=scaled_features[:, 1],
    hue=clusters, palette='Set1', s=100, legend='full'
)
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    color='black', marker='X', s=300, label='Centroids'
)
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend(title='Cluster')
plt.show()

# Print cluster centers (scaled)
print("Cluster centers (scaled):")
print(kmeans.cluster_centers_)

# Save results
df.to_csv('segmented_customers.csv', index=False)
