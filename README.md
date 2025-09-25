# customer-segmentation-kmeans
This project segments customers based on purchasing behavior using the K-Means clustering algorithm in Python. The goal is to group customers into distinct clusters based on their annual income and spending score.
# Customer Segmentation using K-Means Clustering

This project segments customers based on purchasing behavior using the K-Means clustering algorithm in Python. The goal is to group customers into distinct clusters based on their annual income and spending score.

## Dataset

The dataset used contains customer information with the following features:

- CustomerID
- Genre (Gender)
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Methodology

1. **Data Loading**: The customer dataset is loaded from a CSV file.
2. **Preprocessing**: Selected features (`Annual Income (k$)`, `Spending Score (1-100)`) are normalized using StandardScaler.
3. **Elbow Method**: Used to determine the optimal number of clusters (K) by plotting the within-cluster sum of squares (WCSS) for different K values.
4. **K-Means Clustering**: Applied with the optimal K to segment the customers.
5. **Visualization**: Scatter plots of clusters and centroids are plotted to interpret the segmentation.
6. **Output**: The dataset with cluster labels is saved to `segmented_customers.csv`.

## Usage

Run the Python script `task.py` in an environment with the necessary libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
  

Make sure the dataset CSV `Mall_Customers.csv` is in the working directory.

## Results

The project produces an elbow plot to select the number of clusters and a scatter plot visualizing customer groups with cluster centroids. The clustered data is saved for further analysis.

---

## Contact

For questions or improvements, feel free to reach out.


