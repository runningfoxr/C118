import pandas
import plotly.express as px

print("Loading data...")
data_frame = pandas.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C118/stars.csv")
size = data_frame["Size"].to_list()
light = data_frame["Light"].to_list()
print("Data loaded!")

print("\Data: ")
print(data_frame.head())


data_chart = px.scatter(x=size, y=light, color=light, labels=dict(x="Size of Stars", y="Light Level", color="Light Level"), title="Size of Star vs Light the Star gives off")
data_chart.show()

from sklearn.cluster import KMeans

x = data_frame.iloc[:, [0, 1]].values

print("Getting WCSS...")
wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    k_means.fit(x)

    wcss.append(k_means.inertia_)

print("Done getting WCSS!")


import matplotlib.pyplot as plt
import seaborn

plt.figure(figsize=(10,5))
seaborn.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


k_means = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = k_means.fit_predict(x)

plt.figure(figsize=(15,7))
seaborn.scatterplot(x[y_kmeans==0, 0], x[y_kmeans==0, 1], color='pink', label='Cluster 1')
seaborn.scatterplot(x[y_kmeans==1, 0], x[y_kmeans==1, 1], color='blue', label='Cluster 2')
seaborn.scatterplot(x[y_kmeans==2, 0], x[y_kmeans ==2, 1], color='purple', label='Cluster 3')
seaborn.scatterplot(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], color='red', label='Centroids')
plt.grid(False)
plt.title('Clusters of Stars')
plt.xlabel('Size of Stars')
plt.ylabel('Light Level')
plt.legend()
plt.show()
