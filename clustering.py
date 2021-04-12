# Clustering is a process by which you collect a large data points
# into a smaller number of distinct groups based on distance between
# them.

# This is useful to find subset of a given population, such as
# in a census.

# This is unsupervised machine learning technique.

# A common algorithm for this is k-means clustering.

# The basic function of the algorithm is laid out in k-means.md

# Here is a simple example.
#%%
# Import the make_blobs() function from scikit-learn
from sklearn.datasets import make_blobs

# Here we specify the number of data points, the number of blob centers,
# and also a random state variable to ensure consistency between runs.
data, true_labels = make_blobs(n_samples=1000, centers=6, random_state=54)

# Format this blob data with a dataframe.
from pandas import DataFrame
points = DataFrame(data, columns=["x1", "x2"])
points.plot.scatter("x1", "x2")

# We want to find the centre of these blobs

# We import the KMeans model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5).fit(points)

# We can extract the centres from the trained model
cluster_centers = DataFrame(kmeans.cluster_centers_, columns=["x1", "x2"])

# We can plot the original data and the computed centres
ax = points.plot.scatter("x1", "x2")
cluster_centers.plot.scatter("x1", "x2", ax=ax, c="red", s=200, marker="x")

# We can also extract which data points are in each subset and colour them
# on the plot.
points.plot.scatter("x1", "x2", c=kmeans.labels_, colormap="Dark2", colorbar=False)
# %%

# Exercise 2

from sklearn.datasets import load_iris
iris = DataFrame(load_iris().data, columns=load_iris().feature_names)

print(iris.head())

kmeansIRIS = KMeans().fit(iris)

cluster_centers = DataFrame(kmeansIRIS.cluster_centers_, columns=["x1", "x2", "x3", "x4"])

print(cluster_centers)

from pandas.plotting import scatter_matrix

a = scatter_matrix(iris, figsize=(16, 16), c=kmeansIRIS.labels_)
# %%
