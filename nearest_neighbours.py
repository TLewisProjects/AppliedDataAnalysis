# A quick description of k-nearest neighbour algorithm is in
# nearest_neighbour.md
# %%
from pandas import DataFrame
from sklearn.datasets import make_moons

# Creates crescents of data
X, y = make_moons(n_samples=500, noise=0.15, random_state=36)
X = DataFrame(X, columns=["x1", "x2"])

X.plot.scatter("x1", "x2", c=y, colormap="Dark2", colorbar=False)

# Split into training and test datasets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Import K Nearest Neighbour Model
from sklearn.neighbors import KNeighborsClassifier

# Train on training data and use the 5 nearest data points
model = KNeighborsClassifier(n_neighbors=200).fit(train_X, train_y)

print(model.score(test_X, test_y))

def plot_knn(model, X, y, resolution=100, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    x1 = X.iloc[:,0]
    x2 = X.iloc[:,1]
    x1_range = np.linspace(x1.min()*1.1 - x1.max()*0.1, x1.max()*1.1 - x1.min()*0.1, resolution)
    x2_range = np.linspace(x2.min()*1.1 - x2.max()*0.1, x2.max()*1.1 - x2.min()*0.1, resolution)
    grid_x1_values, grid_x2_values = np.meshgrid(x1_range, x2_range)
    x_prime = np.column_stack((grid_x1_values.ravel(), grid_x2_values.ravel()))
    y_hat = model.predict(x_prime).reshape(grid_x1_values.shape)

    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(grid_x1_values, grid_x2_values, y_hat, cmap="Pastel2", alpha=1.0, shading="auto")
    X.plot.scatter(0, 1, c=y, colormap="Dark2", colorbar=False, alpha=0.8, ax=ax)

plot_knn(model, X, y)
# %%
