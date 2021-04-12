# Hyperparameters are the consistant parameters of the model.

# scikit-learn provides a tool to automate the selection of 
# hyperparameters.

# %%
from pandas import DataFrame
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=500, centers=4, cluster_std=2.5, random_state=42)
X = DataFrame(X, columns=["x1", "x2"])

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

X.plot.scatter("x1", "x2", c=y, colormap="Dark2", colorbar=False)

# The tool we are using is GridSearchCV. This runs through various
# hyperparameters and tests which are the best.

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# This is the set of n_neighbours hyperparameters you want to search
parameters = {
    "n_neighbors" : range(1, 50),
}

# We give GridSearchCV a model and the various hyperparameters
# We run this model on the training data above
clf = GridSearchCV(KNeighborsClassifier(), parameters).fit(train_X, train_y)

# We plot the results of this search
# We plot the mean score for each possible hyperparameter candidate
cv_results = DataFrame(clf.cv_results_)
cv_results.plot.scatter("param_n_neighbors", "mean_test_score", yerr="std_test_score")

# This gives an optimum hyperparameter, n_neighbours ~ 12

# GridSearchCV also automatically fits the model with the best
# hyperparameters, so we can simply run this model.

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

plot_knn(clf, X, y)

# %%

# Final exercise

from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X = DataFrame(load_iris().data, columns=load_iris().feature_names)
X = X[["sepal length (cm)", "sepal width (cm)"]]  # Grab just two of the features
y = load_iris().target

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

parametersIRIS = {
    "n_neighbors" : range(1, 50),
}

clfIRIS = GridSearchCV(KNeighborsClassifier(), parametersIRIS).fit(train_X, train_y)

cv_results = DataFrame(clfIRIS.cv_results_)
cv_results.plot.scatter("param_n_neighbors", "mean_test_score", yerr="std_test_score")

print(clfIRIS.score(test_X, test_y))

# %%
