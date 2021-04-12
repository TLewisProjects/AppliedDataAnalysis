# %%
# We need to determine whether a model is actually making good
# predictions.

# We can tweak our fit to closely fit our existing data, but
# this can lead to overfitting as we become overly focussed
# on this particular data rather than the actual process.

# We need more data to get an idea of whether the model is
# correct.

# For example, you can go through every data point using a
# high order polynomial.

# A better approach is to split your original dataset into
# different parts. A test dataset and a training dataset.

from pandas import read_csv
from sklearn.model_selection import train_test_split

# Grab the data from the last section
data = read_csv("https://milliams.com/courses/applied_data_analysis/linear.csv")

# This function performs the splitting for you
train, test = train_test_split(data, random_state=42)

print(len(train))

print(len(test))

# Now we use the training data instead of the full dataset

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)

# Fit against the training set
model.fit(train[["x"]], train["y"])

# Score against the test data
model.score(test[["x"]], test["y"])

# Score will be between 0.0 and 1.0
# %%

# Exercise for validation.py

from pandas import DataFrame
from sklearn.datasets import load_diabetes

diabetes = DataFrame(load_diabetes().data, columns=load_diabetes().feature_names)
diabetes["target"] = load_diabetes().target

train, test = train_test_split(diabetes)

diabetesModel = LinearRegression(fit_intercept=True)

diabetesModel.fit(train[["bmi", "bp"]], train["target"])

print(diabetesModel.score(test[["bmi", "bp"]], test["target"]))

from matplotlib import pyplot as plt

xFit = DataFrame([[diabetes["bmi"].min(), diabetes["bmi"].max()], [diabetes["bp"].min(), diabetes["bp"].max()]])
yPred = diabetesModel.predict(xFit)

# We can then draw a line based on these predictions
fig, ax = plt.subplots()
train.plot.scatter("bmi", "target", ax=ax)
ax.plot(xFit[1], yPred, linestyle=":")

# %%
