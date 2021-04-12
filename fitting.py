#%%
from pandas import read_csv

# First we looked at linear fit or linear regression.

# We'll need some standard input data to work on
data = read_csv("https://milliams.com/courses/applied_data_analysis/linear.csv")
dataHead = data.head()

print(dataHead)

#          x          y
#0  3.745401   3.229269
#1  9.507143  14.185654
#2  7.319939   9.524231
#3  5.986585   6.672066
#4  1.560186  -3.358149

# Here x is the independent variable (feature in ML) and y (target in ML) is the dependent variable
# In more complicated datasets you can have multiple input/features (x_1, x_2, ..., x_n)

dataCount = data.count()

print(dataCount)

data.plot.scatter("x", "y")

# You can plot scatter plots using built-in pandas functionality.

# %%

#%%

# We are going to use scikit-learn
# This module provides many tools for statisical analysis of data.

# There are a number of steps to using this module.
# 1. Choose a model - each module is a Python class
# 2. Fit model to data
# 3. Make predictions using the model

# In this case we will use a linear regression
from sklearn.linear_model import LinearRegression

# Create an instance of the linear regression model
# Here we have set a hyperparameter (fit_intercept)
model = LinearRegression(fit_intercept=True)

# model.fit(inputs/features, targets)
# The function is expecting multiple inputs so we have to create a
# dataframe from the single column
# This will use the data to fill in the parameters of this model based on the
# provided data
model.fit(data[["x"]], data["y"])

# We can check the derived parameters
print(model.coef_)
print(model.intercept_)

# We can now make a prediction based on this model.
model.predict([[4]])

# But we can do better and draw a line from this model.
from pandas import DataFrame
from matplotlib import pyplot as plt

# This extracts the most extreme x points and then
# uses the model to predict where the corresponding y 
# points should be
xFit = DataFrame([data["x"].min(), data["x"].max()])
yPred = model.predict(xFit)

# We can then draw a line based on these predictions
fig, ax = plt.subplots()
data.plot.scatter("x", "y", ax=ax)
ax.plot(xFit[0], yPred, linestyle=":")

# %%

# Let's try without the fit_intercept hyperparameter

modelNoIntercept = LinearRegression(fit_intercept=False)

modelNoIntercept.fit(data[['x']], data['y'])
xFitNoIntercept = DataFrame([data["x"].min(), data["x"].max()])
yPredNoIntercept = modelNoIntercept.predict(xFitNoIntercept)

fig, ax = plt.subplots()
data.plot.scatter("x", "y", ax=ax)
ax.plot(xFitNoIntercept[0], yPredNoIntercept, linestyle=":")

# This forces the fit through 0 so it is a bad fit.


# %%
