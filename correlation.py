# When looking at new data, we want to understand how the factors
# related.

# Often, we look at linear correlation. Does one factor get bigger
# as the another factor gets bigger?

# However, correlation is broader concept. It is more accurate to
# say correlation means how much predictive power does one set of
# provide about another set of data.

# This does not have to be a linear relationship.

# Another definition is the amount of mutual information - how 
# much information is shared between two variables.

# Let's try to understand correlation using Python

#%%
import numpy as np
from pandas import DataFrame

a = np.arange(100)
b = np.arange(100) * 2
df = DataFrame({"a": a, "b": b})
df.head()

# This function provides summary statistics for the dataframe
df.describe()

# This function provides a matrix of correlation coefficients
# It lists out the correlation between all variables in this.
df.corr()

# We can tweak the code to give a negative correlation
b = np.arange(100) * -2
df2 = DataFrame({"a": a, "b": b})
df2.corr()

#%%

## Multiple Cross-Correlation
from sklearn.datasets import fetch_california_housing

califHousingData = fetch_california_housing()

housing = DataFrame(califHousingData.data, columns=califHousingData.feature_names)

housing.head()

# This is different features of different census blocks in California.

# Let's have a look at the correlation.
corr = housing.corr()
corr.replace(1.0, np.nan)

# We can plot a heatmap to make it easier to find the largest correlation
import seaborn as sns

# We define a diverging colourmap so that 0.0 is white
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(corr, vmin=-1.0, vmax=1.0, cmap=cmap)

# We can also use a pandas scatter matrix
from pandas.plotting import scatter_matrix

a = scatter_matrix(housing, figsize=(16, 16))

# There is a positive correlation between income and bedrooms.
# There is a negative correlation between latitude and longitude.
# This is only due to the shape of California.

# %%
from sklearn.datasets import load_breast_cancer

breastCancerData = load_breast_cancer()

data = DataFrame(breastCancerData.data, columns=breastCancerData.feature_names)

breastCorr = data.corr()

print(breastCorr)

cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(breastCorr, vmin=-1.0, vmax=1.0, cmap=cmap)

# %%
