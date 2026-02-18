import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from ISLP.models import ModelSpec as MS
# reset seed
np.random.seed(100)

# number of samples
n = 50
# create x values
x = np.random.uniform(0, 15, size=n)
# set true coefficients
beta0_true = 2.0
beta1_true = -1.5
# set noise parameters
eps = 1.0
y = beta0_true + beta1_true * x + eps * np.random.normal(size=n)

train_df = pd.DataFrame({"x": x, "y": y})

# linear model

design = MS(['x'])
X_train = design.fit_transform(train_df)

# X_train = sm.add_constant(train_df["x"])
lm_fit = sm.OLS(train_df["y"], X_train).fit()

# testing error
n_tests = 1000
mse = np.empty(n_tests)

for j in range(n_tests):
    x = np.random.uniform(0, 15, size=n)
    y = beta0_true + beta1_true * x + eps * np.random.normal(size=n)
    test_df = pd.DataFrame({"x": x, "y": y})
    X_test = design.transform(test_df)
    preds = lm_fit.predict(X_test)
    mse[j] = mean_squared_error(test_df["y"], preds)

fig, axes = plt.subplots()
axes.hist(mse, bins=10)
axes.set_title("Test MSE Distribution")
axes.set_xlabel("Mean Squared Error")
axes.set_ylabel("Frequency")
fig.savefig("mse_variability.pdf")