import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS
from ISLP.models import poly
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

np.random.seed(100)

# test/train sample size
n = 10**3
# number of splits
m = 5
maxp = 25

# storage
test_mse_vals = []
trial_vals = []
p_vals = []

# generate full dataset
x = np.random.uniform(0, 5, size=n)
y = np.exp(x) * (1 + 0.5 * np.random.randn(n))
data_df = pd.DataFrame({"x": x, "y": y})

# repeated 50/50 train-test splits
for j in range(1, m + 1):
    train_df = data_df.sample(frac=0.5)     # 50% train
    test_df  = data_df.drop(train_df.index) # 50% test

    for p in range(1, maxp + 1):
        design = MS([poly("x", degree=p)])
        X_train = design.fit_transform(train_df)
        X_test = design.transform(test_df)

        lm_fit = sm.OLS(train_df["y"], X_train).fit()
        y_pred = lm_fit.predict(X_test)

        test_mse_vals.append(mean_squared_error(test_df["y"], y_pred))
        p_vals.append(p)
        trial_vals.append(j)

mse_df = pd.DataFrame({"p": p_vals, "MSE": test_mse_vals, "Trial": trial_vals})
mse_df["p"] = mse_df["p"].astype(int)
mse_df["Trial"] = mse_df["Trial"].astype("category")

fig, ax = plt.subplots()
for trial, g in mse_df.groupby("Trial"):
    ax.plot(g["p"], g["MSE"], linewidth=2, label=f"Trial {trial}")

ax.set_xlabel("Degree of Polynomial")
ax.set_ylabel("MSE")
ax.set_title("50% Train-Test Data Split")
ax.set_yscale("log")
ax.legend()
fig.savefig("validation_set.pdf")