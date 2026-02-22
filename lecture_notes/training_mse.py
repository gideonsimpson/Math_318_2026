import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Reset seed
np.random.seed(100)

# Training set size values
n_train = np.array([5, 10, 20, 40, 80])

# Validation (testing) set is fixed
n_test = 10**4

# Set true coefficients
beta0_true = 2.0
beta1_true = -1.5
eps = 1.0

# Construct validation set, fixed over the entire computation
x_test = np.random.uniform(0, 15, n_test)
y_test = beta0_true + beta1_true * x_test + eps * np.random.normal(size=n_test)
test_df = pd.DataFrame({"x": x_test, "y": y_test})

# Number of samples per training set size
n_samples = 100

# Lists to store results
mse_vals = []
n_vals = []

design = MS(["x"])

# Loop over each training set size, n_samples times
for n in n_train:
    for i in range(n_samples):
        x = np.random.uniform(0, 15, n)
        y = beta0_true + beta1_true * x + eps * np.random.normal(size=n)
        train_df = pd.DataFrame({"x": x, "y": y})
        X_train = design.fit_transform(train_df)
        # Fit linear model
        model = sm.OLS(train_df["y"], X_train).fit()
        
        # Predict on test set and compute MSE
        X_test = design.transform(test_df)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(test_df["y"], y_pred)  # Alternatively: mse =
                
        # Record n and MSE
        n_vals.append(n)
        mse_vals.append(mse)

# Convert to DataFrame
mse_df = pd.DataFrame({"n": n_vals, "MSE": mse_vals})
mse_df['n'] = mse_df['n'].astype('category')

# Create boxplot
fig, ax = plt.subplots(figsize=(10, 6))
mse_df.boxplot(column="MSE", by="n", ax=ax)
ax.set_yscale("log")
ax.set_title(f"Mean Squared Error of a Linear Model\n{n_samples} Training Sets of Size {n_test}")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("MSE")
fig.suptitle("")  # Remove default title
fig.savefig("msebox.pdf")
