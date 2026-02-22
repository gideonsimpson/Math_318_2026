import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from ISLP.models import ModelSpec as MS
from ISLP.models import poly

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_validate


np.random.seed(100)

# consruct data
n = 100
maxp = 10

# generate full dataset
x = np.random.uniform(0, 5, size=n)
y = np.exp(x) * (1 + 0.5 * np.random.randn(n))
data_df = pd.DataFrame({"x": x, "y": y})

# fit and estimate LOOCV MSE
cv_test_vals = []
cv_train_vals = []
for p in range(1, maxp+1):
    # turn intercept off since LinearRegression will handle it
    design = MS([poly("x", degree=p)],intercept=False)
    X = design.fit_transform(data_df)

    model = LinearRegression()
    # by default, it does not retain the training data
    cv_results = cross_validate(model, X, data_df["y"], cv=LeaveOneOut(), scoring='neg_mean_squared_error', return_train_score=True)
    cv_train_vals.append(-cv_results['train_score'].mean())
    cv_test_vals.append(-cv_results['test_score'].mean())
   
# visualize
fig, ax = plt.subplots()
ax.scatter(np.arange(1, maxp+1), cv_train_vals,label='Training')
ax.scatter(np.arange(1, maxp+1), cv_test_vals,label='Testing')
ax.set_xlabel('Polynomial Degree $p$')
ax.set_ylabel('LOOCV MSE')
ax.legend()
fig.savefig('loocv_exp.pdf')