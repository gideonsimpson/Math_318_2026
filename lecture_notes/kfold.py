import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from ISLP.models import ModelSpec as MS
from ISLP.models import poly

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold, cross_validate


np.random.seed(100)

# consruct data
n = 100
maxp = 10

# generate full dataset
x = np.random.uniform(0, 5, size=n)
y = np.exp(x) * (1 + 0.5 * np.random.randn(n))
data_df = pd.DataFrame({"x": x, "y": y})

# fit and estimate LOOCV and KFold MSE
loocv_cv_test_vals = []
loocv_cv_train_vals = []
kfold_cv_test_vals = []
kfold_cv_train_vals = []

n_folds = 10

loocv_cv = LeaveOneOut()
kfold_cv = KFold(n_splits=n_folds, shuffle=True, random_state=318)

for p in range(1, maxp+1):
    # turn intercept off since LinearRegression will handle it
    design = MS([poly("x", degree=p)],intercept=False)
    X = design.fit_transform(data_df)

    model = LinearRegression()
    # LOOCV
    cv_results = cross_validate(model, X, data_df["y"], cv=loocv_cv, scoring='neg_mean_squared_error', return_train_score=True)
    loocv_cv_train_vals.append(-cv_results['train_score'].mean())
    loocv_cv_test_vals.append(-cv_results['test_score'].mean())

    # KFold CV
    cv_results = cross_validate(model, X, data_df["y"], cv=kfold_cv, scoring='neg_mean_squared_error', return_train_score=True)
    kfold_cv_train_vals.append(-cv_results['train_score'].mean())
    kfold_cv_test_vals.append(-cv_results['test_score'].mean())



# visualize
fig, ax = plt.subplots()
ax.scatter(np.arange(1, maxp+1), loocv_cv_train_vals,label='LOOCV Training')
ax.scatter(np.arange(1, maxp+1), loocv_cv_test_vals,label='LOOCV Testing')
ax.scatter(np.arange(1, maxp+1), kfold_cv_train_vals,label=f'KFold ({n_folds}) Training')
ax.scatter(np.arange(1, maxp+1), kfold_cv_test_vals,label=f'KFold ({n_folds}) Testing')
ax.set_xlabel('Polynomial Degree $p$')
ax.set_ylabel('MSE')
ax.legend()
fig.savefig('loocv_kfold_exp.pdf')