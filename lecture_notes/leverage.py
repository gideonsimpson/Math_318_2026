import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm

from ISLP.models import ModelSpec as MS

# construct data
n = 100
rng = np.random.default_rng(1234)

x_ = np.sort(rng.uniform(-2, 3, n-1))
x_ = np.append(x_, 6)  # add a high leverage point
y_ = 1 + 1.5 * x_ + .2* rng.normal(size=x_.shape)
y_[int(n/2)]+=-5  # add an outlier

y_[-1]+=+10  # add an outlier with high leverage

outlier_df = pd.DataFrame({'x': x_, 'y': y_})
outlier_df.head()

# visualize data with outliers
fig, ax = plt.subplots()
outlier_df.plot.scatter('x', 'y', ax=ax,label='Data')
ax.scatter([x_[int(n/2)]], [y_[int(n/2)]],marker='x', color='C1', label='Outlier')
ax.scatter([x_[-1]], [y_[-1]],marker='x', color='C2', label='High Leverage Outlier')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
fig.savefig('leverage_outliers1.pdf')


# fit and get influence measures
design = MS(['x'])
X = design.fit_transform(outlier_df)
model = sm.OLS(outlier_df['y'], X)
results = model.fit()

infl = results.get_influence()

# visualaize leverage vs studentized residuals
fig, ax = plt.subplots()
ax.scatter(infl.hat_matrix_diag, infl.resid_studentized_internal, label='Data')
ax.set_xlabel('Leverage')
ax.set_ylabel('Studentized Residuals')

ax.scatter([infl.hat_matrix_diag[n//2]], [infl.resid_studentized_internal[n//2]],marker='x', color='C1', label='Outlier')
ax.scatter([infl.hat_matrix_diag[-1]], [infl.resid_studentized_internal[-1]],marker='x', color='C2', label='High Leverage Outlier')
ax.legend()

fig.savefig('leverage_outliers2.pdf')

# compare fits with and without high leverage point
design2 = MS(['x'])
X2 = design2.fit_transform(outlier_df[:-1])  # remove high leverage outlier
model2 = sm.OLS(outlier_df['y'][:-1], X2)
results2 = model2.fit()

b0 = results.params['intercept']
b1 = results.params['x']
c0 = results2.params['intercept']
c1 = results2.params['x']

fig, ax = plt.subplots()
outlier_df.plot.scatter('x', 'y', ax=ax,label='Data')
ax.scatter([x_[n//2]], [y_[n//2]],marker='x', color='C1', label='Outlier')
ax.scatter([x_[-1]], [y_[-1]],marker='x', color='C2', label='High Leverage Outlier')
ax.plot(x_, b0 + b1 * x_, color='C3', label='OLS')
ax.plot(x_, c0 + c1 * x_, color='C4', label='OLS without High Leverage')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()

fig.savefig('leverage_outliers3.pdf')
