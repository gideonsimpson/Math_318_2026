from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.api as sm

# construct the data frame 
x_ = np.linspace(0,10,15)
rng = np.random.default_rng(1234)
y_ = 2 - 1.5 *x_ + rng.normal(size=x_.shape)
df = pd.DataFrame({'x': x_, 'y': y_})

# build the design matrix
X = df['x']
X = sm.add_constant(X)  # Accounts for the constant
# define the model
model=sm.OLS(df['y'], X)
# fit the model
results = model.fit()

# Visualize the fitted (least-squares) line
fig, ax = plt.subplots()
df.plot.scatter(x='x', y='y', ax=ax, label='Data')
ax.plot(df['x'], results.fittedvalues, color='C1', label='OLS Fit')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
fig.savefig('ols1.pdf')
