import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from ISLP.models import (ModelSpec as MS, summarize , poly)

n = 50
rng = np.random.default_rng(1234)

x_ = rng.uniform(-2, 2, n)
y_ = 3 - 3* x_ + x_**3 +  0.5 * rng.normal(size=x_.shape)

cubic_df = pd.DataFrame({'x': x_, 'y': y_})

design = MS([ poly('x', degree =3)])
X = design.fit_transform(cubic_df)

model = sm.OLS(cubic_df['y'], X)
results = model.fit()

xx = np.linspace(-2,2,100)
new_df = pd.DataFrame({'x': xx})
newX = design.transform(new_df)
newY = results.predict(newX)
fig, ax = plt.subplots()
cubic_df.plot.scatter('x', 'y', ax=ax, label='Data')
ax.plot(xx, newY, color='C1', label='OLS Fit')
ax.plot(xx, 3 - 3* xx + xx**3, color='black', linestyle='--', label='Truth')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()
fig.savefig('ols2.pdf')


