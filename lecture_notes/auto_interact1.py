import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize)

# load and prepare data
Auto = pd.read_csv("../data/Auto.csv")
Auto.set_index('name', inplace=True)
Auto.cylinders = Auto['cylinders'].astype('category')
Auto.origin = Auto['origin'].astype('category')

# build design matrix with interaction terms
design = MS(['cylinders', 'weight', ('cylinders', 'weight')])
X = design.fit_transform(Auto)

# fit model
model = sm.OLS(Auto['mpg'], X)
results = model.fit()

# visualize results

fig, ax = plt.subplots()

for i, (cyl, g) in enumerate(Auto.groupby('cylinders')):
    ax.scatter(g['weight'], g['mpg'], alpha=0.5, label=str(cyl), color=f'C{i}')
    if cyl>3:
        xrange = np.asarray([g['weight'].min(), g['weight'].max()])
        slope = results.params['weight'] + results.params[f'cylinders[{cyl}]:weight']
        ax.plot(xrange,  results.params['intercept'] + results.params[f'cylinders[{cyl}]'] + slope *xrange, color=f'C{i}')
    else:
        xrange = np.asarray([g['weight'].min(), g['weight'].max()])
        slope = results.params['weight']
        ax.plot(xrange,  results.params['intercept']  + slope *xrange, color=f'C{i}')

ax.set_xlabel('Weight')
ax.set_ylabel('MPG')
ax.legend(title='Cylinders')

fig.tight_layout()
fig.savefig("auto_mpg_weight_cylinders_interaction.pdf")
plt.show()
