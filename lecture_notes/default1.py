import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm
from ISLP.models import ModelSpec as MS

default = pd.read_csv("../data/Default.csv")

default['student'] = default['student'].astype('category')
default['default'] = default['default'].astype('category')

fig, ax = plt.subplots()
default.boxplot(column='balance', by='default', ax=ax)
ax.set_title('Boxplot of Balance by Default Status')
ax.set_xlabel('Default Status')
ax.set_ylabel('Balance')
fig.suptitle('')  # Suppress the automatic title to make it cleaner
fig.savefig('default_boxplot.pdf')

design = MS(['balance'])
X = design.fit_transform(default)
# .cat.codes converts the categorical variable to numeric codes
model = sm.OLS(default['default'].cat.codes, X)
results = model.fit()

fig, ax = plt.subplots()
ax.scatter(default['balance'], default['default'].cat.codes,label='Data', alpha=0.5)

new_balance = pd.DataFrame({'balance': np.linspace(default['balance'].min(), default['balance'].max(), 10)})
X_new = design.transform(new_balance)
preds = results.get_prediction(X_new)

ax.plot(new_balance['balance'], preds.predicted_mean, color='C1',label='OLS')

ax.set_xlabel('Balance')
ax.set_ylabel('Default (0=No, 1=Yes)')
ax.legend()
fig.savefig('default_ols.pdf')
