import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

x_ = np.linspace(0, 10, 15)
rng = np.random.default_rng(1234)
y_ = 2 - 1.5 * x_ + rng.normal(size=x_.shape)

df = pd.DataFrame({'x': x_, 'y': y_})

# build design matrix
X = sm.add_constant(df['x'])  # Adds a constant term to the predictor

# OLS = Ordinary Least Squares, specify the response variable and the design matrix
model = sm.OLS(df['y'], X) 

# Fit the model
results = model.fit()

# get predictions at the existing points
preds = results.get_prediction(X) 

# get the confidence intervals
ci_vals = preds.conf_int()

fig, ax = plt.subplots()
df.plot.scatter(x='x', y='y', ax=ax, label='Data')
ax.plot(df['x'], preds.predicted_mean, label='OLS Prediction', color='C1')
ax.fill_between(
    df['x'],
    ci_vals[:,0], # lower bound
    ci_vals[:,1], # upper bound
    color='C1',
    alpha=0.25,     # transparency
    label='95% CI',
)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('Confidence Intervals for OLS')
ax.legend()
fig.savefig('ci_plot.pdf')

# get the prediction intervals
# obs = True for the observed values
pi_vals = preds.conf_int(obs=True) 

fig, ax = plt.subplots()
df.plot.scatter(x='x', y='y', ax=ax, label='Data')
ax.plot(df['x'], preds.predicted_mean, label='OLS Prediction', color='C1')
ax.fill_between(
    df['x'],
    ci_vals[:,0], # lower bound
    ci_vals[:,1], # upper bound
    color='C1',
    alpha=0.25,     # transparency
    label='95% CI',
)
ax.fill_between(
    df['x'],
    pi_vals[:,0], # lower bound
    pi_vals[:,1], # upper bound
    color='C2',
    alpha=0.25,     # transparency
    label='95% PI',
)


ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('Confidence and Prediction Intervals for OLS')
ax.legend()
fig.savefig('ci_pi_plot.pdf')