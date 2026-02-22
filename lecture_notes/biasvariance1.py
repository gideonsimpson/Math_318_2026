import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from ISLP.models import ModelSpec as MS
from ISLP.models import poly
from sklearn.metrics import mean_squared_error

# Reproducibility
np.random.seed(100)

# test/train sample size
n = 10**2
# number of independent runs
m = 10**2

maxp = 10

# preallocate arrays
test_MSE_vals = np.zeros(maxp)
train_MSE_vals = np.zeros(maxp)

# generate training data
x_train = np.random.uniform(0, 5, n)
y_train = np.exp(x_train) * (1 + 0.5 * np.random.randn(n))
train_df = pd.DataFrame({"x": x_train, "y": y_train})

# loop over independent test sets
for j in range(m):

    # generate testing data
    x_test = np.random.uniform(0, 5, n)
    y_test = np.exp(x_test) * (1 + 0.5 * np.random.randn(n))
    test_df = pd.DataFrame({"x": x_test, "y": y_test})

    # try polynomial fits
    for p in range(1, maxp + 1):

        # create model specification using orthogonal polynomials
        design = MS([poly("x", degree=p)])
        
        X_train = design.fit_transform(train_df)
        X_test = design.transform(test_df)

        # fit model
        model = sm.OLS(train_df["y"], X_train).fit()

        # training MSE
        train_MSE_vals[p-1] += np.mean(model.resid**2) / m

        # testing MSE
        y_pred_test = model.predict(X_test)
        test_MSE_vals[p-1] +=mean_squared_error(test_df["y"], y_pred_test) / m


# plot
fig, ax = plt.subplots()
ax.plot(range(1, maxp + 1), train_MSE_vals, label="Train MSE")
ax.plot(range(1, maxp + 1), test_MSE_vals, label="Test MSE")
ax.set_xlabel("Degree Polynomial")
ax.set_ylabel("MSE")
ax.set_title("MSE as a Function of Fit")
ax.legend()
fig.savefig("biasvariance1.pdf")

design1 = MS([poly("x", degree=1)])
X_train1 = design1.fit_transform(train_df)
model1 = sm.OLS(train_df["y"], X_train1).fit()

design3 = MS([poly("x", degree=3)])
X_train3 = design3.fit_transform(train_df)
model3 = sm.OLS(train_df["y"], X_train3).fit()


design10 = MS([poly("x", degree=10)])
X_train10 = design10.fit_transform(train_df)
model10 = sm.OLS(train_df["y"], X_train10).fit()


fig, ax = plt.subplots()
train_df.plot.scatter(x="x", y="y", ax=ax, label="Training Data")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

x_vals = np.linspace(0,5, 100)

X_pred1 = design1.transform(pd.DataFrame({"x": x_vals}))
y_pred1 = model1.get_prediction(X_pred1)
y_pred1_ci = y_pred1.conf_int()
ax.plot(x_vals, y_pred1.predicted_mean, label="Linear Model", color='C1',)
ax.fill_between(
    x_vals,
    y_pred1_ci[:,0], # lower bound
    y_pred1_ci[:,1], # upper bound
    color='C1',
    alpha=0.25,     # transparency
    label='',
)


X_pred3 = design3.transform(pd.DataFrame({"x": x_vals}))
y_pred3 = model3.get_prediction(X_pred3)
y_pred3_ci = y_pred3.conf_int()
ax.plot(x_vals, y_pred3.predicted_mean, label="Cubic Model", color='C2',)
ax.fill_between(
    x_vals,
    y_pred3_ci[:,0], # lower bound
    y_pred3_ci[:,1], # upper bound
    color='C2',
    alpha=0.25,     # transparency
    label='',
)


X_pred10 = design10.transform(pd.DataFrame({"x": x_vals}))
y_pred10 = model10.get_prediction(X_pred10)
y_pred10_ci = y_pred10.conf_int()
ax.plot(x_vals, y_pred10.predicted_mean, label="Degree 10 Model", color='C3',)
ax.fill_between(
    x_vals,
    y_pred10_ci[:,0], # lower bound
    y_pred10_ci[:,1], # upper bound
    color='C3',
    alpha=0.25,     # transparency
    label='',
)
ax.legend()
fig.savefig("regression_exp.pdf")

