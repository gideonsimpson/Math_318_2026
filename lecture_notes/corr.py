import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

rng = np.random.default_rng(100)

x = np.linspace(-5, 5, 101)
xi = rng.normal(size=x.size)

eps_vals = np.array([0.1, 10, 100])

# create data 
# y = x**3 + eps * xi for all eps values, and store in data frame
eps= eps_vals[0]
y = x**3 + eps * xi
cubic_df = pd.DataFrame({'x': x, 'y': y, 'eps': eps})

for eps in eps_vals[1:]:
    y = x**3 + eps * xi
    temp_df = pd.DataFrame({'x': x, 'y': y, 'eps': eps})
    cubic_df = pd.concat([cubic_df, temp_df])

# treat epsilon value as a categorical variable
cubic_df.eps = cubic_df.eps.astype('category')


fig, ax = plt.subplots()
for (eps, group) in cubic_df.groupby('eps'):
    ax.scatter(group['x'], group['y'],alpha=0.6, label=rf"$\epsilon={eps}$")
ax.set_title(r'$y=x^3 + \epsilon \cdot \xi$')
ax.set_xlabel(r'$x$')   
ax.set_ylabel(r'$y$')
ax.legend()
fig.savefig('cubic_data_varying_noise.pdf')

# compare with uncorrelated data
x = rng.uniform(-5, 5, size=500)
y = rng.normal(size=x.size)
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.6)
ax.set_title('Uncorrelated Data')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.savefig('uncorrelated_data.pdf')