import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

rng = np.random.default_rng(100)
n = 100 # size of each sample set
m = 10**4 # number of independent sample sets

# generate data
sample_means = np.zeros(m)
for i in range(m):
    x = rng.exponential(size=n)
    sample_means[i] = x.mean()

# generate one sample for plotting
x = rng.exponential(size=n)
fig, ax = plt.subplots(1,2, figsize=(12, 5))
# plot
ax[0].hist(x, density=True)
ax[1].hist(sample_means, bins=30, density=True)

# annotate
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel('Density')
ax[0].set_title(f'Exponential Data, Sample Size: $n = {n}$')
ax[1].set_xlabel('Sample Means')
ax[1].set_ylabel('Density')
ax[1].set_title(f'Independent Sample Means: $m = {m}$')

fig.savefig('sample_means_exponential.pdf')
