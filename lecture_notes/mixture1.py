import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

rng = np.random.default_rng(318)

# Mixture model parameters
n = 1000
weights = np.array([0.3, 0.7])          # mixing probabilities (sum to 1)
means   = np.array([-2.0,  1.0])          # component means
sds     = np.array([ 2,  1])          # component standard deviations

# 1) sample component labels z in {0,1}
z = rng.choice(len(weights), size=n, p=weights)

# 2) sample x | z ~ Normal(mean[z], sd[z])
x = rng.normal(loc=means[z], scale=sds[z], size=n)

df = pd.DataFrame({"x": x, "z": z})
df.z = df.z.astype('category')

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

df.hist('x', bins=20, density=True, alpha=0.5, ax=ax[0])
ax[0].set_title('')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylim(0, 0.5)

for (z_val, group) in df.groupby('z'):
    group.hist('x', bins=10, density=True, alpha=0.5, ax=ax[1], label=f'z={z_val}')
ax[1].legend()
ax[1].set_title('')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylim(0, 0.5)

fig.savefig('mixture_model.pdf', bbox_inches='tight')