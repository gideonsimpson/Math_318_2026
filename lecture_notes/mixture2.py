import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


np.random.seed(123)

n = 10**4
weights = np.array([0.25, 0.5, 0.25])

means = np.array([
    [-1.0,  -2.0],
    [ -1.0,  1.0],
    [ 2.0, 2.0]
])

covs = np.array([
    [[1.0,  0.3],
     [0.3,  1.0]],
    [[1.0,  0.0],
     [0.0,  1.0]],
    [[1.0,  0.0],
     [0.0,  4.0]]
])

# Sample classes
z = np.random.choice(3, size=n, p=weights)

# Sample positions
x = np.zeros((n, 2))
for i in range(n):
    x[i,:] =np.random.multivariate_normal(means[z[i]], covs[z[i]])

# store in a data frame
df = pd.DataFrame(x, columns=['x1', 'x2'])
df['class'] = z

fig, axes = plt.subplots()
for k, g in df.groupby("class"):
    g.plot.scatter("x1", "x2", s=10, alpha=0.5, label=f"Class {k}", color=f"C{k}", ax=axes)
axes.set_title("3-Component 2D Gaussian Mixture")
axes.set_xlabel(r"$x_1$")
axes.set_ylabel(r"$x_2$")
axes.legend()
fig.savefig("2dmixture.pdf")
