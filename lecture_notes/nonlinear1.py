

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

np.random.seed(318)
n = 10**4
x1 = np.random.uniform(-0.5, 0.5, n)
x2 = np.random.uniform(0, 1, n)

cls = 1 - (x1 < 0.25 * np.sin(4 * np.pi * x2)).astype(int)
nonlinear_df = pd.DataFrame({"x1": x1, "x2": x2, "class": pd.Categorical(cls)})

fig, ax = plt.subplots()
for k, g in nonlinear_df.groupby("class"):
    g.plot.scatter("x1", "x2", s=10, alpha=0.5, label=f"Class {k}", c='class',cmap="viridis", colorbar=False, ax=ax)
ax.set_title("Nonlinear Decision Boundary")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.legend()

fig.savefig("nonlinear_boundary.pdf")