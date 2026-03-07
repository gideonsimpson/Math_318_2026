from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

np.random.seed(123)

n = 10**3
weights = np.array([0.4, 0.6])
# x = np.zeros((n, 2))
mean = np.array([2.0, -1.0])
cov = np.array([[0.1, .4], [.4, 4.0]])
x=np.random.multivariate_normal(mean, cov, size=n)
cluster = pd.DataFrame(x, columns=['x1', 'x2'])


pca = PCA(n_components=2)
pca.fit(cluster)

fig, axes = plt.subplots()
cluster.plot.scatter("x1", "x2", s=10, alpha=0.5, ax=axes)

# Draw principal directions from the center of the cloud.
center = pca.mean_
scales = 2 * np.sqrt(pca.explained_variance_)

for i, (direction, scale) in enumerate(zip(pca.components_, scales), start=1):
    vec = direction * scale
    axes.arrow(
        center[0], center[1], vec[0], vec[1],
        color=f"C{i}", width=0.03, length_includes_head=True,
        head_width=0.25, head_length=0.35
    )
    axes.text(center[0] + vec[0], center[1] + vec[1], f"PC{i}", color=f"C{i}")

axes.set_xlabel(r"$x_1$")
axes.set_ylabel(r"$x_2$")
axes.set_xlim([-8, 8])
axes.set_ylim([-8, 8])
axes.set_title("Cluster Data with PCA Component Directions")
fig.savefig("pca_component_directions.pdf")