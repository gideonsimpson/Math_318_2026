from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load iris data and manually add headers
iris = pd.read_csv('../data/iris.data', header=None, 
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
# set species to categorical type for better handling
iris['species'] = iris['species'].astype('category')

# set up PCA pipeline with standard scaling and 2 principal components
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
iris_pca_scaled = pca_pipe.fit_transform(iris.drop(columns='species'))

# visualize results
fig, axes = plt.subplots()
for species, color in zip(iris['species'].cat.categories, ['C0', 'C1', 'C2']):
    species_idx = iris['species'] == species
    axes.scatter(
        iris_pca_scaled[species_idx, 0], iris_pca_scaled[species_idx, 1],
        s=10, alpha=0.5, label=species, color=color
    )

# Add loading vectors for the original features in PC space
loadings = pca_pipe[1].components_.T
feature_names = iris.drop(columns='species').columns
score_scale = np.max(np.abs(iris_pca_scaled[:, :2]), axis=0)
arrow_scale = 0.8

for i, feature in enumerate(feature_names):
    vec = loadings[i, :] * score_scale * arrow_scale
    axes.arrow(
        0, 0, vec[0], vec[1],
        color='black', width=0.015, alpha=0.7,
        head_width=0.12, head_length=0.16, length_includes_head=True
    )
    axes.text(vec[0] * 1.1, vec[1] * 1.1, feature, color='black', fontsize=9)

axes.set_xlabel("PC1")
axes.set_ylabel("PC2")
axes.set_title("PCA Projection of Iris Data with Loading Vectors")  
axes.legend()
fig.savefig('pcairis.pdf')