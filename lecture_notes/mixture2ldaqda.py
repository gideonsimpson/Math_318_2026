import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from ISLP.models import ModelSpec as MS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


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

df['class'] = df['class'].astype('category')

train_df = df.sample(frac=0.7, random_state=123)   # 70% train
test_df  = df.drop(train_df.index)                 # 30% test

design = MS(['x1', 'x2'], intercept=False) # intercept already handled in LDA/QDA
X_train = design.fit_transform(train_df)

X_test = design.transform(test_df)

# fit
lda = LDA()
lda.fit(X_train, train_df['class'])

qda = QDA()
qda.fit(X_train, train_df['class'])

# predict on a regular grid for visualization
x1_min, x1_max = df["x1"].min() - 1, df["x1"].max() + 1
x2_min, x2_max = df["x2"].min() - 1, df["x2"].max() + 1
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 400),
    np.linspace(x2_min, x2_max, 400),
)
grid = pd.DataFrame({"x1": xx1.ravel(), "x2": xx2.ravel()})
X_grid = design.transform(grid)

# record predictions as integer values and reshape to match the grid
pred_lda = lda.predict(X_grid).astype(int).reshape(xx1.shape)
pred_qda = qda.predict(X_grid).astype(int).reshape(xx1.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

axes[0].contourf(xx1, xx2, pred_lda, alpha=0.25, cmap="viridis")
axes[0].scatter(df["x1"], df["x2"], c=df["class"], s=10, alpha=0.6, cmap="viridis")
axes[0].set_title("LDA Decision Boundaries")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")

axes[1].contourf(xx1, xx2, pred_qda, alpha=0.25, cmap="viridis")
axes[1].scatter(df["x1"], df["x2"], c=df["class"], s=10, alpha=0.6, cmap="viridis")
axes[1].set_title("QDA Decision Boundaries")
axes[1].set_xlabel(r"$x_1$")
axes[1].set_ylabel(r"$x_2$")
fig.savefig("lda_qda_decision_boundaries.pdf")