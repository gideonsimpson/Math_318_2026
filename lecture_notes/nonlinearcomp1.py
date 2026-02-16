

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm

from ISLP.models import ModelSpec as MS

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sklearn.neighbors import KNeighborsClassifier

np.random.seed(318)
n = 10**4
x1 = np.random.uniform(-0.5, 0.5, n)
x2 = np.random.uniform(0, 1, n)

cls = 1 - (x1 < 0.25 * np.sin(4 * np.pi * x2)).astype(int)
nonlinear_df = pd.DataFrame({"x1": x1, "x2": x2, "class": pd.Categorical(cls)})

train_df = nonlinear_df.sample(frac=0.7, random_state=123)   # 70% train
test_df  = nonlinear_df.drop(train_df.index)                 # 30% test

# train and test logistic
design = MS(['x1', 'x2'], intercept=True) 
X_train= design.fit_transform(train_df)
X_test = design.transform(test_df)
model = sm.GLM(train_df['class'], X_train, family=sm.families.Binomial())
results = model.fit()

prob_ = results.predict(X_train)
logistic_pred_train = (prob_ > 0.5).astype(int)
prob_ = results.predict(X_test)
logistic_pred_test = (prob_ > 0.5).astype(int)

print('Logistic:', np.mean(logistic_pred_train != train_df['class']),  np.mean(logistic_pred_test != test_df['class']))

# LDA
design = MS(['x1', 'x2'], intercept=False) 
X_train= design.fit_transform(train_df)
X_test = design.transform(test_df)

lda = LDA()
lda.fit(X_train, train_df['class'])

pred_train_lda = lda.predict(X_train)
pred_test_lda = lda.predict(X_test)

print('LDA:', np.mean(pred_train_lda != train_df['class']), np.mean(pred_test_lda != test_df['class']))

# QDA

qda = QDA()
qda.fit(X_train, train_df['class'])

pred_train_qda = qda.predict(X_train)
pred_test_qda = qda.predict(X_test)

print('QDA:', np.mean(pred_train_qda != train_df['class']), np.mean(pred_test_qda != test_df['class']))

# KNN
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, train_df['class'])

pred_train_knn3 = knn3.predict(X_train)
pred_test_knn3 = knn3.predict(X_test)

print('KNN3:', np.mean(pred_train_knn3 != train_df['class']), np.mean(pred_test_knn3 != test_df['class']))