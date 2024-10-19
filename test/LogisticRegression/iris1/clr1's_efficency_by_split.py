import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from hym.LogisticRegression import BiClassfier

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

np.random.seed(64)
fig, ax = plt.subplots(3, 3, figsize=(10, 5))
for i, test in enumerate(range(30, 120, 10)):
    A = []
    for _ in range(1000):
        ix = np.random.permutation(X.shape[0])
        X = X[ix]
        y = y[ix]
        num_test = test
        X_train = X[:-num_test]
        y_train = y[:-num_test]
        X_test = X[-num_test:]
        y_test = y[-num_test:]
        y0 = (y_train == 0).astype(int)
        clr1 = BiClassfier(X_train, y0, lr=0.01, epoch=300)
        clr1.fit()
        y0_test = (y_test == 0).astype(int)
        y0_pred = clr1(X_test, threshold=0.5)
        y0_test = y0_test.reshape(-1, 1)
        acc = (y0_pred == y0_test).mean()
        A.append(acc)

    ax[i // 3, i % 3].hist(A, bins=10, range=(0.4, 1.0), edgecolor='black')
    ax[i // 3, i % 3].set_title(f'test percentage = {test / X.shape[0]:.2f}, unfairness_of_train = {(X.shape[0] - test) / test:.2f}')
    ax[i // 3, i % 3].tick_params(axis='both', which='both', direction='in')

plt.tight_layout()
plt.savefig('./clr1\'s_efficency_by_split.png over 1000 times.png')
plt.show()
