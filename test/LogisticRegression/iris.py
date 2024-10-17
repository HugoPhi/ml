import numpy as np
from hym.LogisticRegression import BiClassfier, Metrics
import sklearn.datasets

# sd = np.random.randint(0, 1000)
# print(f'sd is: {sd}')
# np.random.seed(sd)

# np.random.seed(217)  # ->  40
# num_test = 40  # test set number

np.random.seed(648)  # ->  50
num_test = 50  # test set number


iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target
print(f'shape of X: {X.shape}, shape of y: {y.shape}')
ix = np.random.permutation(X.shape[0])
X = X[ix]
y = y[ix]

X_train = X[:-num_test]
y_train = y[:-num_test]
X_test = X[-num_test:]
y_test = y[-num_test:]

# train
# 0 classifier
y0 = (y_train == 0).astype(int)
clr1 = BiClassfier(X_train, y0, lr=0.01, epoch=300)
clr1.fit()  # train

# train the classifier for judge versicolor and virginica
indices = y_train != 0  # remove all 0s
X_train = X_train[indices]
y_train = y_train[indices]
y1 = (y_train == 1).astype(int)
clr2 = BiClassfier(X_train, y1, lr=0.01, epoch=500)
clr2.fit()  # train

# test
y0_test = (y_test == 0).astype(int)
y0_pred = clr1(X_test, threshold=0.5)
y0_test = y0_test.reshape(-1, 1)
filter_index = y0_pred.flatten() == 0  # index of pred is not 0
acc1 = (y0_test == y0_pred).mean()
print(f'acc for clr1 on test is: {acc1}')

y1_test = (y_test[filter_index] == 1).astype(int)
y1_pred = clr2(X_test[filter_index], threshold=0.5)
y1_test = y1_test.reshape(-1, 1)
acc2 = (y1_pred == y1_test).mean()
print(f'acc for clr2 on test is: {acc2}')


# final test function
pred = lambda x: 0 if clr1(x) == 1 else (1 if clr2(x) == 1 else 2)
L = []
y_pred = []
for xx, yy in zip(X_test, y_test):
    L.append((yy == pred(xx)).astype(int))
    y_pred.append(pred(xx))

y_pred = np.array(y_pred)
L = np.array(L)
print(f'acc on test: {L.mean()}')
mtr = Metrics(y=y_test, y_pred=y_pred, classes=3)
print(mtr)

L = []
for _ in range(100000):
    ix = np.random.randint(0, 150)
    xx = X[ix]
    xx = xx + np.random.normal(0, 0.01, xx.shape)
    yy = y[ix]

    L.append((yy == pred(xx)).astype(int))

L = np.array(L)
print(f'acc on whole dataset: {L.mean()}')
