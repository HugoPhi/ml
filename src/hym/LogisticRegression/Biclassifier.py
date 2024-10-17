import numpy as np


class BiClassfier:

    def __init__(self, X_train, y_train, lr=1e-2, epoch=100):  # init model
        self.w = np.random.rand(1, X_train.shape[1])
        self.b = np.random.rand()
        self.X = X_train
        self.y = y_train
        self.lr = lr
        self.epoch = epoch

    def fit(self):  # train model
        self.hist = []

        # functions to be used
        loss = lambda y_pred, y: -np.mean((y * np.log(y_pred + 1e-12) + (1 - y) * np.log(1 - y_pred + 1e-12)))
        grad = lambda y_pred, y: y - y_pred
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        # training
        for _ in range(self.epoch):
            y_pred = sigmoid(self.X @ self.w.T + self.b)

            record_loss = loss(y_pred, self.y)
            self.hist.append(record_loss)

            gd = grad(y_pred, self.y.reshape(-1, 1))
            dw = np.mean(gd * self.X, axis=0)
            db = np.mean(gd, axis=0)

            self.w += self.lr * dw
            self.b += self.lr * db

    def __call__(self, x, threshold=0.5):
        # prediction, return precise classification
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        y_pred = sigmoid(x @ self.w.T + self.b)
        y_pred = (y_pred >= threshold).astype(int)

        return y_pred

    def history(self):  # history of loss
        return self.hist
