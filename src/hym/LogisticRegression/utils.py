import numpy as np


class Metrics:

    def __init__(self, y, y_pred, classes=3):  # y \ y_pred
        matrix = np.zeros((classes, classes))
        for i, j in zip(y, y_pred):
            matrix[i, j] += 1
        self.matrix = matrix

    def precision(self):
        precision = np.diag(self.matrix) / self.matrix.sum(axis=0)
        return precision

    def recall(self):
        recall = np.diag(self.matrix) / self.matrix.sum(axis=1)
        return recall

    def f1(self):
        f1 = 2 * self.precision() * self.recall() / (self.precision() + self.recall())
        return f1

    def accuracy(self):
        accuracy = np.diag(self.matrix).sum() / self.matrix.sum()
        return accuracy

    def macro_avg(self):
        return np.mean(self.f1())

    def micro_avg(self):
        return np.diag(self.matrix).sum() / self.matrix.sum()

    def confusion_matrix(self):
        return self.matrix

    def __repr__(self) -> str:
        table = ' ' * 6
        print(f'        {table}Precision{table}Recall{table}  F1')
        for i in range(len(self.precision())):
            print(f'Class {i} {table}{self.precision()[i]:.6f} {table}{self.recall()[i]:.6f}{table}{self.f1()[i]:.6f}')
        print()
        print(f'Accuracy      {self.accuracy():.6f}')
        print(f'Macro avg     {self.macro_avg():.6f}')
        print(f'Micro avg     {self.micro_avg():.6f}')

        return ''
