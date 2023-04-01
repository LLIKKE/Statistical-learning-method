from sklearn import datasets
import numpy as np

## Example 1
# breast cancer for classification(2 classes) X(569, 30) y(569,)
X, y = datasets.load_breast_cancer(return_X_y=True)
y = np.where(y == 0, -1, 1)  # 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y


# my perceptron
class Perceptron:
    def __init__(self):
        self.W = np.ones((len(X[0]),), dtype=float)
        self.b = 0
        self.lr = 0.01
        self.epoch = 100

    def fit(self, x, y):
        for ep in range(self.epoch):
            for i in range(len(x)):
                if y[i] * (np.dot(x[i], self.W) + self.b) <= 0:
                    self.W += self.lr * y[i] * x[i]
                    self.b += self.lr * y[i]
                print(self.score(X, y))

    def predict(self, x):
        return np.where(np.dot(x, self.W) + self.b > 0, 1, -1)

    def score(self, x, y):
        y_pred = self.predict(x)
        return 1 - np.count_nonzero(y - y_pred) / len(y)


perceptron = Perceptron()
perceptron.fit(X, y)
y_pred = perceptron.predict(X)
print(perceptron.score(X, y))
