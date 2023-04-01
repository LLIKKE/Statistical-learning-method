import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import csv

from sklearn import datasets

from Func import seed_up

seed_up()
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class data_loader:
    def __init__(self, batch_size=256, data='iris', normalize=False, shulffe=False):

        self.classes = None
        self.features = None
        self.start = 0
        if shulffe:
            random.shuffle(list(range(len(x))))
        self.batch_size = batch_size
        if data == 'iris':
            self.path = 'E:\DATASET\iris\iris.csv'
            self.data_x, self.data_y = self.read_csv_iris()
        elif data == 'heart':
            self.path = 'E:\DATASET\Heart Disease Dataset\heart.csv'
            self.data_x, self.data_y = self.read_csv_heart()

        elif data == 'breast':
            self.breast_cancer()
        self.data_x = np.array(self.data_x, dtype=float)
        if normalize:
            ave = np.mean(self.data_x, axis=1)
            st = np.std(self.data_x, axis=1)
            self.data_x = ((self.data_x.T - ave) / (st + 1e-6)).T
        self.data_y = np.array(self.data_y, dtype=float)
        self.end = len(self.data_y)

    def read_csv_heart(self):
        with open(self.path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        self.classes = ['-1', '1']
        self.features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                         'slope', 'ca', 'thal']

        rows = rows[1:]
        random.shuffle(rows)
        train_data_list = []

        for i in range(len(rows)):
            train_data = []
            for j in range(len(self.features)):
                train_data.append(float(rows[i][j]))
            if rows[i][13] == '0':
                label = -1
            elif rows[i][13] == '1':
                label = 1
            else:
                raise RuntimeError('data_labeling_Error')
            train_data_list.append([train_data, label])
        data_x = [i[0] for i in train_data_list]
        data_y = [i[1] for i in train_data_list]
        return data_x, data_y

    def read_csv_iris(self):
        with open(self.path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        self.classes = ['setosa', 'versicolor', 'virginica']
        self.features = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
        rows = rows[1:]
        random.shuffle(rows)
        train_data_list = []

        for i in range(len(rows)):
            train_data = []
            for j in range(1, 5):
                train_data.append(float(rows[i][j]))
            if rows[i][5] == 'setosa':
                label = 0
            elif rows[i][5] == 'versicolor':
                label = 1
            elif rows[i][5] == 'virginica':
                label = 2
            else:
                raise RuntimeError('data_labeling_Error')
            train_data_list.append([train_data, label])
        data_x = [i[0] for i in train_data_list]
        data_y = [i[1] for i in train_data_list]
        return data_x, data_y

    def breast_cancer(self):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        y = np.where(y == 0, -1, 1)  # 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
        self.data_x = X
        self.data_y = y

    def __iter__(self):
        return self

    def __next__(self):
        start = self.start
        self.start += self.batch_size
        if self.start > self.end:
            self.start = 0
            return self.data_x[start:self.end], \
                   self.data_y[start:start + self.end]
        return self.data_x[start:start + self.batch_size], \
               self.data_y[start:start + self.batch_size]


class model(data_loader):
    def __init__(self, feature, lr=0.001):
        '''
        :param feature: 特征数
        :param lr: 学习率
        '''
        super().__init__()
        self.feature = feature
        self.w = np.array([0]*feature)  # 初始化参数矩阵w = [w1,w2,w3...,wfeature]
        self.b = 0  # 初始化偏执参数b标量
        self.lr = lr
        self.input = None  # 用self变量，程序中多方调用
        self.target = None
        self.pred = None

    def loss_function(self, target):

        self.target = target
        return np.sum(target * self.pred)

    def acc(self, data_x,target):
        '''

        '''

        pred = np.dot(data_x, self.w) + self.b
        pred[pred >= 0] = 1
        pred[pred < 0] = -1

        acc = np.sum(pred == target) / pred.size
        return acc

    def fit(self):
        '''
        计算参数偏导，根据梯度优化
        :return:
        '''

        grad_w = self.input.T * self.target  # 输入[[x1,x3,x5],乘[[y1,y2,y3]]   得 [[x1y1,x3y2,x5y3],  求sum得[x1y1+x3y2+x5y3,x2y1+x4y2+x5y3]
        grad_w_sum = np.sum(grad_w, axis=1)  # 转置 [x2,x4,x5]]                    [x2y1,x4y2,x5y3]]
        grad_b_sum = np.sum(self.target)
        self.w = self.w - self.lr * (1 / grad_w.size) * grad_w_sum  # self.w = [w1,w2]-lr*[grad_sum_w1,grad_sum.w2]
        self.b = self.b - self.lr * (1 / self.target.size) * grad_b_sum

    def __call__(self, x):

        self.input = x
        # pred = sigmoid(np.dot(x, self.w) + self.b)
        pred = np.dot(x, self.w) + self.b
        self.pred = pred
        return self.pred


if __name__ == '__main__':
    para = dict()
    para['iters'] = 100
    para['batch_size'] = 300

    Data_load = data_loader(data='breast', batch_size=para['batch_size'])
    all_data_x = Data_load.data_x
    all_data_y = Data_load.data_y
    Model = model(feature=len(Data_load.data_x[0]),lr=0.1)
    for i in range(para['iters']):
        x, y = next(Data_load)
        pred = Model(x)
        loss = Model.loss_function(y)
        print('-----------')
        # print(loss)
        print(Model.acc(Data_load.data_x,Data_load.data_y))
        # print(y)
        # print(Model.pred)
        #print(Model.w,Model.b)
        Model.fit()

