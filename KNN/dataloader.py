import numpy as np
import csv
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import datasets
class data_loader:
    def __init__(self, batch_size=256, data='iris', normalize=False, shuffle=False, huafen=None):
        '''

        :param batch_size: 批次大小
        :param data: 选择数据集
        :param normalize: 数据是否进行归一化处理
        :param shuffle: 是否打乱数据顺序
        :param huafen: 数据集比例划分[a,b],a+b=10,a为训练集比例，b为验证集比例
        '''
        if huafen is None:
            huafen = [8, 2]
        self.classes = None
        self.features = None
        self.start = 0

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
        # 数据归一化，正太归一化
        if normalize:
            ave = np.mean(self.data_x, axis=1)
            st = np.std(self.data_x, axis=1)
            self.data_x = ((self.data_x.T - ave) / (st + 1e-6)).T
        self.data_y = np.array(self.data_y, dtype=float)

        # 打乱列表顺序，制备一个乱序序列表，同时进行数据集分割
        shuffle_list = [i for i in range(self.data_x.shape[0])]
        if shuffle:
            random.shuffle(shuffle_list)
        list1 = shuffle_list[:int(self.data_x.shape[0] * huafen[0] * 0.1)]
        list2 = shuffle_list[int(self.data_x.shape[0] * huafen[0] * 0.1):]
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        for i in list1:
            self.train_x.append(self.data_x[i])
            self.train_y.append(self.data_y[i])
        for i in list2:
            self.test_x.append(self.data_x[i])
            self.test_y.append(self.data_y[i])
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        self.test_x = np.array(self.test_x)
        self.test_y = np.array(self.test_y)
        # 数据长度标记
        self.end = self.train_x.shape[0]

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
                label = 0.
            elif rows[i][5] == 'versicolor':
                label = 1.
            elif rows[i][5] == 'virginica':
                label = 2.
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

    def len(self):
        return self.data_x.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        # 数据迭代器，根据batchsize进行提取
        start = self.start
        self.start += self.batch_size
        if self.start > self.end:
            self.start = 0
            return self.train_x[start:self.end], \
                   self.train_y[start:start + self.end]
        return self.train_x[start:start + self.batch_size], \
               self.train_y[start:start + self.batch_size]