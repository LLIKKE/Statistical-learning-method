import numpy as np
from dataloader import data_loader


class KNearestNeighbor:
    def __init__(self, k, classes,mode=''):
        '''
        :param k: k值
        :param classes: 分类数
        '''
        self.classes = classes
        self.K = k
        self.X_train = None
        self.Y_train = None

    def __call__(self, X, Y, x, k=None):
        '''
        :param X: 全部训练数据集输入
        :param Y: 全部训练数据集标签
        :param x: 输入待分类的数据集
        :param k: 是否更新k值，若不则使用前面初始化时的k值
        :return:
        '''
        if k is None:
            k = self.K
        self.X_train = X
        self.Y_train = Y
        # 计算距离矩阵
        Dis = self.distance(self.X_train, x)
        class_list = []
        # 选择K个最近的数据点
        for i in range(k):
            class_list.append(np.argmax(-Dis))
            Dis[np.argmax(-Dis)] = np.inf
        # 选择这k个数据点中出现频率最高的点
        classes = [0 for i in range(self.classes)]
        for i in class_list:
            if self.Y_train[i] == 0:
                classes[0] += 1
            elif self.Y_train[i] == 1:
                classes[1] += 1
            elif self.Y_train[i] == 2:
                classes[2] += 1
        return np.argmax(classes)   # 返回分类结果

    def distance(self, x1, x2, mode='l1'):
        if mode == 'l1':
            return np.linalg.norm(x1 - x2, ord=1, axis=1)
        elif mode == 'l2':
            return np.linalg.norm(x1 - x2, ord=2, axis=1)
        elif mode == '':
            pass

    def acc(self, pre, y):
        return np.mean(np.array(pre) == y)


if __name__ == '__main__':
    Data_loader = data_loader(huafen=[5, 5], normalize=True)
    train_x, train_y = next(Data_loader)
    KNN = KNearestNeighbor(k=6, classes=3)
    pre = []
    for i in range(len(Data_loader.test_x)):
        t = Data_loader.test_x[i]
        p = Data_loader.test_y[i]
        pred = KNN(train_x, train_y, t)
        pre.append(pred)
        print('-----------------')
        print('predict:', pred)
        print('target:', p)
    print('准确率acc:', KNN.acc(pre, Data_loader.test_y))
