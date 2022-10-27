from sklearn.model_selection import train_test_split
import numpy as np
from dataset import get_data


def classify_knn(input_x, X, y, k):
    """
    :param input_x: 输入待分类图片一张
    :param X: 训练集图片(array)
    :param y: 训练集标签(array)
    :param k: 用于选择最近邻居的数目
    :return: 返回input_x的标签
    """
    X_size = X.shape[0]
    # 1 计算input_x和其他训练集元素的距离，这里使用L2范数计算向量距离
    # 将input_x复制，复制成大小为(X_size, 1)，再减去训练集X得到每个向量对应点的差值
    diffMat = np.tile(input_x, (X_size, 1)) - X
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)  # 横向加和
    distance = sqDistance ** 0.5
    # 2 将算得的距离递增排序，使用argsort()函数返回排序后的index索引号
    sortedDistIndices = distance.argsort()
    # 3 选取当前距离最小的k个点
    # 通过记录前k个元素的分类标签数量的方式。classCount字典key=标签名，value=出现次数
    classCount = {}
    for i in range(k):
        label = y[sortedDistIndices[i]]
        if label in classCount.keys():
            classCount[label] += 1
        else:
            classCount[label] = 1
    # 给字典按照value排序（从大到小）
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    data_root_path = 'data_set'
    class_num = 2
    label_list = ["1", "2"]  # 1=中 2=国
    X, y = get_data(data_root_path, label_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print('训练数据集大小为: ', X_train.shape, '测试数据集大小为: ', X_test.shape)
    acc = 0
    for i in range(X_test.shape[0]):
        predict = classify_knn(X_test[i], X_train, y_train, k=20)
        if predict == y_test[i]:
            print('正确类别为: ', y_test[i], '测试为: ', predict, '√')
            acc += 1
        else:
            print('正确类别为: ', y_test[i], '测试为: ', predict, '×')
    print('准确率为: ', float(acc) / y_test.shape[0])
