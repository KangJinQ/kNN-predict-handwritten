from dataset import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 此文件使用sklearn中写好的函数KNeighborsClassifier。
# 手写版本在kNN.py中实现。
dataset_path = './data_set'
class_labels = ['1', '2']
X, y = get_data(dataset_path, class_labels)  # 数据集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2)
print('X_train:{},X_test:{}'.format(X_train.shape, X_valid.shape))
clf = KNeighborsClassifier(n_neighbors=15)  # knn分类器实例化, 只能训练array.dim <= 2的，所以训练之前要把图片拉成向量！！！！！！！
# kNN邻居数量设置为15效果不错
clf.fit(X_train, y_train)  # 模型训练
print('测试集评估：{:.2f}'.format(clf.score(X_valid, y_valid)))
print('训练集评估：{:.2f}'.format(clf.score(X_train, y_train)))
