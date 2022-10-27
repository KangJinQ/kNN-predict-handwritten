import os
from PIL import Image
import numpy as np

data_root_path = 'data_set'
class_num = 2
label_list = ["1", "2"]  # 1=中 2=国


def get_data(root, class_labels):
    """
    :param root: 数据集根目录
    # :param class_num: 一共分几类
    :param class_labels: 每类的label名字，注意名字要和文件名命名相同
    :return: return_X, return_y数据和标签, X为ndarray类型的图片信息（归一化后）
    (X为一通道图像数组)
    """
    return_X = []
    return_y = []
    for i in range(len(class_labels)):
        data_path = os.path.join(root, class_labels[i])
        one_img_path_list = os.listdir(data_path)
        for j in range(len(one_img_path_list)):
            img_path = os.path.join(data_path, one_img_path_list[j])
            img = Image.open(img_path).convert('L')  # 为了简化计算，读入单通道数据（后加入的）
            img = img.resize((70, 70))  # 将输入的图像大小固定为70*70，通道数为3
            img_array = np.array(img)  # 将输入图像类型变为ndarray
            img_array_norm = img_array / 255.0  # 归一化像素值, 区间0-1
            img_vector = img_array_norm.flatten()
            return_X.append(img_vector)
            return_y.append(class_labels[i])
    return np.array(return_X), np.array(return_y)


if __name__ == '__main__':
    X, y = get_data(root=data_root_path, class_labels=label_list)
    print(X[0], '\n', y[0])
    print(X[0].shape)
