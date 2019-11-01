import numpy as np
import os
import cv2  # 我用的opencv_python-4.0.0-cp36-cp36m-win_amd64.whl
import re


SIZE = 224


def get_class(dir):
    """
    :param dir: 图片存放路径
    :return: 类别列表
    """
    l = []
    for i in os.listdir(dir):
        l.append(i)
    return l


def get_img_dirs(dir):
    """
    :param dir: 图片存放路径
    :return: 所有图片路径列表
    """

    l = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                l.append(root + '/' + file)
    return l


def get_own_img_data(imgs, cla, batch_size):
    """
    为了实现更快速，随机，本来想存txt里的，感觉又有点麻烦，读取也不方便
    索性就设置随机图片制作数据组和列表
    :param imgs: 所有图片目录列表
    :param cla: 所有类别列表
    :param batch_size: 一组的大小
    :return: img，label
    """

    batch = []
    for i in range(batch_size):
        rng = np.random.randint(0, len(imgs))
        batch.append(imgs[rng])

    l = []
    for img in batch:
        im = cv2.imread(img)
        a = np.reshape(im, (SIZE, SIZE, 3))
        l.append(a * (1. / 255))

    label = []
    for img in batch:
        a = [0 for i in range(len(cla))]
        index = cla.index(re.split('/', img)[-2])
        a[index] = 1
        label.append(a)
    return l, label


def main(batch_size, dir):
    cla = get_class(dir)
    imgs = get_img_dirs(dir)
    data = get_own_img_data(imgs, cla, batch_size)
    return data



