import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import operator
import os
from collections import Counter

"""提取已有标签的数据分为x, y"""
"""没有练习的必要了"""
def exact_data(filename):
    with open(filename, 'r') as f:
        all_data = f.readlines()
        line_num = len(all_data)
        x_num = len(all_data[0].strip().split(',')) - 1

        x_data = np.zeros((line_num, x_num))
        y_data = np.zeros(line_num)
        for i in range(line_num):
            line = all_data[i].strip().split(',')
            x_data[i] = line[:x_num]
            y_data[i] = int(line[-1])
        return x_data, y_data

"""将x归一化,归一化有很多方法，本质上都是将数据集的范围纳入-1， 1之间。"""
def normalize_data(data_set):
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val

    line_num = data_set.shape[0]
    norm_data = data_set - np.tile(min_val, (line_num, 1))
    norm_data = norm_data / np.tile(ranges, (line_num, 1))
    return norm_data, ranges, min_val

"""对一个数据进行分类, 计算test与dataset数据之间的距离（通常是欧式距离）， 
选择前k个最小的距离，将这k个中出现最多次的标签作为预测值"""
def classfiy(test_data, data_set, labels, k=3):
    test_data = np.array(test_data, dtype='float64')
    line_num = data_set.shape[0]

    diff = (np.tile(test_data, (line_num, 1)) - data_set) ** 2
    distance = (diff.sum(axis=1)) ** 0.5
    idx_dis_sort = np.argsort(distance)

    classCount = {}
    for i in range(k):
        vote_label = labels[idx_dis_sort[i]]
        classCount[vote_label] = classCount.get(vote_label, 0) + 1
    max_count = max(classCount, key=classCount.get)
    return max_count

"""对所有测试数据进行训练，测试数据来源于整体的数据， 使用ratio=0.1来区分test, train.
并且返回分类错误的个数。预测标签和实际标签不匹配的个数。"""
def data_test(ratio=0.1, k=3):
    train_data, train_label = exact_data("E:\ml\happiness_train_complete_.csv")
    norm_train_data, ranges, min_val = normalize_data(train_data)

    line_num = norm_train_data.shape[0]
    test_num = int(line_num * ratio)
    print('number of test_data:', test_num)
    error_count = 0
    for i in range(test_num):
        classifierResult = classfiy(norm_train_data[i, :], norm_train_data[test_num: line_num, :],
                                    train_label[test_num:line_num], k)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, train_label[i]))
        if classifierResult != train_label[i]:
            error_count += 1
    print('the total error rate is: %.2f' % (error_count / test_num))
    print(error_count)


data_test(0.2, 20)