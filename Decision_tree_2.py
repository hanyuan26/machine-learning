import numpy as np
from math import log


def exact_data(filename):
    with open(filename, 'r', encoding='UTF-8') as file:
        all_data = file.readlines()
        line_num = len(all_data) - 1
        feature0 = all_data[0].strip().split(',')
        x_num = len(feature0)
        feature = feature0[1:x_num - 1]

        data_set = [[0] * x_num] * line_num
        for i in range(1, line_num + 1):
            line = all_data[i].strip().split(',')
            data_set[i - 1] = line[1: x_num]
        return data_set, feature


"""需要重复利用Di来计算熵，所以要获取相应于Ai的Di, 
输入：index是第几列的也就是第几个特征。value是Ai的值。
输出：就是移除了Ai的data_set，且Ai的值是value"""
def get_ADi(data_set, index, value):
    line_num = len(data_set)
    ADi = []
    for i in range(line_num):
        if data_set[i][index] == value:
            line0 = data_set[i][: index]
            line0.extend(data_set[i][index + 1:])
            ADi.append(line0)
    return ADi


def get_Di(data_set, index, value):
    line_num = len(data_set)
    Di = []
    for i in range(line_num):
        if data_set[i][index] == value:
            Di.append(data_set[-1])
    return Di


"""计算香农熵"""


def channon_entropy(data_set):
    line_num = len(data_set)
    label_count = {}
    for line in data_set:
        label = line[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1

    channon_ent = 0.
    for key in label_count:
        prob = float(label_count[key] / line_num)
        channon_ent -= prob * log(prob, 2)
    return channon_ent


"""输入所有x_data,标签D，输出每个特征的信息增益，并返回按照从大到小排列的信息增益索引表"""


def info_gain(data_set):
    x_num = len(data_set[0]) - 1
    line_num = len(data_set)
    H_D = channon_entropy(data_set)

    info_gain_list = []
    for i in range(x_num):
        A_i = [A[i] for A in data_set]
        A_val = set(A_i)
        cond_entropy = 0.
        for ai in A_val:
            Di = get_ADi(data_set, i, ai)
            prob = float(len(Di) / line_num)
            cond_entropy += prob * channon_entropy(Di)
        info_gain_list.append(H_D - cond_entropy)
    index_gain = list(np.argsort(info_gain_list))
    max_A = index_gain.index(max(index_gain))
    return max_A


"""选择出现次数最多的一个结果。"""


def max_(labeli):
    label_count = {}
    for label in labeli:
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    max_count = max(label_count, key=label_count.get)
    return max_count


"""树的创建过程也太妙了吧"""


def create_tree(data_set, feature):
    labels = [label[-1] for label in data_set]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if len(data_set[0]) == 1:
        return max_(data_set)

    Ai_gain_index = info_gain(data_set)
    Ai_label = feature[Ai_gain_index]
    my_tree = {Ai_label: {}}
    del feature[Ai_gain_index]

    featValues = [example[Ai_gain_index] for example in data_set]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sub_feature = feature[:]
        my_tree[Ai_label][value] = create_tree(get_ADi(data_set, Ai_gain_index, value), sub_feature)
    return my_tree


def classify(inputTree, featlabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]

    featIndex = featlabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featlabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


data_set, feature = exact_data('decision_tree.txt')
my_tree = create_tree(data_set, feature)
data_set0, feature0 = exact_data('decision_tree.txt')
class_reasult = classify(my_tree, feature0, ['中年', '是', '否', '一般'])
