import numpy as np

"""
输入：文件名
输出：X和Y
"""
def exact_data(filename):
    with open(filename, 'r', encoding='UTF-8') as file:
        all_data = file.readlines()

        x_num = len(all_data[0].strip().split(',')) - 1
        line_num = len(all_data) - 1

        data_set = []
        labels = []
        for i in range(1, line_num+1):
            line_new = all_data[i].strip().split(',')
            data_set.append(line_new[:x_num])
            labels.append(line_new[-1])
    return data_set, labels

"""
输入：标签
输出：Y的先验概率的列表，按照搜索列表时的顺序排列。值是小数。
"""
def cal_Y(labels):
    all_num = len(labels)
    y_prob = []
    Y_count = {}

    for i in range(all_num):
        if Y_count.__contains__(labels[i]) is not True:
            Y_count[labels[i]] = 0
        Y_count[labels[i]] += 1
    for k in Y_count.values():
        y_prob.append(float(k / all_num))
    return y_prob

"""
输入：XY数据，测试集（X）不包含标签;单个标签。
输出：对应于单个标签的所有X的条件概率的对数值。
"""
def cal_cond(data_set, labels, test_data, label):
    x_num = len(data_set[0])
    cond_prob = 0.
    for i in range(x_num):
        x_count = 0
        x_data = []
        for k in range(len(labels)):
            if label == labels[k]:
                x_data.append(data_set[k][i])

        for j in x_data:
            if j == test_data[i]:
                x_count += 1
        cond_prob += np.log2(float(x_count / len(x_data)))
    return cond_prob

"""
输入：XY，测试集。
输出：贝叶斯概率的最大值所对应的标签，也就是分类的结果。
"""
def trainNB(data_set, labels, test_data):
    set_labels = set(labels)
    list_labels = list(set_labels)
    prior_prob = cal_Y(labels)

    postrior_prob = []
    for i in range(len(set_labels)):
        cond_prob = cal_cond(data_set, labels, test_data, list_labels[i])
        postrior_prob.append(cond_prob + np.log2(prior_prob[i]))
    max_prob = max(postrior_prob)
    index_prob = postrior_prob.index(max_prob)
    return list_labels[index_prob]


a, b = exact_data('navie_bayes.txt')
c = cal_Y(b)
e = trainNB(a, b, ['2', 'S'])
print(e)
