# 实现高斯分布的朴素贝叶斯

import numpy as np
import pandas as pd
import random

"""
函数说明：读取鸢尾花数据集
参数说明：无
返回：DF格式的数据集
"""
def load_data():
    iris_data = pd.read_csv('./datasets/iris/iris.data', header=None)
    iris_data = pd.DataFrame(iris_data)
    return iris_data


"""
函数功能：随机将数据集切分为训练集和测试集
参数说明：
    dataset:输入的数据集
    rate:训练集所占的比例
返回：切分好的数据集和测试集
"""
def random_split_data(dataset, rate):
    index = list(dataset.index)  # 提取出索引
    random.shuffle(index)  # 随机打乱索引
    dataset.index = index  # 将打乱后的索引值重新赋值给原数据集
    dataset_num = dataset.shape[0]  # 数据集的总数量
    train_num = int(dataset_num * rate)  # 所需训练集的数量
    train_data = dataset.loc[range(train_num), :]  # 提取前train_num个记录作为训练集
    test_data = dataset.loc[range(train_num, dataset_num), :]  # 剩下的作为测试集
    dataset.index = range(dataset.shape[0])  # 更新原数据集的索引
    # train_data.index = range(train_data.shape[0]) # 更新训练集的索引
    test_data.index = range(test_data.shape[0])  # 更新测试数据集的索引
    return train_data, test_data

"""
函数功能：运行高斯朴素贝叶斯分类并输出准确率
参数说明：
    train_data:训练数据集
    test_data:测试数据集
返回：无
"""
def gauss_nb_classify(train_data, test_data):
    labels = train_data.iloc[:, -1].value_counts().index  # 提取训练集的标签种类
    mean_list = []  # 存放各个类别的均值
    std_list = []  # 存放各个类别的方差
    result_list = []  # 存放测试机的预测结果
    for i in labels:  # 求每一类的均值和方差
        item = train_data.loc[train_data.iloc[:, -1]==i, :]  # 分别提取出每一种类别
        mean = item.iloc[:, :-1].mean()  # i类的坐标的均值向量
        std = np.sum((item.iloc[:, :-1] - mean)**2) / (item.shape[0])  # i类的坐标的方差
        mean_list.append(mean)
        std_list.append(std)
    mean_df = pd.DataFrame(mean_list, index=labels)  # 变成DF格式，索引为类标签
    std_df = pd.DataFrame(std_list, index=labels)  # 变成DF格式，索引为类标签
    for j in range(test_data.shape[0]):  # 遍历测试集中的每一个样本
        iset = test_data.iloc[j, :-1].tolist()
        ipro = np.exp(-1*(iset-mean_df)**2/(2*std_df**2)) / (np.sqrt(2*np.pi)*std_df)  # 正态分布公式
        probility = 1  # 初始化当前实例总概率
        for k in range(test_data.shape[1]-1):  # 遍历每一个特征
            probility *= ipro[k]  # 特征概率之积即为当前实例的概率
            iris_class = probility.index[np.argmax(probility.values)]  # 返回最大概率的类别
        result_list.append(iris_class)
    test_data['predict'] = result_list
    acc = (test_data.iloc[:, -1]==test_data.iloc[:, -2]).mean()
    print(f'模型预测准确率为{acc}')

def main():
    for i in range(10):
        rate = 0.8  # 训练集所占比例
        dataset = load_data()
        train_data, test_data = random_split_data(dataset, rate)
        gauss_nb_classify(train_data, test_data)


if __name__ == '__main__':
    main()