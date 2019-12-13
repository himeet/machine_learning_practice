# 基于iris数据集的k-means聚类
# iris数据集有4个特征，聚类结果不便于可视化

import numpy as np
import pandas as pd

"""
函数功能:读取数据集
输入:无
返回:数据集的DataFrame格式
"""
def load_data():
    iris_data = pd.read_csv('./datasets/iris/iris.data', header=None)  # data文件内容是逗号分隔，所以可以使用read_csv
    #print(iris_data.head())
    #print(iris_data.shape)  # 150行5列
    iris_data = pd.DataFrame(iris_data)
    return iris_data

"""
函数功能：计算两个数据集之间的欧式距离
参数说明：
    arr1:array的数据集1
    arr2:array的数据集2
返回：两个数据集之间的欧式距离
"""
def distance_euclidean(arr1, arr2):
    d = np.sum(np.power(arr1 - arr2, 2), axis=1)  # axis=1，对列求和
    dist = np.power(d, 0.5)
    return dist

"""
函数功能：随机生成k个质心，从整个列的最大最小值中选取
参数说明：
    dataset:包含标签的数据集
    k:簇的个数
返回：k个质心，array格式
"""
def random_centroid(dataset, k):
    column_num = dataset.shape[1]
    data_min = dataset.iloc[:, 0:column_num-1].min()  # 每一列的最小值
    data_max = dataset.iloc[:, 0:column_num-1].max()  # 每一列的最大值
    data_centroid = np.random.uniform(data_min, data_max, size=(k, column_num-1))  # 参数size为几行几列
    return data_centroid

"""
函数功能：k-means聚类核心算法
参数说明：
    dataset:带标签的数据集
    k:簇的个数
    dist_measure:欧式距离计算函数
    create_random_centroid:随机质心生成函数
返回：
    centroids:聚类后的质心
    result_set:所有数据的划分结果
"""
def kmeans(dataset, k, dist_measure=distance_euclidean, create_random_centroid=random_centroid):
    row_n, col_n = dataset.shape
    centroids =create_random_centroid(dataset, k)
    cluster_info = np.zeros((row_n, 3))
    cluster_info[:, 0] = np.inf  # 第0列存放距离，初始为无穷大
    cluster_info[:, 1:3] = -1  # 第1列存放本次迭代中簇的编号， 第2列中存放上一次迭代中簇的编号
    cluster_info = pd.DataFrame(cluster_info)
    result_set = pd.concat([dataset, cluster_info], axis=1, ignore_index=True)  # axis=1，对列拼接;ignore_index=True使得列号可以顺延
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(row_n):  # 遍历每一个样本点
            dist = dist_measure(dataset.iloc[i, :col_n-1].values, centroids)
            result_set.iloc[i, col_n] = dist.min()
            result_set.iloc[i, col_n+1] = np.where(dist == dist.min())[0]
        cluster_changed = not((result_set.iloc[:, -1] == result_set.iloc[:, -2]).all())
        if cluster_changed:
            centroid_renewed =  result_set.groupby(col_n+1).mean()
            centroids = centroid_renewed.iloc[:, :col_n-1].values
            result_set.iloc[:, -1] = result_set.iloc[:, -2]  # 当前点所在簇的编号成为了上一次簇的编号
        return centroids, result_set

def main():
    dataset = load_data()
    k = 3
    centroids, result_set = kmeans(dataset, k)
    print(centroids)  # 最终质心
    print(result_set)  # 聚类最终结果
    print(result_set.iloc[:, -1].value_counts())


if __name__ == '__main__':
    main()
