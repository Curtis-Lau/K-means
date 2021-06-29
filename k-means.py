import numpy as np
import pandas as pd

# 计算欧氏距离
def distEuclid(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离

# 初始化聚簇中心，取k个随机质心
def init_Cent(dataSet, k):
    n = dataSet.shape[1]
    centcoords = np.mat(np.zeros((k, n)))  # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = np.min(dataSet.iloc[:, j])
        maxJ = np.max(dataSet.iloc[:, j])
        rangeJ = float(maxJ - minJ)
        centcoords[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # 最小值加上一个随机数
    return centcoords

# k-means聚类算法：收敛的标志是所有点的类别不再改变
def kMeans(dataSet, k, distEuclid=distEuclid, init_Cent = init_Cent):
    m = dataSet.shape[0]
    clusterDist = np.mat(np.zeros((m, 2)))  # 用于存放该样本属于哪类及质心距离
    # clusterDist第一列存放该数据所属的中心点（哪一类），第二列是该数据到中心点的距离
    centcoords = init_Cent(dataSet, k)
    clusterChanged = True  # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = np.inf
            label = -1
            for j in range(k):
                distJI = distEuclid(centcoords[j, :], np.mat(list(dataSet.iloc[i, :])))
                if distJI < minDist:
                    minDist = distJI
                    label = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterDist[i, 0] != label: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterDist[i, :] = label, minDist ** 2  # 并将第i个数据点的分配情况存入字典
        # print(clusterAssment)
        for cent in range(k):  # 重新计算中心点
            ptsInClust = dataSet.iloc[np.nonzero(clusterDist[:, 0].A == cent)[0],:]  # 取出每一个类别的所有数据
            centcoords[cent, :] = np.mean(ptsInClust, axis=0)  # 算出这些数据的中心点
    return centcoords, clusterDist

# # ----------------------测试----------------------

df = pd.read_excel('q2data.xlsx')
df.set_index('eventid',inplace=True)

myCentcoords, clustDist = kMeans(df, 20)
print(myCentcoords)
print(clustDist)

# # 用测试数据及测试kmeans算法
# import estimate_criterion
# import matplotlib.pyplot as plt
# import tushare as ts
# pro = ts.pro_api()
#
# start_date = '20190101'
# end_date = '20191231'
# duration = 1
# k = 2
#
# hs300 = pro.index_weight(index_code='399300.SZ', start_date='20190603', end_date='20190603')
# code = list(hs300.con_code)
# dataSet = pro.index_daily(ts_code='399300.SZ', start_date=start_date, end_date=end_date, fields="trade_date")
# dataSet.set_index("trade_date",inplace=True)
# for i in code:
#     stock_data = pro.daily(ts_code=i, start_date=start_date, end_date=end_date, fields='trade_date,close')
#     stock_data.set_index('trade_date',inplace=True)
#     dataSet = pd.merge(dataSet, stock_data, how='outer', left_on=dataSet.index, right_on=stock_data.index)
#     dataSet.set_index('key_0',inplace=True)
#     dataSet.index.name = 'date'
# dataSet.columns = code
# dataSet.index = pd.to_datetime(dataSet.index)
# dataSet.sort_index(inplace=True)
# nan_sum = dataSet.isna().sum().sort_values()
# del_stock = list(nan_sum[nan_sum > (len(dataSet)//10)].index)
# dataSet.drop(del_stock,axis=1,inplace=True)
# dataSet.fillna(method='bfill',inplace=True)
# retSet = dataSet/dataSet.iloc[0,:] - 1
#
# cluster_data = pd.DataFrame(estimate_criterion.calc_estimate(retSet,duration))
# cluster_data = (cluster_data-cluster_data.mean())/cluster_data.std()
# myCentcoords, clustDist = kMeans(cluster_data, k)
# # print(myCentcoords)
# # print(clustDist)
# cluster_data['class'] = clustDist[:,0].astype('int')
# # print(cluster_data)
#
# for i in range(k):
#     plt.scatter(cluster_data[cluster_data['class']==i].iloc[:,0],cluster_data[cluster_data['class']==i].iloc[:,1],label=i)
# plt.legend()
# plt.show()