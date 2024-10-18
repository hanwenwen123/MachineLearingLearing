import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


def loadData(filePath):
    fr = open(filePath, 'r+', encoding='utf-8')
    lines = fr.readlines()
    retData = []
    retCityName = []
    # 将城市名称和数据分开
    for line in lines:
        items = line.strip().split(",")  # 将字符串按","分开，存储为列表形式
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName

'''
聚类效果影响因素：1.初始质心、簇的个数
注：KMeans 的 init 参数默认使用k-means++方法来选择初始质心，存在一定的随机性，每次运行初始质心可能不同，聚类结果也会不一致。
要使每次运行的聚类结果一致，可以通过设置 random_state 参数来固定随机种子。
'''

if __name__ == '__main__':
    cluster_number = 3
    # data列名：食品、衣着、家庭用品及服务、医疗保健、交通通讯、娱乐教育文化、居住及杂项商品和服务
    data, cityName = loadData('CityConsumption.txt')
    km = KMeans(n_clusters=cluster_number)  # 创建 KMeans 聚类模型，n_clusters指定簇的个数;init初始聚类中心的初始化方法，默认kmeans++；max_iter最大迭代次数，默认300
    # km = KMeans(n_clusters=cluster_number, random_state=42)
    label = km.fit_predict(data)  # 聚类并获取簇的标签(一维数组)
    print('label =', label)
    expenses = np.sum(km.cluster_centers_, axis=1)  # 每个簇(每行)的中心点的特征值求和
    # km.cluster_centers_是二维数组n_clusters*8列，每行表示一个簇的质心，每列表示质心的特征值。
    print(expenses)
    CityCluster = [[] for _ in range(cluster_number)]
    for i in range(len(cityName)):  # 根据簇划分城市
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])

    # 检验聚类效果
    # 1.误差平方和(Sum of Squared Errors)，SSE值越小，聚类效果越好。
    print('SSE_value =', km.inertia_)
    # 2. 轮廓系数(Silhouette Coefficient)，轮廓系数越接近 1，表示簇内样本越紧密且簇间样本分离得越好，聚类效果越佳。
    silhouette_avg = silhouette_score(data, label)
    print(f"Silhouette Score: {silhouette_avg}")
    # 3.Calinski-Harabasz指数（也称为方差比准则）衡量簇间分散度和簇内紧密度的比例，值越大，表示聚类效果越好。
    score = calinski_harabasz_score(data, label)
    print(f"Calinski-Harabasz Score: {score}")
    # Davies-Bouldin指数衡量的是簇间的相似性和簇内的紧密性。值越小，表示聚类效果越好。
    score = davies_bouldin_score(data, label)
    print(f"Davies-Bouldin Score: {score}")
    # 5.肘部法(Elbow Method)：绘制不同簇数对应的SSE值，当SSE值随簇数增加出现明显的减缓趋势时，该转折点即为最佳簇数。
    sse_list = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        sse_list.append(km.inertia_)

    plt.plot(range(1, 10), sse_list, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
