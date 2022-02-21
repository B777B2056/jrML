from Utils.DataLoader import DataLoader
from Utils.Ploter import drawDistribution, ClassificationPloter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 读取csv文件，载入iris数据集
    dataloader = DataLoader('DataSets/iris.csv')
    iris = dataloader.toNumpyArray()  # 转换为numpy数组
    drawDistribution(iris[:, :-1], iris[:, -1], 'IRIS Dataset')  # 画IRIS数据集的分布情况
    # 分割数据集，将其分割为训练集，测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(iris[:, :-1], iris[:, -1], test_size=0.2)
    # 引入kNN模型
    model = KNeighborsClassifier()
    # 训练模型
    model.fit(X_train, y_train)
    # 测试集输入训练好的模型，测试模型
    pred = model.predict(X_test)
    print('模型测试集准确率为'+str(model.score(X_test, y_test)*100)+'%')
    ClassificationPloter(y_test, pred).drawConfusionMatrix()
