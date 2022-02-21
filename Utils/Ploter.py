import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import matplotlib
matplotlib.use('TkAgg')

def drawDistribution(x_data: np.ndarray, y_data: np.ndarray, title: str) -> None:
    colors = {}
    i = 0
    for l in set(y_data):
        colors[l] = i
        i += 1
    labels = []
    for y in y_data:
        labels.append(colors[y])
    plt.title(title)
    scatter = plt.scatter(x_data[:, 0], x_data[:, 1], c=labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=set(y_data), loc='best')
    plt.show()

class ClassificationPloter:
    def __init__(self, test_label, pred_label):
        self.test_label = test_label
        self.pred_label = pred_label
        self.label_n = list(set(self.test_label))

    def drawConfusionMatrix(self, is_save=False, save_path=''):
        maxtrix = sm.confusion_matrix(self.test_label, self.pred_label, labels=self.label_n)
        plt.matshow(maxtrix)
        plt.colorbar()
        plt.xlabel('Predict')
        plt.ylabel('Real')
        plt.xticks(np.arange(maxtrix.shape[1]), self.label_n)
        plt.yticks(np.arange(maxtrix.shape[1]), self.label_n)
        for first_index in range(len(maxtrix)):
            for second_index in range(len(maxtrix[first_index])):
                plt.text(first_index, second_index, maxtrix[first_index][second_index])
        plt.show()
        if is_save:
            plt.savefig(save_path)

    def printReport(self, is_save=False, save_path=''):
        dataframe = pd.DataFrame(sm.classification_report(self.test_label, self.pred_label)).transpose()
        print(dataframe)
        if is_save:
            dataframe.to_csv(save_path+'classification_report.csv', index=False)
