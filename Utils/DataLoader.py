import numpy as np
import pandas as pd
from typing import Tuple

class DataLoader:
    """从文件中加载数据，可进行顺序打乱、转换为numpy数组等操作
        Public Attributes:
            None
    """
    def __init__(self, path: str, shuffle: bool = True, sep: str = ',', header: bool = True):
        """从文件中读取数据
            Args:
                path: 类型为字符串，待读取的文件的路径；
                shuffle: 类型为bool，是否打乱数据集顺序，默认为True（即打乱顺序）；
                sep: 类型为字符串，文件中各内容的分隔符，默认为英文逗号；
                header: 类型为bool，用于指示文件第一行是否为各列名称，若为True则不将其读为有效数据行，默认为True。

            Returns:
                None
        """
        __pddata = pd.read_csv(filepath_or_buffer=path, sep=sep, header='infer' if header else None)
        if shuffle:
            __pddata = __pddata.sample(frac=1.0)
        self.__data = __pddata.values

    def __getitem__(self, item):
        return self.__data[item]

    def __iter__(self):
        for data in self.__data:
            yield data

    def shape(self) -> Tuple[int, int]:
        """返回数据的行和列长度
            Returns:
                类型为元组，即Tuple[int, int]，含义为数据的(行数，列数)
        """
        return self.__data.shape

    def toNumpyArray(self) -> np.ndarray:
        """返回数据的行和列长度
            Returns:
                包含了所有数据的numpy数组
        """
        return self.__data
