import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

MAXDOUBLE = sys.float_info.max

# Fast Nearest Neighbor class
class FNN():
    '''
        This function is the constructor of the class.
        It takes as input the training and testing data.
        It also takes as input the number of components for the PCA.
        It initializes the attributes of the class.
    '''
    def __init__(self, X_train: list, y_train: list, X_test: list, y_test: list, n_components: int = 30, normalize_data=True) -> None:
        # TRAIN PART
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        if normalize_data:
            X_train_pca = X_train_pca / 255

        df_X_train = pd.DataFrame(data=X_train_pca)
        self.pointSet: list[list[float]] = [df_X_train[i].values.tolist() for i in df_X_train]
        self.y_train = y_train

        # n => nombre de ligne
        self.n = len(self.pointSet[0])
        # d => nombre de dimesions
        self.d = len(self.pointSet)

        # TEST PART
        self.X_test = self.pca.transform(X_test)
        if normalize_data:
            self.X_test = self.X_test / 255
        self.y_test = y_test

        self.orderedSet: list[list[float]] = [[0 for col in range(self.n)] for row in range(self.d)]
        self.bmap: list[list[int]] = [[0 for col in range(self.n)] for row in range(self.d)]
        self.bmap = self.bmap[0]
        self.fmap: list[list[int]] = [[0 for col in range(self.n)] for row in range(self.d)]

    '''
        Sort the pointSet by the value of the points.

        :param pointSet: the pointSet to be sorted
        :param tempMap: the tempMap to store the sorted order of poinSet
        :return: None
    '''
    def sort(self, pointSet: list[float], tempMap: list[int]):
        # init tempMap
        for i in range(self.n):
            tempMap[i] = i
        # bubble sort
        for i in range(self.n):
            for j in range(self.n-i-1):
                if pointSet[tempMap[j]] > pointSet[tempMap[j+1]]:
                    t = tempMap[j+1]
                    tempMap[j+1] = tempMap[j]
                    tempMap[j] = t

    '''
        This function preprocesses the data.
        The sorted order of pointSet is stored in tempMap.
        It sorts the data in each dimension and stores the sorted data in orderedSet.
        It stores the mapping from the pointSet to the orderedSet in fmap.
        It stores the mapping from the orderedSet to the pointSet in bmap.
        The function is called in the constructor.

        :param self: The object of the class.
        :return: None
    '''
    def preprocess(self):
        tempMap: list[int] = [0] * self.n

        # sort pointSet and stor the order in tempMap
        self.sort(self.pointSet[0], tempMap)
        # Create backward map and forward map for first dimension
        for i in tqdm(range(self.n)):
            self.orderedSet[0][i] = self.pointSet[0][tempMap[i]]
            self.bmap[i] = tempMap[i]
            self.fmap[0][tempMap[i]] = i

        # Create forward map for each dimensions by sorting corresponding coordinate in point set
        for i in tqdm(range(1, self.d)):
            self.sort(self.pointSet[i], tempMap)
            for j in range(self.n):
                self.orderedSet[i][j] = self.pointSet[i][tempMap[j]]
                self.fmap[i][tempMap[j]] = j

    def binarySearch(self, orderedSet: list[float], v: float) -> int :
        bottom = 0
        top = self.n

        while top > bottom+1:
            center = (bottom + top) // 2
            if v < orderedSet[center]:
                top = center
            else:
                bottom = center
        return bottom
    
    '''
        This function finds the closest point to the given point p within the given epsilon.
        It does this by first finding the range of points that could possibly be within epsilon
        of p. It then trims this list by checking the other dimensions. Finally, it performs
        an exhaustive search on the remaining points to find the closest point.

        :param self: The object of the class.
        :param p: the point we want to find the closest value to
        :param epsilon: the distance value we use to find the closest point to p
        :return: the closest point to p found within 2 epsilon
    '''
    def closet(self, p: list[float], epsilon: float) -> int:
        bottom = self.binarySearch(self.orderedSet[0], p[0]-epsilon-1)
        top = self.binarySearch(self.orderedSet[0], p[0]+epsilon)

        # create list
        listElem = 0
        my_list = [0] * self.n
        for i in range(bottom, top):
            my_list[listElem] = self.bmap[i]
            listElem += 1
        
        # trim list with binary searches on other dim along with lookups
        for i in range(1, self.d):
            bottom = self.binarySearch(self.orderedSet[i], p[i]-epsilon-1)
            top = self.binarySearch(self.orderedSet[i], p[i]+epsilon)

            m = listElem
            listElem = 0

            for j in range(m):
                if self.fmap[i][my_list[j]] <= top and self.fmap[i][my_list[j]] >= bottom:
                    my_list[listElem] = my_list[j]
                    listElem += 1
            
        # perform exhaustive search on remaining points
        max = MAXDOUBLE
        pos: int = 0
        for i in range(listElem):
            t = 0
            for j in range(self.d):
                t += (p[j] - self.orderedSet[j][self.fmap[j][my_list[i]]]) ** 2
            if t < max:
                max = t
                pos = my_list[i]
        return pos