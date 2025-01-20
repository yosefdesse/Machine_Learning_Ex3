import numpy as np


class KNN:
    def __init__(self, k, p):
        self.k = k
        self.p = p

    def p_distance(self, point1, point2, p):
        X = abs(point1[0]-point2[0])
        Y = abs(point1[1]-point2[1])
        if p==0:
            if X > Y:
                return X
            else:
                return Y

        X = pow(X,p)
        Y= pow(Y,p)
        Z = X+Y
        return pow(Z,1/p)    

    def init_base(self, train_data, train_label):
        self.base = train_data 
        self.base_label = train_label

    def predict_label(self, dist):
        labels = []
        for i in range(self.k):
            idx = 0
            min = dist[0]
            for j in range(len(dist)):
                if dist[j] < min and dist[j] != -1:
                    idx = j
                    min = dist[j]
            labels.append(self.base_label(idx))
            del dist[idx]  
        count0 = 0
        count1 = 0
        for i in range(len(labels)):
            if labels[i] == 0:
                count0 += 1
            else:
                count1 += 1

        return max(count1, count0)         

    def run_knn(self, test_data):
        dist = []
        idx = []
        y_pred = []
        for i in range(len(test_data)):
            for j in range(len(self.base)):
                dist.append(self.p_distance(test_data[i], self.base[j], self.p))
            y_pred.append(self.predict_label(dist))
        return y_pred

                

        





