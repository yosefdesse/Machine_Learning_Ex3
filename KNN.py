import numpy as np


class KNN:
    '''
    K-Nearest Neighbors classifier. 
    Uses a specified distance metric (p-norm) to classify data points.
    '''

    def __init__(self, k, p):
        self.k = k
        self.p = p

    def init_base(self, train_data, train_label):

        self.base = train_data  # Store the training data
        self.base_label = train_label  # Store the training labels

    def p_distance(self, vector_a, vector_b, p):

        # Ensure points have the same dimension
        assert len(vector_a) == len(vector_b), "Vectors must have the same dimension."

        if p == 0:
            return np.sum([1 if vector_a[i] != vector_b[i] else 0 for i in range(len(vector_a))]) # Hamming distance

        distance = sum(pow(abs(vector_a[i] - vector_b[i]), p) for i in range(len(vector_a)))

        if np.isinf(p):
            return max(abs(vector_a[i] - vector_b[i]) for i in range(len(vector_a)))  # Frechet distance
        else:
            return pow(distance, 1/p)

    def predict_label(self, dist):

        labels = []

        labels = []  
        sorted_indices = np.argsort(dist)[:self.k]  # Get the indices of the k smallest distances (nearest neighbors)
        
        # Collect the labels of the k nearest neighbors
        for idx in sorted_indices:
            labels.append(self.base_label[idx])

        count0 = labels.count(0)
        count1 = labels.count(1)

        return 1 if count1 > count0 else 0

    def predict_test(self, test_data):

        y_pred = [] 
     
        for i in range(len(test_data)):
            dist = [] 
            for j in range(len(self.base)):
                dist.append(self.p_distance(test_data[i], self.base[j], self.p))  # Compute distance to each training point
            y_pred.append(self.predict_label(dist))  # Predict label based on nearest neighbors' labels
        return y_pred 

    def predict_train(self):

        y_pred = [] 
      
        for i in range(len(self.base)):
            dist = []  
            for j in range(len(self.base)):
                dist.append(self.p_distance(self.base[i], self.base[j], self.p))  # Compute distance to each other training point
            y_pred.append(self.predict_label(dist))  # Predict label based on nearest neighbors' labels
        return y_pred 

        



