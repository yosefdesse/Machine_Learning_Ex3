import numpy as np 
import random
from KNN import KNN

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

def split_data(data1, data2):
    combine_data = np.vstack(data1, data2)
    labels = np.vstack([np.ones(len(data1)), np.zeros(len(data2))])
    data = np.hstack([combine_data, labels])
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_indices = indices[:len(data) // 2]
    test_indices = indices[len(data) // 2:]
    return train_indices, test_indices

def main (data_txt):
    versicolor, virginica = write_from_file(data_txt)
    train_data, test_data = split_data(versicolor, virginica)

    train_label = 



def write_from_file(data_txt):
    versicolor = []
    virginica = []
    with open(data_txt, 'r') as file:

        for line in file:
            columns = line.split()  
            if columns[4] == 'Iris-versicolor':
                versicolor.append([float(columns[1]), float(columns[2])])
            if columns[4] == 'Iris-virginica':
                virginica.append([float(columns[1]), float(columns[2])])
    versicolor = np.array(versicolor)
    virginica = np.array(virginica)

    return versicolor, virginica

if __name__ == "__main__":

    DATA_PATH = 'iris.txt'  # Update with the path to your data
    main(DATA_PATH)