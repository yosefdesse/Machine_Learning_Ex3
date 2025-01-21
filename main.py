import numpy as np 
from KNN import KNN
from sklearn.model_selection import train_test_split

def compute_error(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return 1-acc

def split_data(data1, data2):
    data = np.vstack([data1, data2])
    labels = np.hstack([np.ones(len(data1)), np.zeros(len(data2))]) 

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.5, stratify=labels, random_state=None
    )

    return x_train, x_test, y_train, y_test

def run_knn(k, p, data1, data2):
    empirical_errors = []
    true_errors = []
    knn = KNN(k, p)

    for i in range(100):
        train_data, test_data, train_label, test_label = split_data(data1, data2)
        knn.init_base(train_data, train_label)
        train_error = compute_error(knn.predict_train(), train_label)
        test_error = compute_error(knn.predict_test(test_data), test_label)   
        empirical_errors.append(train_error)
        true_errors.append(test_error)
    
    avg_empirical_errors = np.mean(empirical_errors, axis=0)
    avg_true_errors = np.mean(true_errors, axis=0)
    diff = abs(avg_empirical_errors - avg_true_errors)

    return avg_empirical_errors, avg_true_errors, diff


def main (data_txt):
    versicolor, virginica = write_from_file(data_txt)

    for i in range(0, 3):
        for j in range(1, 10, 2):
            avg_empirical_errors, avg_true_errors, diff = run_knn(i, j, versicolor, virginica)
            if i == 0:
                print(f'p : inf, k : {j}, Average empirical Error = {avg_empirical_errors},  Average true Error = {avg_true_errors}, Difference = {diff}')
            else:
                print(f'p : {i}, k : {j}, Average empirical Error = {avg_empirical_errors},  Average true Error = {avg_true_errors}, Difference = {diff}')
        print('\n')


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