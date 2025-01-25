import numpy as np 
from KNN import KNN
import matplotlib.pyplot as plt
from Desicion_Tree import DecisionTreeBruteForce, DecisionTreeEntropy
from utils import write_from_file, split_data, run_knn_multiple_times, visualize_tree, prepare_desicion_tree_data

def compute_error(y_true, y_pred):
    error = np.sum(y_true != y_pred) / len(y_true)  # Compute accuracy
    return error  

def plot_all_errors(empirical_errors, true_errors, k_values, p_value):

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Empirical vs True Errors for p = {p_value}", fontsize=16)

    # Loop through k_values to create the subplots
    for idx, k in enumerate(k_values):
        row = idx // 3  
        col = idx % 3 
        axs[row, col].plot(empirical_errors[idx], label="Empirical Error", marker='o', color='blue')
        axs[row, col].plot(true_errors[idx], label="True Error", marker='o', color='red')
        axs[row, col].set_title(f"k = {k}")
        axs[row, col].set_xlabel("Iteration")
        axs[row, col].set_ylabel("Error")
        axs[row, col].legend()
        axs[row, col].grid(True)

    fig.delaxes(axs[1, 2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.show()

def write_from_file(data_txt):

    versicolor = []
    virginica = []
    with open(data_txt, 'r') as file:

        for line in file:
            columns = line.split()  
            # Assign versicolor or virginica based on the species name
            if columns[4] == 'Iris-versicolor':
                versicolor.append([float(columns[1]), float(columns[2])])
            if columns[4] == 'Iris-virginica':
                virginica.append([float(columns[1]), float(columns[2])])
    versicolor = np.array(versicolor)
    virginica = np.array(virginica)

    return versicolor, virginica

def split_data(data1, data2):

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    
    # Split data1 (class 0) using Bernoulli process
    for i in range(len(data1)):
        result = np.random.binomial(1, 0.5)
        if result == 0:
            x_train.append(data1[i])
            y_train.append(0)  
        else:
            x_test.append(data1[i])
            y_test.append(0)

    # Split data2 (class 1) using Bernoulli process
    for i in range(len(data2)):
        result = np.random.binomial(1, 0.5)
        if result == 0:
            x_train.append(data2[i])
            y_train.append(1) 
        else:
            x_test.append(data2[i])
            y_test.append(1)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def run_knn_multiple_times(p, k, data1, data2, n):

    empirical_errors = []
    true_errors = []
    knn = KNN(k, p)

    # Run the KNN algorithm for n iterations
    for i in range(n):
        train_data, test_data, train_label, test_label = split_data(data1, data2)
        knn.init_base(train_data, train_label)  # Initialize KNN with the training data
        train_error = compute_error(knn.predict_train(), train_label)  # Calculate training error
        test_error = compute_error(knn.predict_test(test_data), test_label)  # Calculate test error   
        empirical_errors.append(train_error)
        true_errors.append(test_error)

    # Calculate the average errors and the difference between empirical and true errors
    avg_empirical_errors = np.mean(empirical_errors, axis=0)
    avg_true_errors = np.mean(true_errors, axis=0)
    diff = abs(avg_empirical_errors - avg_true_errors)
    return avg_empirical_errors, avg_true_errors, diff, empirical_errors, true_errors


def main(data_txt):

    versicolor, virginica = write_from_file(data_txt)

# ----------------------- Run KNN -----------------------

    print("----------------------- Run KNN -----------------------")

    k_values = [1, 3, 5, 7, 9] # Use for plot

    # Run KNN for different p values (1, 2, ∞) and different k values (1, 3, 5, 7, 9)
    for p in range(0, 3):
        if p == 0 : p = float('inf')
        all_empirical_errors = []
        all_true_errors = []
        
        for k in range(1, 10, 2):
            avg_empirical_errors, avg_true_errors, diff, empirical_errors, true_errors = run_knn_multiple_times(p, k, versicolor, virginica, 100)
            all_empirical_errors.append(empirical_errors)
            all_true_errors.append(true_errors)
            print(f'p : {p}, k : {k}, Average empirical Error = {avg_empirical_errors},  Average true Error = {avg_true_errors}, Difference = {diff}')
        
        print('\n')
        plot_all_errors(all_empirical_errors, all_true_errors, k_values, p)
        if np.isinf(p) : p = 0

# ----------------------- Run Desicion Tree -----------------------

    print("----------------------- Run Desicion Tree -----------------------")
    
    # Preparing the data for the decision tree
    data = prepare_desicion_tree_data(versicolor, virginica)

    # Creating a decision tree using brute force method with a maximum depth of 2 (3 levels)
    brutce_force_tree = DecisionTreeBruteForce(data=data, max_depth=2)
    brutce_force_tree.train()  

    print("Desicion Tree Brute Force ERROR :")
    print(f"Minimum error (Brute Force): {brutce_force_tree.min_error}\n")

    
    visualize_tree(brutce_force_tree.root) # Visualizing decision tree 

    # Creating a decision tree using entropy method with a maximum depth of 2 (3 levels)
    binary_entropy_tree = DecisionTreeEntropy(data=data, max_depth=2)
    binary_entropy_tree.train()  

    print("Desicion Tree Binary Entropy ERROR :\n")
    print(f"Minimum error (Entropy): {binary_entropy_tree.min_error}")

    visualize_tree(binary_entropy_tree.root) # Visualizing decision tree 

if __name__ == "__main__":
    DATA_PATH = 'iris.txt'  
    main(DATA_PATH)
