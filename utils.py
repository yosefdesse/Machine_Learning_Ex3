import numpy as np 
import matplotlib.pyplot as plt
from KNN import KNN


# ----------------------------- KNN functions -----------------------------

def compute_error(y_true, y_pred):
    error = np.sum(y_true != y_pred) / len(y_true)  # Compute accuracy
    return error  

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

# ----------------------------- Desicion Tree functions -----------------------------

def prepare_desicion_tree_data(data1, data2):
    combine_data = []
    for i in range(len(data1)):
        element = (data1[i], 0)
        combine_data.append(element)
    for i in range(len(data2)):
        element = (data2[i], 1)
        combine_data.append(element)

    return combine_data

def visualize_tree(node, depth=0, ax=None, x_pos=0, y_pos=0, x_step=1, y_step=1):
    if node is None:
        return
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if node.error is not None:
        text = f"Error:\n{node.error:.4f}"
    else:
        text = "Split\n"
        text += f"Feature: {node.feature}\nThreshold: {node.threshold}"

    ax.text(x_pos, y_pos, text, ha='center', va='center', bbox=dict(boxstyle="round", facecolor="wheat"))

    if node.left_child is not None:
        ax.plot([x_pos, x_pos - x_step], [y_pos, y_pos - y_step], 'k-')
        visualize_tree(node.left_child, depth + 1, ax, x_pos - x_step, y_pos - y_step, x_step / 2, y_step)

    if node.right_child is not None:
        ax.plot([x_pos, x_pos + x_step], [y_pos, y_pos - y_step], 'k-')
        visualize_tree(node.right_child, depth + 1, ax, x_pos + x_step, y_pos - y_step, x_step / 2, y_step)

    if depth == 0:
        plt.show()


# ----------------------------- Data Loading and Preparation -----------------------------

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