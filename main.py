import numpy as np 
from KNN import KNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def compute_error(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return 1-acc

def plot_all_errors(empirical_errors, true_errors, k_values , p_value):
    """
    Plots multiple graphs for different k values in a 2x3 layout.

    Args:
        empirical_errors_list (list of lists): List of empirical error arrays for each k.
        true_errors_list (list of lists): List of true error arrays for each k.
        k_values (list): List of k values.
        p_value (float or str): The value of p used in the KNN algorithm.
    """
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

    # Hide the unused subplot in the bottom row
    fig.delaxes(axs[1, 2])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # To make space for the suptitle
    plt.show()



# def plot_graph(avg_empirical_errors, avg_true_errors, title="Empirical vs True Errors", xlabel="Iteration", ylabel="Error"):
#     """
#     Plots the empirical error and true error on the same graph.

#     Args:
#         avg_empirical_errors (list or np.ndarray): List of average empirical errors.
#         avg_true_errors (list or np.ndarray): List of average true errors.
#         title (str): Title of the plot.
#         xlabel (str): Label for the x-axis.
#         ylabel (str): Label for the y-axis.
#     """
#     iterations = range(1, len(avg_empirical_errors) + 1)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, avg_empirical_errors, label="Empirical Error", color="blue", marker="o")
#     plt.plot(iterations, avg_true_errors, label="True Error", color="red", marker="o")
    
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def split_data(data1, data2):
    data = np.vstack([data1, data2])
    labels = np.hstack([np.ones(len(data1)), np.zeros(len(data2))]) 

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.5, stratify=labels, random_state=None
    )

    return x_train, x_test, y_train, y_test

def run_knn(p, k, data1, data2):
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
    return avg_empirical_errors, avg_true_errors, diff, empirical_errors, true_errors


def main (data_txt):
    versicolor, virginica = write_from_file(data_txt)

    for i in range(0, 3):
        all_empirical_errors = []
        all_true_errors = []
        k_values = [1,3,5,7,9]
        for j in range(1, 10, 2):
            avg_empirical_errors, avg_true_errors, diff, empirical_errors, true_errors = run_knn(i, j, versicolor, virginica)
            all_empirical_errors.append(empirical_errors)
            all_true_errors.append(true_errors)
            if i == 0:
                print(f'p : inf, k : {j}, Average empirical Error = {avg_empirical_errors},  Average true Error = {avg_true_errors}, Difference = {diff}')
            else:
                print(f'p : {i}, k : {j}, Average empirical Error = {avg_empirical_errors},  Average true Error = {avg_true_errors}, Difference = {diff}')
        print('\n')
        plot_all_errors(all_empirical_errors, all_true_errors, k_values, i)


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