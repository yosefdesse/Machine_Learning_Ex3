import numpy as np 
from Desicion_Tree import DecisionTreeBruteForce, DecisionTreeEntropy
from utils import write_from_file, plot_all_errors, run_knn_multiple_times, visualize_tree, prepare_desicion_tree_data

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
       # plot_all_errors(all_empirical_errors, all_true_errors, k_values, p)
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
