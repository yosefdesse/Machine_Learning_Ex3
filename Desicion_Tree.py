import numpy as np
import matplotlib.pyplot as plt
import math


class Node:
    '''
    Represents a single node in the decision tree. 
    Stores value, feature, threshold, and child nodes.
    '''
    def __init__(self, val=None, feature=None, threshold=None, left_child=None, right_child=None):
        self.val = val
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.error = None
        
class DecisionTreeBase:
    '''
    Base class for decision trees. 
    Handles common operations like data splitting and error computation.
    '''

    def __init__(self, data, max_depth=2):
        self.data = data
        self.max_depth = max_depth
        self.root = None
        self.min_error = float('inf')

    def init_root(self, data): # Initialize the tree root
        self.root = Node(val=data)

    def split_data(self, data, threshold, feature_idx): # Splits the data based on a feature and threshold
        left, right = [], []
        for features, label in data:
            if features[feature_idx] < threshold:
                left.append((features, label))
            else:
                right.append((features, label))
        return left, right

    def compute_leaf_error(self, data): # Computes the error for a leaf node (based on label distribution)
        if not data:
            return 0
        count_0 = sum(1 for _, label in data if label == 0)
        count_1 = len(data) - count_0
        guess_error = min(count_0, count_1) / len(data)
        donate = len(data)/len(self.root.val)
        return guess_error*donate


    def compute_tree_error(self, node):
    
        if not node.left_child and not node.right_child:
            return self.compute_leaf_error(node.val)
        
        error = 0

        if node.left_child:
            left_error = self.compute_tree_error(node.left_child)
            error += left_error 

        if node.right_child:
            right_error = self.compute_tree_error(node.right_child)
            error += right_error  
        return error


class DecisionTreeBruteForce(DecisionTreeBase):
    '''
    Decision tree using brute force method. 
    Evaluates all possible splits to find the best tree structure.
    '''

    def build_tree_brute_force(self, node, depth=0): # Recursively builds the tree using brute force splits
        
        if depth >= self.max_depth or len(node.val) <= 1:
            error = self.compute_leaf_error(node.val)
            node.error = error
            return [node]

        all_trees = []

        for feature_idx in range(len(node.val[0][0])):
            thresholds = set(features[feature_idx] for features, _ in node.val)

            for threshold in thresholds:
                left, right = self.split_data(node.val, threshold, feature_idx)

                if not left or not right: 
                    continue

                node.feature = feature_idx
                node.threshold = threshold
                node.left_child = Node(val=left)
                node.right_child = Node(val=right)

                left_trees = self.build_tree_brute_force(node.left_child, depth + 1)
                right_trees = self.build_tree_brute_force(node.right_child, depth + 1)
                for left_tree in left_trees:
                    for right_tree in right_trees:
                        new_root = Node(val=node.val)
                        new_root.feature = feature_idx
                        new_root.threshold = threshold
                        new_root.left_child = left_tree
                        new_root.right_child = right_tree
                        all_trees.append(new_root)

        return all_trees
    
    def train(self): # Initializes the root and trains the tree using brute force

        self.init_root(self.data)
        all_trees = self.build_tree_brute_force(self.root)
        
        best_tree = None
        min_error = float('inf') 

        for i, tree in enumerate(all_trees):
            error = self.compute_tree_error(tree)
            if error < min_error:
                min_error = error
                best_tree = tree

        self.root = best_tree
        self.min_error = min_error  
        return best_tree, self.min_error



class DecisionTreeEntropy(DecisionTreeBase):
    '''
    Decision tree using entropy method (information gain).
    Choose the best feature and threshold for splits by entropy.
    '''

    def compute_entropy(self, data): # Calculates entropy of a dataset
        
        p0 = sum(1 for _, label in data if label == 0) / len(data)
        p1 = 1 - p0
        if p0 == 0 or p1 == 0: # Avoid log(0) --> ∞
            return 0
        return -p0 * math.log2(p0) - p1 * math.log2(p1)

    def best_entropy(self, data): # Finds the best split based on entropy
        
        min_entropy = float('inf') 
        split_feature_idx = None
        split_threshold = None
        left_val = None
        right_val = None
        for feature_idx in range(len(data[0][0])): 
            thresholds = set(features[feature_idx] for features, _ in data)

            for threshold in thresholds:
                left, right = self.split_data(data, threshold, feature_idx)

                if not left or not right:
                    continue

                left_entropy = self.compute_entropy(left)
                right_entropy = self.compute_entropy(right)

                total_entropy = left_entropy + right_entropy

                if total_entropy < min_entropy:
                    min_entropy = total_entropy
                    split_feature_idx = feature_idx
                    split_threshold = threshold
                    left_val = left
                    right_val = right

        return left_val, right_val, split_feature_idx, split_threshold

    def build_tree_binary_entropy(self, node, depth=0): # Recursively builds the tree using entropy splits
        
        if depth >= self.max_depth or len(node.val) <= 1:
            error = self.compute_leaf_error(node.val)
            node.error = error
            return node 

        left_val, right_val, feature_idx, threshold = self.best_entropy(node.val)

        if feature_idx is None or threshold is None:
            error = self.compute_leaf_error(node.val)
            node.error = error
            return node

        node.feature = feature_idx
        node.threshold = threshold

        node.left_child = self.build_tree_binary_entropy(Node(val=left_val), depth + 1)
        node.right_child = self.build_tree_binary_entropy(Node(val=right_val), depth + 1)

        return node

    def train(self): # Initializes the root and trains the tree using entropy
        
        self.init_root(self.data) 
        self.root = self.build_tree_binary_entropy(self.root)
        self.min_error = self.compute_tree_error(self.root)
