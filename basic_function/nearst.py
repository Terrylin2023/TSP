import numpy as np
import math

def load_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  # Read the first line to get n and remove any leading/trailing whitespace
        matrix = [list(map(float, line.strip().split())) for line in file]  # Read and process each subsequent line  
    return matrix

class Node:
    def __init__(self, path, solution_cost, graph):
        self.path = path
        self.solution_cost = solution_cost
        self.graph = graph
        self.lower_bound = 0

def findNearestNeighbor(current_node, path, m):
    """
    Find the nearest neighbor node to the current node that has not been visited.
    """
    n = len(m)
    nearest_distance = float('inf')
    nearest_node = None
    for neighbor in range(n):
        if neighbor not in path and m[current_node][neighbor] < nearest_distance:
            nearest_distance = m[current_node][neighbor]
            nearest_node = neighbor
    return nearest_node

def NearestNeighbor(m, start):
    """
    Finds a path through all nodes starting from a specific node using the nearest neighbor algorithm.
    """
    n = len(m)
    node = start
    path = [node]

    for _ in range(1, n):
        next_node = findNearestNeighbor(node, path, m)
        path.append(next_node)
        node = next_node
    
    return path

# # Example usage
# file_path = '/Users/terrylin/Desktop/2024 Winter/Iintroduction of AI/Project/code/tsp-problem-4-5-3-1-2.txt'  # Replace with your actual file path
# matrix = load_matrix_from_file(file_path)
# path = NearestNeighbor(matrix, 0)
# print("Path:", path)

def cost_of(path, m):
    """
    Calculate the total cost of a given path based on the distance matrix 'm'.
    
    Parameters:
    - path: A list of node indices representing the visited order.
    - m: The distance matrix where m[i][j] represents the distance from node i to node j.
    
    Returns:
    - The total cost of the path.
    """
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += m[path[i]][path[i+1]]
    # Optionally add the cost to return to the start for TSP
    total_cost += m[path[-1]][path[0]]
    return total_cost

# Example usage
file_path = '/Users/terrylin/Desktop/2024 Winter/Iintroduction of AI/Project/code/tsp-problem-15-10-5-1-1.txt'  # Replace with your actual file path
matrix = load_matrix_from_file(file_path)
path = NearestNeighbor(matrix, 0)
cost = cost_of(path, matrix)
print("Path:", path)
print("Cost:", cost)


def precompute_distances(matrix):
    n=len(matrix)
    distances = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[(i, j)] = matrix[i][j]
    return distances

def calculate_cost_change(distances, path, i, j):
    # 确保不越界
    n = len(path)
    before_swap = distances[(path[i-1], path[i])] + distances[(path[j], path[(j+1) % n])]
    after_swap = distances[(path[i-1], path[j])] + distances[(path[i], path[(j+1) % n])]
    return after_swap - before_swap


# v2
def two_opt(path, m):
    improved = True
    best_cost = cost_of(path, m)
    while improved:
        improved = False
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue  # 忽略相邻的节点
                new_path = path[:]
                new_path[i:j] = path[j-1:i-1:-1]  # 逆转路径段
                new_cost = cost_of(new_path, m)
                if new_cost < best_cost:
                    path, best_cost = new_path, new_cost
                    improved = True
    return path


# v3
# def two_opt(path, distances):
#     improved = True
#     while improved:
#         improved = False
#         for i in range(1, len(path) - 2):
#             for j in range(i + 1, len(path)-1):
#                 cost_change = calculate_cost_change(distances, path, i, j)
#                 if cost_change < 0:  # 说明交换可以减少路径长度
#                     path[i:j] = path[j-1:i-1:-1]  # 执行交换
#                     improved = True
#     return path

def upper_bound(node):
    start_node = node.path[0]  # Assuming the start node is the first in the path
    nn_path = NearestNeighbor(node.graph, start_node)
    nn_cost = cost_of(nn_path, node.graph)  # Pass the graph as the distance matrix
    # apply 2-opt to improve the upper bound
    optimized_path = two_opt(nn_path, node.graph)
    optimized_cost = cost_of(optimized_path, node.graph)
    return optimized_cost, optimized_path

# Example usage
file_path = '/Users/terrylin/Desktop/2024 Winter/Iintroduction of AI/Project/code/tsp-problem-15-10-5-1-1.txt'  # Replace with your actual file path
matrix = load_matrix_from_file(file_path)
node = Node([0], 0, matrix)
ub_cost, ub_path = upper_bound(node)
print("Upper Bound Path:", ub_path)
print("Upper Bound Cost:", ub_cost)

