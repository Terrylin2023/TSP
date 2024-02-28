import numpy as np
import math
import networkx as nx
import time
import heapq
import random

start_time = time.time()

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
    
    def __lt__(self, other):
        return self.lower_bound < other.lower_bound


def is_leaf(node):
    # Assuming a complete graph, a leaf is when the path includes all nodes.
    return len(node.path) == len(node.graph)

def generate_children(current_node):
    children = []
    last_node_in_path = current_node.path[-1]
    for i in range(len(current_node.graph)):
        if i not in current_node.path:
            new_path = current_node.path + [i]
            new_solution_cost = current_node.solution_cost + current_node.graph[last_node_in_path][i]
            new_node = Node(new_path, new_solution_cost, current_node.graph)
            children.append(new_node)
    return children

# upper bound
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


def upper_bound(node):
    start_node = node.path[0]  # Assuming the start node is the first in the path
    nn_path = NearestNeighbor(node.graph, start_node)
    nn_cost = cost_of(nn_path, node.graph)  # Pass the graph as the distance matrix
    return nn_cost, nn_path



# lower bound

def select_root_node(graph):
    # 示例中选择0作为根节点
    # 实际选择可能基于特定逻辑

    return random.randint(0, len(graph)-1)

def find_minimum_spanning_tree_cost(graph, root_node):
    G = nx.Graph()
    n = len(graph)
    # 添加边和权重，忽略根节点
    for i in range(n):
        for j in range(i+1, n):
            if i != root_node and j != root_node:
                G.add_edge(i, j, weight=graph[i][j])
    mst = nx.minimum_spanning_tree(G)
    return mst.size(weight='weight')

def find_two_shortest_edges_to_root(graph, root_node):
    edges = [(graph[root_node][i], i) for i in range(len(graph)) if i != root_node]
    two_shortest_edges = sorted(edges, key=lambda x: x[0])[:2]
    return sum(edge[0] for edge in two_shortest_edges)

def one_tree_relaxation(graph):
    root_node = select_root_node(graph)
    mst_cost = find_minimum_spanning_tree_cost(graph, root_node)
    two_shortest_edges_cost = find_two_shortest_edges_to_root(graph, root_node)
    return mst_cost + two_shortest_edges_cost

# BnB

def BnB(initial_node):
    best_solution = math.inf
    best_solution_path = None
    solution_queue = []

    initial_upper_bound, initial_path = upper_bound(initial_node)
    if initial_upper_bound < best_solution:
        best_solution = initial_upper_bound
        best_solution_path = initial_path

    heapq.heappush(solution_queue, initial_node)
    while solution_queue:
        current_node = heapq.heappop(solution_queue)  # Node with the lowest cost
        if is_leaf(current_node):
            if current_node.solution_cost < best_solution:
                best_solution = current_node.solution_cost
                best_solution_path = current_node.path
        else:
            for child_node in generate_children(current_node):
                child_node.lower_bound = one_tree_relaxation(child_node.graph)
                if child_node.lower_bound < best_solution:
                    heapq.heappush(solution_queue, child_node)

    return best_solution_path, best_solution

# Example usage
file_path = '/Users/terrylin/Desktop/2024 Winter/Iintroduction of AI/Project/code/tsp-problem-10-10-5-1-2.txt'  # Replace with your actual file path
matrix = load_matrix_from_file(file_path)
initial_node = Node([0], 0, matrix)
path, cost = BnB(initial_node)
print("Path:", path)
print("Cost:", cost)
end_time = time.time()
print("Time:", end_time - start_time)


