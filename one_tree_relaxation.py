import networkx as nx
import numpy as np

def load_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  # Read the first line to get n and remove any leading/trailing whitespace
        matrix = [list(map(float, line.strip().split())) for line in file]  # Read and process each subsequent line  
    return matrix

def select_root_node():
    # 示例中选择0作为根节点
    # 实际选择可能基于特定逻辑
    return 0

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
    root_node = select_root_node()
    mst_cost = find_minimum_spanning_tree_cost(graph, root_node)
    two_shortest_edges_cost = find_two_shortest_edges_to_root(graph, root_node)
    return mst_cost + two_shortest_edges_cost

# Example usage
file_path = '/Users/terrylin/Desktop/2024 Winter/Iintroduction of AI/Project/code/tsp-problem-15-10-5-1-1.txt'  # Replace with your actual file path
matrix = load_matrix_from_file(file_path)
relaxation = one_tree_relaxation(matrix)
print("One Tree Relaxation:", relaxation)
