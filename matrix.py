def load_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  # Read the first line to get n and remove any leading/trailing whitespace
        matrix = [list(map(float, line.strip().split())) for line in file]  # Read and process each subsequent line  
    return matrix

file_path = '/Users/terrylin/Desktop/2024 Winter/Iintroduction of AI/Project/code/tsp-problem-10-10-5-1-1.txt'  # Replace with your actual file path
matrix = load_matrix_from_file(file_path)
print(matrix)