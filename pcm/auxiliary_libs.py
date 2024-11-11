import numpy as np
import igraph as ig
import random
import torch
import psutil  

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    print('Random seed is controlled.')

def is_dag(W: np.ndarray) -> bool:
    """
    Check if the given adjacency matrix represents a Directed Acyclic Graph (DAG).

    Parameters:
    - W (np.ndarray): A square numpy array representing the adjacency matrix of a graph.
                      The shape of W should be (n, n), where n is the number of nodes.

    Returns:
    - bool: Returns True if the graph represented by the adjacency matrix W is a DAG,
            otherwise returns False.

    Description:
    This function converts the input adjacency matrix W into a weighted graph using the igraph library.
    It then checks if the graph is acyclic (i.e., it does not contain any directed cycles).
    A directed graph is considered acyclic if there are no paths that start and end at the same vertex
    while following the direction of the edges.
    """
    # Convert the adjacency matrix to a list format for igraph
    adjacency_list = W.tolist()
    
    # Create a weighted graph from the adjacency matrix
    G = ig.Graph.Weighted_Adjacency(adjacency_list)
    
    # Check if the graph is a Directed Acyclic Graph (DAG)
    return G.is_dag()

def batch_generator(X: torch.Tensor, batch_size: int, batch_num: int, continum: bool) -> torch.Tensor:
    """
    Generate batches of samples from the input tensor X.

    Parameters:
    - X (torch.Tensor): An n*d tensor where n is the number of samples and d is the number of features.
    - batch_size (int): The number of samples in each batch.
    - batch_num (int): The number of batches to generate.
    - continum (bool): If True, samples are taken from contiguous rows; if False, samples are taken randomly.

    Returns:
    - torch.Tensor: A tensor of shape (batch_num, batch_size, d) containing the generated batches.
    """
    n, d = X.shape  # Get the dimensions of X
    batches = torch.zeros((batch_num, batch_size, d), dtype=X.dtype)  # Initialize the output tensor

    for i in range(batch_num):
        if continum:
            # Randomly select a starting index for contiguous samples
            start_index = random.randint(0, n - batch_size)
            batches[i] = X[start_index:start_index + batch_size]  # Get contiguous rows
        else:
            # Randomly select batch_size indices from the range of n
            indices = random.sample(range(n), batch_size)
            batches[i] = X[indices]  # Get non-contiguous rows

    return batches  # Return the generated batches

def check_memory_usage(threshold=0.8):
    """
    Check the current memory usage of the process and return True if it exceeds the threshold.
    
    Parameters:
    - threshold: A float representing the memory usage threshold (0 to 1).
    
    Returns:
    - bool: True if memory usage exceeds the threshold, False otherwise.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    # Get the total memory and used memory
    total_memory = psutil.virtual_memory().total
    used_memory = mem_info.rss  # Resident Set Size
    memory_usage = used_memory / total_memory
    return memory_usage > threshold

class ensemble_options:
    def __init__(self, use_parallel: bool = False, mem_threshold: float = 0.8,
                 batch_size: int = 32, batch_num: int = 10, continuum: bool = True):
        """
        Class to hold hyperparameters for the ensemble function.

        Parameters
        ----------
        use_parallel : bool
            Indicates whether to use parallel processing. Default is False.
        mem_threshold : float
            Memory usage threshold (0 to 1). Must be greater than 0 and less than or equal to 1.
        batch_size : int
            The number of samples in each batch. Default is 32.
        batch_num : int
            The number of batches to generate. Default is 10.
        continuum : bool
            If True, samples are taken from contiguous rows; if False, samples are taken randomly. Default is True.
        """
        if not (0 < mem_threshold <= 1):
            raise ValueError("mem_threshold must be between 0 (exclusive) and 1 (inclusive).")
        
        self.use_parallel = use_parallel  # Store whether to use parallel processing
        self.mem_threshold = mem_threshold  # Store the memory threshold
        self.batch_size = batch_size  # Store the batch size
        self.batch_num = batch_num  # Store the number of batches
        self.continuum = continuum  # Store whether to use contiguous samples

def compute_expectation(all_sub_matrices, use_masked_expectation=False):
    """
    Compute the expectation of matrices in all_sub_matrices.
    
    Parameters:
    - all_sub_matrices: A list of numpy arrays, each representing a matrix.
    - use_masked_expectation: A boolean indicating whether to use masked expectation.
    
    Returns:
    - A numpy array representing the expectation matrix, with the same shape as the matrices in all_sub_matrices.
    """
    if not all_sub_matrices:
        raise ValueError("The input list all_sub_matrices is empty.")
    
    # Initialize the expectation matrix with zeros, having the same shape as the first matrix in all_sub_matrices
    expectation_matrix = np.zeros_like(all_sub_matrices[0])
    
    # Sum all matrices in all_sub_matrices
    for matrix in all_sub_matrices:
        expectation_matrix += matrix
    
    # Divide by the number of matrices to get the average
    expectation_matrix /= len(all_sub_matrices)
    
    if use_masked_expectation:
        # Find the minimum percentage threshold n that ensures the expectation_matrix is a DAG
        n = 50  # Start with 50%
        found_dag = True

        mask = np.zeros_like(all_sub_matrices[0], dtype=int)
        for matrix in all_sub_matrices:
            mask += (matrix != 0).astype(int)  # Increment mask where matrix is non-zero
        
        while n > 0 and found_dag:
            # Set sel_mask entries to 1 if they exceed n% of the number of matrices
            threshold_count = (len(all_sub_matrices) * n) / 100
            sel_mask = (mask > threshold_count).astype(int)
            
            # Apply the mask to the expectation matrix
            masked_expectation_matrix = expectation_matrix * sel_mask
            
            # Check if the masked expectation matrix is a DAG
            found_dag = is_dag(masked_expectation_matrix != 0)
            n -= 1

            if n <= 0:
                raise ValueError("No valid percentage found that results in a DAG.")

        optimal_n = n + 2
        threshold_count = (len(all_sub_matrices) * optimal_n) / 100
        optimal_mask = (mask > threshold_count).astype(int)
        expectation_matrix *= optimal_mask

    return expectation_matrix



def conformal_inference(adjacency_matrices, confidence_level=0.95):
    """
    Aggregate multiple weighted adjacency matrices to construct a significant causal relationship matrix.
    
    Parameters:
    - adjacency_matrices (np.ndarray): A 3D array of shape (k, d, d), where k is the number of bootstrap samples and d is the number of nodes.
    - confidence_level (float): The significance level used to determine which edges are retained in the aggregated matrix.
    
    Returns:
    - aggregated_matrix (np.ndarray): The aggregated weighted adjacency matrix of shape (d, d).
    """
    # Get the dimensions
    k = len(adjacency_matrices)  # Number of bootstrap samples
    d = adjacency_matrices[0].shape[0]

    # Calculate the median value for each edge across the bootstrap samples
    median_matrix = np.median(adjacency_matrices, axis=0)
    
    # Calculate the deviation matrix for each edge (deviation from the median)
    deviations = np.abs(adjacency_matrices - median_matrix)
    
    # Sort the deviations for each edge across bootstrap samples and set the deviation threshold based on confidence_level
    deviation_thresholds = np.percentile(deviations, q=confidence_level*100, axis=0)

    # Construct the aggregated matrix: only edges with deviations below the threshold are retained
    aggregated_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if np.mean(deviations[:, i, j] <= deviation_thresholds[i, j]) >= confidence_level:
                aggregated_matrix[i, j] = median_matrix[i, j]
    
    return aggregated_matrix

if __name__ == '__main__':
    k = 100  # Number of bootstrap samples
    d = 5    # Number of variables
    # Generate sample adjacency matrices as bootstrap examples
    adjacency_matrices = [np.random.rand(d, d) for _ in range(k)]

    # Compute conformal inference-based summary matrix
    final_matrix = conformal_inference(adjacency_matrices, confidence_level=0.99)

    print("Aggregated adjacency matrix:")
    print(final_matrix)
