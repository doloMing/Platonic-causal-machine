import numpy as np
from scipy.stats import pearsonr, spearmanr
import auxiliary_libs as aux

def causal_structure_accuracy(B_actual: np.ndarray, B_discovered: np.ndarray) -> dict:
    """
    Calculate accuracy metrics for the discovered causal graph compared to the actual graph.

    Parameters
    ----------
    B_actual : np.ndarray
        The ground truth binary adjacency matrix of the causal graph.
    B_discovered : np.ndarray
        The estimated binary adjacency matrix of the causal graph.

    Returns
    -------
    dict
        A dictionary containing accuracy metrics: fdr (False Discovery Rate), 
        tpr (True Positive Rate), fpr (False Positive Rate), shd (Structural Hamming Distance), 
        nnz (Number of Non-Zero edges), and f1_score.
    """
    # Check if the discovered graph contains undirected edges (represented by -1)
    if (B_discovered == -1).any():  # cpdag
        # Ensure that discovered edges only take values in {0, 1, -1}
        if not ((B_discovered == 0) | (B_discovered == 1) | (B_discovered == -1)).all():
            raise ValueError('B_discovered should take value in {0,1,-1}, please correct it.')
        # Ensure that undirected edges appear only once in the adjacency matrix
        if ((B_discovered == -1) & (B_discovered.T == -1)).any():
            raise ValueError('Undirected edge should only appear once, please correct it.')
    else:  # If the discovered graph is a directed acyclic graph (DAG)
        # Ensure that discovered edges only take values in {0, 1}
        if not ((B_discovered == 0) | (B_discovered == 1)).all():
            raise ValueError('B_discovered should take value in {0,1}, please correct it.')
        # Check if the discovered graph is a valid DAG
        if not aux.is_dag(B_discovered):
            raise ValueError('B_discovered should be a DAG, please correct it.')

    d = B_actual.shape[0]  # Get the number of nodes in the graph

    # Identify the linear indices of predicted undirected and directed edges
    pred_und = np.flatnonzero(B_discovered == -1)  # Indices of undirected edges
    pred = np.flatnonzero(B_discovered == 1)  # Indices of directed edges
    cond = np.flatnonzero(B_actual)  # Indices of actual edges
    cond_reversed = np.flatnonzero(B_actual.T)  # Indices of actual edges in the transposed graph
    cond_skeleton = np.concatenate([cond, cond_reversed])  # Combine actual edges and their reverses

    # Calculate true positives (TP)
    true_pos = np.intersect1d(pred, cond, assume_unique=True)  # Correctly predicted directed edges
    # Treat undirected edges favorably by considering them as true positives
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)  # Correctly predicted undirected edges
    true_pos = np.concatenate([true_pos, true_pos_und])  # Combine true positives from directed and undirected edges

    # Calculate false positives (FP)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)  # Incorrectly predicted directed edges
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)  # Incorrectly predicted undirected edges
    false_pos = np.concatenate([false_pos, false_pos_und])  # Combine false positives from directed and undirected edges

    # Calculate reverse edges
    extra = np.setdiff1d(pred, cond, assume_unique=True)  # Edges predicted but not in the actual graph
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)  # Reverse edges that were incorrectly predicted

    # Compute ratios for accuracy metrics
    pred_size = len(pred) + len(pred_und)  # Total number of predicted edges (both directed and undirected)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)  # Total number of non-edges in the actual graph
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)  # False Discovery Rate
    tpr = float(len(true_pos)) / max(len(cond), 1)  # True Positive Rate
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)  # False Positive Rate

    # Calculate Structural Hamming Distance (SHD)
    pred_lower = np.flatnonzero(np.tril(B_discovered + B_discovered.T))  # Lower triangular indices of predicted edges
    cond_lower = np.flatnonzero(np.tril(B_actual + B_actual.T))  # Lower triangular indices of actual edges
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)  # Extra edges in the prediction
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)  # Missing edges in the prediction
    shd = len(extra_lower) + len(missing_lower) + len(reverse)  # Total SHD

    # Calculate precision, recall, and F1 score
    precision = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0.0  # Precision
    recall = tpr  # Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # F1 Score

    # Return all calculated metrics in a dictionary
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 'f1_score': f1_score}


def causal_structure_consistency(W_actual: np.ndarray, W_discovered: np.ndarray, method: str = 'pearson') -> dict:
    """
    Compare the correlation of non-zero elements in the actual and discovered causal graphs.

    Parameters
    ----------
    W_actual : np.ndarray
        Ground truth adjacency matrix of the causal graph.
    W_discovered : np.ndarray
        Estimated adjacency matrix of the causal graph.
    method : str
        Method for correlation calculation: 'pearson' or 'spearman'.

    Returns
    -------
    dict
        Dictionary containing correlation metrics: correlation coefficient and p-value.
    """
    # Get the positions of non-zero elements in both matrices
    actual_positions = np.argwhere(W_actual != 0)
    discovered_positions = np.argwhere(W_discovered != 0)

    # Combine the positions to find unique (i, j) pairs
    combined_positions = np.unique(np.vstack((actual_positions, discovered_positions)), axis=0)

    # Extract the corresponding elements from both matrices
    actual_edges = []
    discovered_edges = []

    for i, j in combined_positions:
        actual_edges.append(W_actual[i, j])
        discovered_edges.append(W_discovered[i, j])

    # Convert lists to numpy arrays
    actual_edges = np.array(actual_edges)
    discovered_edges = np.array(discovered_edges)

    # Check if the arrays have enough variation
    if np.all(actual_edges == actual_edges[0]) or np.all(discovered_edges == discovered_edges[0]):
        return {
            'correlation_coefficient': np.nan,
            'p_value': np.nan
        }

    # Calculate correlation based on the specified method
    if method == 'pearson':
        correlation_coefficient, p_value = pearsonr(actual_edges, discovered_edges)
    elif method == 'spearman':
        correlation_coefficient, p_value = spearmanr(actual_edges, discovered_edges)
    else:
        raise ValueError("Unsupported method. Use 'pearson' or 'spearman'.")

    return {
        'correlation_coefficient': correlation_coefficient,
        'p_value': p_value
    }
