import numpy as np
import igraph as ig
import typing
from scipy.special import expit as sigmoid

def dag_generator(d: int, e_num: int, graph_type: str, w_ranges: typing.List[typing.Tuple[float, float]] = ((-2.0, -0.5), (0.5, 2.0))) -> np.ndarray:
    r"""
    Simulate a random Directed Acyclic Graph (DAG) with a specified expected number of edges.

    Parameters
    ----------
    d : int
        The number of nodes in the graph.
    e_num : int
        The expected number of edges in the graph.
    graph_type : str
        The type of graph to generate. Options are ``["ER", "SF", "BP"]``.
    w_ranges : typing.List[typing.Tuple[float, float]], optional
            Disjoint weight ranges for the parameters. By default, it is set to 
            :math:`((-2.0, -0.5), (0.5, 2.0))`.
    
    Returns
    -------
    numpy.ndarray
        A binary adjacency matrix of the generated DAG with shape :math:`(d, d)`.
    """
    
    def _random_permutation(M):
        """
        Generate a random permutation of the input matrix M.

        Parameters
        ----------
        M : np.ndarray
            The input matrix to be permuted.

        Returns
        -------
        np.ndarray
            The permuted matrix.
        """
        # np.random.permutation permutes the first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        """
        Randomly orient the edges of an undirected graph to ensure acyclicity.

        Parameters
        ----------
        B_und : np.ndarray
            The undirected adjacency matrix.

        Returns
        -------
        np.ndarray
            The acyclic oriented adjacency matrix.
        """
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        """
        Convert an igraph graph object to a NumPy adjacency matrix.

        Parameters
        ----------
        G : ig.Graph
            The igraph graph object.

        Returns
        -------
        np.ndarray
            The adjacency matrix of the graph.
        """
        return np.array(G.get_adjacency().data)
    
    def _weight_assignment(B, w_ranges):
        r"""
        Simulate Structural Equation Model (SEM) parameters for a Directed Acyclic Graph (DAG).

        Parameters
        ----------
        B : np.ndarray
            A binary adjacency matrix of the DAG with shape :math:`[d, d]`.
        w_ranges : typing.List[typing.Tuple[float, float]], optional
            Disjoint weight ranges for the parameters. By default, it is set to 
            :math:`((-2.0, -0.5), (0.5, 2.0))`.

        Returns
        -------
        np.ndarray
            A weighted adjacency matrix of the DAG with shape :math:`[d, d]`.
        """
        W = np.zeros(B.shape)  # Initialize the weighted adjacency matrix with zeros
        S = np.random.randint(len(w_ranges), size=B.shape)  # Randomly select which weight range to use for each entry

        # Iterate over the specified weight ranges
        for i, (low, high) in enumerate(w_ranges):
            # Generate random weights within the specified range
            U = np.random.uniform(low=low, high=high, size=B.shape)
            # Assign weights to the edges based on the binary adjacency matrix B
            W += B * (S == i) * U  # Only assign weights where B is 1 and the selected range matches

        return W  # Return the weighted adjacency matrix

    # Generate the graph based on the specified type
    if graph_type == 'ER':
        # Erdos-Renyi graph generation
        G_und = ig.Graph.Erdos_Renyi(n=d, m=e_num)  # Create an undirected Erdos-Renyi graph
        B_und = _graph_to_adjmat(G_und)  # Convert to adjacency matrix
        B = _random_acyclic_orientation(B_und)  # Randomly orient edges to ensure acyclicity
    elif graph_type == 'SF':
        # Scale-free graph generation using Barabasi-Albert model
        G = ig.Graph.Barabasi(n=d, m=int(round(e_num / d)), directed=True)  # Create a directed scale-free graph
        B = _graph_to_adjmat(G)  # Convert to adjacency matrix
    elif graph_type == 'BP':
        # Bipartite graph generation
        top = int(0.2 * d)  # Number of nodes in the top partition
        G = ig.Graph.Random_Bipartite(top, d - top, m=e_num, directed=True, neimode=ig.OUT)  # Create a directed bipartite graph
        B = _graph_to_adjmat(G)  # Convert to adjacency matrix
    else:
        raise ValueError('Receive unknown graph type, please correct it.')  # Raise error for unknown graph type

    # Randomly permute the adjacency matrix
    B = _random_permutation(B)
    
    # Ensure the generated graph is a DAG
    assert ig.Graph.Adjacency(B.tolist()).is_dag()

    ## Assign edge weights for this DAG
    W = _weight_assignment(B, w_ranges)
    
    return B, W  # Return the generated DAG via its binary and weighted adjacency matrix


def sem_over_dag(W: np.ndarray, 
                 n: int, 
                 sem_type: str, 
                 noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None) -> np.ndarray:
    r"""
    Simulate samples from a Structural Equation Model (SEM) over a Directed Acyclic Graph (DAG).

    This function can simulate both linear and nonlinear SEMs based on the specified type.

    Parameters
    ----------
    W : np.ndarray
        A weighted adjacency matrix of the DAG with shape :math:`[d, d]`.
    n : int
        The number of samples to generate. When ``n=inf``, it mimics the population risk, 
        only applicable for Gaussian noise.
    sem_type : str
        The type of SEM to simulate. Options are elements in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson'] for linear SEM and 
        belong to ['mlp', 'mim', 'gp', 'gp-add'] for nonlinear SEM.
    noise_scale : typing.Optional[typing.Union[float, typing.List[float]]], optional
        Scale parameter of the additive noises. If ``None``, all noises have scale 1. 
        Default is ``None``.

    Returns
    -------
    np.ndarray
        A sample matrix with shape :math:`[n, d]` for finite samples, or :math:`[d, d]` if ``n=inf``.
    """
    
    def _simulate_single_equation(X, w, scale):
        """Simulate a single equation for linear SEM."""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('Receive unknown SEM type, please correct it.')
        return x

    def _simulate_single_nonlinear_equation(X, scale):
        """Simulate a single equation for nonlinear SEM."""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('Receive unknown SEM type, please correct it.')
        return x
    
    # Ensure noise_scale is treated as a list
    if noise_scale is None:
        noise_scale = [1.0] * W.shape[0]  # Default to 1.0 for all nodes
    elif isinstance(noise_scale, (float, int)):
        noise_scale = [noise_scale] * W.shape[0]  # Convert single float to list

    d = W.shape[0]  # Number of nodes in the graph
    scale_vec = noise_scale if noise_scale else np.ones(d)  # Set noise scale
    X = np.zeros([n, d])  # Initialize sample matrix
    G = ig.Graph.Weighted_Adjacency(W.tolist())  # Create a graph from the weighted adjacency matrix
    ordered_vertices = G.topological_sorting()  # Get the topological order of the vertices
    assert len(ordered_vertices) == d  # Ensure the number of ordered vertices matches the number of nodes

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)  # Get the parents of the current node
        if sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])  # Simulate linear SEM
        elif sem_type in ['mlp', 'mim', 'gp', 'gp-add']:
            X[:, j] = _simulate_single_nonlinear_equation(X[:, parents], scale_vec[j])  # Simulate nonlinear SEM
        else:
            raise ValueError('Receive unknown SEM type, please correct it.')

    return X  # Return the generated sample matrix


def simulation_func(d: int, e_num: int, graph_type: str, sem_type: str, n: int, 
                    w_ranges: typing.List[typing.Tuple[float, float]] = ((-2.0, -0.5), (0.5, 2.0)), 
                    noise_scale: typing.Optional[typing.Union[float, typing.List[float]]] = None) -> tuple:
    r"""
    Simulate a Directed Acyclic Graph (DAG) and generate samples from a Structural Equation Model (SEM).

    This function first generates a random DAG and then simulates samples based on the specified SEM type.

    Parameters
    ----------
    d : int
        The number of nodes in the graph.
    e_num : int
        The expected number of edges in the graph.
    graph_type : str
        The type of graph to generate. Options are ``["ER", "SF", "BP"]``.
    sem_type : str
        The type of SEM to simulate. Options are ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson'] (linear SEM) and ['mlp', 'mim', 'gp', 'gp-add'] (non-linear SEM).
    n : int
        The number of samples to generate.
    w_ranges : typing.List[typing.Tuple[float, float]], optional
        Disjoint weight ranges for the parameters. By default, it is set to 
        :math:`((-2.0, -0.5), (0.5, 2.0))`.
    noise_scale : typing.Optional[typing.Union[float, typing.List[float]]], optional
        Scale parameter of the additive noises. If ``None``, all noises have scale 1. 
        Default is ``None``.

    Returns
    -------
    tuple
        A tuple containing:
        - B : np.ndarray
            The binary adjacency matrix of the generated DAG.
        - W : np.ndarray
            The weighted adjacency matrix of the generated DAG.
        - X : np.ndarray
            The sample matrix generated from the SEM.
    """
    # Generate the DAG and its weighted adjacency matrix
    B, W = dag_generator(d, e_num, graph_type, w_ranges)
    
    # Generate samples from the SEM over the generated DAG
    X = sem_over_dag(W, n, sem_type, noise_scale)
    
    return B, W, X  # Return the generated binary adjacency matrix, weighted adjacency matrix, and sample matrix

if __name__ == '__main__':
    # Test various configurations for data generation
    test_cases = [
        {
            "d": 5,  # Number of nodes
            "e_num": 4,  # Expected number of edges
            "graph_type": "ER",  # Erdos-Renyi graph
            "sem_type": "gauss",  # Change to a recognized SEM type
            "n": 100,  # Number of samples
            "w_ranges": [(-2.0, -0.5), (0.5, 2.0)],  # Weight ranges
            "noise_scale": None  # Default noise scale
        },
        {
            "d": 5,
            "e_num": 4,
            "graph_type": "SF",  # Scale-free graph
            "sem_type": "mlp",  # Change to a recognized SEM type
            "n": 100,
            "w_ranges": [(-1.0, 1.0)],  # Different weight range
            "noise_scale": 1.0  # Specific noise scale
        },
        {
            "d": 6,
            "e_num": 5,
            "graph_type": "BP",  # Bipartite graph
            "sem_type": "exp",  # Use a recognized SEM type
            "n": 50,
            "w_ranges": [(-1.5, -0.5), (0.5, 1.5)],
            "noise_scale": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Different noise scales for each node
        },
        {
            "d": 6,
            "e_num": 5,
            "graph_type": "BP",  # Fully connected graph
            "sem_type": "mlp",  # Use a recognized SEM type
            "n": 200,
            "w_ranges": [(-0.7, 0.0), (0.1, 2.0)],
            "noise_scale": 0
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"Running test case {i + 1}:")
        B, W, X = simulation_func(
            d=test_case["d"],
            e_num=test_case["e_num"],
            graph_type=test_case["graph_type"],
            sem_type=test_case["sem_type"],
            n=test_case["n"],
            w_ranges=test_case["w_ranges"],
            noise_scale=test_case["noise_scale"]
        )
        
        # Output the results
        print("Binary Adjacency Matrix (B):")
        print(B)
        print("Weighted Adjacency Matrix (W):")
        print(W)
        print("Sample Matrix (X):")
        print(X)
        print("\n" + "="*50 + "\n")  # Separator for readability