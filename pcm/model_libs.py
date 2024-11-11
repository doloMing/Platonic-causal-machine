import torch
from tqdm.auto import tqdm
import typing
import numpy as np
import eval_libs as eva
import auxiliary_libs as aux
import simulation_libs as sim


class fit_options:
    def __init__(self, lambda_v: float = 0.01, w_threshold: float = 0.3, T: int = 5,
                 mu_init: float = 1.0, mu_factor: float = 0.1,
                 s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6],
                iter_num: int = 6000, checkpoint: int = 100,
                 loss_type: str = 'l2'):  # Add loss_type parameter
        """
        Class to hold hyperparameters for the fit function.

        Parameters
        ----------
        lambda_v : float
            The regularization parameter. Default is 0.01.
        w_threshold : float
            Threshold for setting small weights to zero. Default is 0.3.
        T : int
            Number of iterations for the outer loop. Default is 5.
        mu_init : float
            Initial value for mu. Default is 1.0.
        mu_factor : float
            Factor by which to reduce mu after each outer iteration. Default is 0.1.
        s : typing.Union[typing.List[float], float]
            Scalar or list of scalars for the constraint. Default is [1.0, .9, .8, .7, .6].
        iter_num : int
            The number of iterations for optimization. Default is 6000.
        checkpoint : int
            Frequency of printing progress. Default is 100.
        loss_type : str
            Specifies the type of loss function to use. Options are 'l2' for least squares loss
            and 'logistic' for logistic loss. Default is 'l2'.
        """
        self.lambda_v = lambda_v
        self.w_threshold = w_threshold
        self.T = T
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.s = s
        self.iter_num = iter_num
        self.checkpoint = checkpoint
        self.loss_type = loss_type  # Store the loss type

class pcm_unit:
    def __init__(self, loss_type: str, dtype: torch.dtype = torch.float64):
        """
        Initializes the pcm_unit class.

        Parameters
        ----------
        loss_type : str
            Specifies the type of loss function to use. Options are 'l2' for least squares loss
            and 'logistic' for logistic loss.
        dtype : torch.dtype, optional
            Specifies the data type for tensors. Default is torch.float64.
        """
        super().__init__()
        loss_types = ['l2', 'logistic']  # Define acceptable loss types
        assert loss_type in loss_types, f"loss_type should be one of {loss_types}"  # Ensure valid loss type
        self.loss_type = loss_type  # Store the loss type
        self.dtype = dtype  # Store the data type
    
    def score_func(self, W: torch.Tensor):
        """
        Computes the score (loss) based on the specified loss type.

        Parameters
        ----------
        W : torch.Tensor
            The weight matrix to be optimized.

        Returns
        -------
        loss : torch.Tensor
            The computed loss value.
        """
        if self.loss_type == 'l2':
            R = self.X @ W  # Compute the predicted values
            loss = 0.5 * torch.norm(self.X - R) ** 2  # Calculate L2 loss
        elif self.loss_type == 'logistic':
            R = self.X @ W  # Compute the predicted values
            # Calculate logistic loss using log-sum-exp trick for numerical stability
            loss = (1.0 / self.n) * (torch.logsumexp(R, dim=1) - (self.X * R).sum())
        return loss

    def constraint_func(self, W: torch.Tensor, s: float = 1.0):
        """
        Computes the constraint value based on the weight matrix.

        Parameters
        ----------
        W : torch.Tensor
            The weight matrix.
        s : float, optional
            A scalar value used in the constraint calculation. Default is 1.0.

        Returns
        -------
        constraint : torch.Tensor
            The computed constraint value.
        """
        M = s * torch.eye(self.d, dtype=self.dtype) - W * W  # Compute the matrix M
        constraint = -torch.logdet(M) + self.d * torch.log(torch.tensor(s, dtype=self.dtype))  # Calculate the constraint
        return constraint

    def regularization_func(self, W: torch.Tensor):
        """
        Computes the regularization term based on the weight matrix.

        Parameters
        ----------
        W : torch.Tensor
            The weight matrix.

        Returns
        -------
        reg : torch.Tensor
            The computed regularization value.
        """
        reg = self.lambda_v * W.abs().sum()  # L1 regularization
        return reg
    
    def loss_func(self, W: torch.Tensor, mu: float, s: float = 1.0):
        """
        Computes the total loss, including score, constraint, and regularization.

        Parameters
        ----------
        W : torch.Tensor
            The weight matrix.
        mu : float
            A scalar that weights the score and regularization.
        s : float, optional
            A scalar value used in the constraint calculation. Default is 1.0.

        Returns
        -------
        loss : torch.Tensor
            The total computed loss.
        score : torch.Tensor
            The score (loss) value.
        constraint : torch.Tensor
            The constraint value.
        reg : torch.Tensor
            The regularization value.
        """
        score = self.score_func(W)  # Calculate the score
        constraint = self.constraint_func(W, s)  # Calculate the constraint
        reg = self.regularization_func(W)  # Calculate the regularization
        loss = mu * (score + reg) + constraint  # Total loss
        return loss, score, constraint, reg

    def optimization(self, W: torch.Tensor, mu: float, i: int, iter_num: int, s: float, lr: float, tol: float = 1e-3, pbar = None):
        """
        Optimizes the weight matrix W using gradient descent.

        Parameters
        ----------
        W : torch.Tensor
            The weight matrix to be optimized.
        mu : float
            A scalar that weights the score and regularization.
        i : int
            The index of inner iteration
        iter_num : int
            The number of iterations for optimization.
        s : float
            A scalar value used in the constraint calculation.
        lr : float
            The learning rate for the optimizer.
        tol : float, optional
            The tolerance for convergence. Default is 1e-3.
        pbar : tqdm, optional
            Progress bar for tracking optimization progress.

        Returns
        -------
        W : torch.Tensor
            The optimized weight matrix.
        success : bool
            Indicates whether the optimization was successful.
        """
        optimizer = torch.optim.Adam([W], lr=lr)  # Initialize the Adam optimizer

        for iter in range(iter_num):
            optimizer.zero_grad()  # Clear previous gradients
            loss, score, constraint, reg = self.loss_func(W, mu, s)  # Compute loss and other metrics
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update the weights

            # Print progress and check for convergence
            if iter % self.checkpoint == 0 or iter == iter_num - 1:
                print(f'\nIteration {i * iter_num + iter}: Loss = {loss.item()}, score = {score:.3e}, constraint = {constraint:.3e}, reg = {reg:.3e}')
                if (torch.abs(loss) < tol) or (constraint < 1e-10):  # Check for convergence
                    pbar.update(iter_num - iter + 1)  # Update progress bar
                    break
            pbar.update(1)  # Update progress bar
        return W, True  # Return optimized weights and success flag

    def compute_jacobians(self, X: torch.Tensor, W: torch.Tensor, mu: float, s: float):
        """
        Computes the Jacobian matrices for each component of the loss.

        Parameters
        ----------
        X : torch.Tensor
            The input data matrix.
        W : torch.Tensor
            The weight matrix.
        mu : float
            A scalar that weights the score and regularization.
        s : float
            A scalar value used in the constraint calculation.

        Returns
        -------
        jacobians : dict
            A dictionary containing the Jacobian matrices for each component of the loss.
        """
        # Enable gradient tracking for X
        X.requires_grad = True
        
        # Compute the total loss
        loss, _, _, _ = self.loss_func(W, mu, s)
        
        # Compute the Jacobian of the total loss with respect to X
        loss.backward()  # Backpropagate to compute gradients
        loss_jacobian = X.grad.clone()  # Jacobian of the total loss
        
        # Clear gradients for the next computation
        X.grad.zero_()
        
        # Compute score Jacobian
        score = self.score_func(W)
        score.backward()  # Backpropagate for score
        score_jacobian = X.grad.clone()  # Jacobian of the score
        X.grad.zero_()  # Clear gradients

        # Compute constraint Jacobian
        constraint = self.constraint_func(W, s)
        constraint.backward()  # Backpropagate for constraint
        constraint_jacobian = X.grad.clone()  # Jacobian of the constraint
        X.grad.zero_()  # Clear gradients

        # Compute regularization Jacobian
        reg = self.regularization_func(W)
        reg.backward()  # Backpropagate for regularization
        reg_jacobian = X.grad.clone()  # Jacobian of the regularization
        X.grad.zero_()  # Clear gradients
        
        # Store all Jacobians in a dictionary
        jacobians = {
            'loss_jacobian': loss_jacobian,
            'score_jacobian': score_jacobian,
            'constraint_jacobian': constraint_jacobian,
            'reg_jacobian': reg_jacobian
        }
        
        return jacobians

    def fit(self, X: torch.Tensor, opt: fit_options):
        """
        Fits the model to the input data X.

        Parameters
        ----------
        X : torch.Tensor
            The input data matrix.
        opt : fit_options
            An instance of fit_options containing hyperparameters for fitting.

        Returns
        -------
        W_est : numpy.ndarray
            The estimated weight matrix after fitting.
        """

        self.X, self.lambda_v, self.checkpoint = X, opt.lambda_v, opt.checkpoint  # Store input data and parameters
        self.n, self.d = X.shape  # Get the dimensions of the input data
        
        mu = opt.mu_init  # Initialize mu
        lr = 0.001  # Set learning rate
        
        # Handle the constraint parameter s
        if isinstance(opt.s, list):
            if len(opt.s) < opt.T: 
                print(f"Length of s is {len(opt.s)}, using last value in s for iteration t >= {len(opt.s)}")
                opt.s = opt.s + (opt.T - len(opt.s)) * [opt.s[-1]]  # Extend s to match T
        elif isinstance(opt.s, (int, float)):
            opt.s = [opt.s] * opt.T  # Convert scalar to list
        else:
            raise ValueError("s should be a list, int, or float.")
        
        W = torch.zeros((self.d, self.d), dtype=self.dtype, requires_grad=True)  # Initialize weight matrix

        with tqdm(total=opt.T * opt.iter_num) as pbar:  # Initialize progress bar
            for i in range(int(opt.T)):
                success = False
                inner_iters = int(opt.iter_num)   # Set number of inner iterations
                while not success:
                    W, success = self.optimization(W, mu, i, inner_iters, opt.s[i], lr=lr, pbar=pbar)  # Optimize weights
                    if not success:
                        print(f'Retrying with larger s.')  # If optimization fails, increase s
                        opt.s[i] += 0.1
                mu *= opt.mu_factor  # Reduce mu after each outer iteration
        
        # Final calculations after fitting
        self.score_final = self.score_func(W)  # Final score
        self.constraint_final = self.constraint_func(W, opt.s[-1])  # Final constraint
        self.reg_final = self.regularization_func(W)  # Final regularization

        self.W_est = W.detach().numpy()  # Convert optimized weights to NumPy array
        self.W_est[np.abs(self.W_est) < opt.w_threshold] = 0  # Set small weights to zero based on threshold

        # Compute Jacobians for each component of the loss
        jacobians = self.compute_jacobians(X, W, mu, opt.s[-1])

        return self.W_est, jacobians  # Return estimated weights


def test():
    """
    Test function to evaluate the pcm_unit model.

    Generates synthetic data, fits the model, and prints the results.
    """
    from timeit import default_timer as timer
    
    n, d, e_num = 500, 100, 50  # Define parameters for synthetic data
    graph_type, sem_type = 'ER', 'gauss'  # Define graph and SEM types

    aux.set_random_seed(1)
    B_true, W_true, X = sim.simulation_func(
            d=d,
            e_num=e_num,
            graph_type=graph_type,
            sem_type=sem_type,
            n=n,
            w_ranges=[(-1, -0.2), (0.1, 1.0)],
            noise_scale=None
        )
    
    X_tensor = torch.tensor(X, dtype=torch.float64)  # Ensure X is a PyTorch tensor
    start = timer()  # Start timer
    options = fit_options(lambda_v=0.01, w_threshold=0.3, T=8, iter_num=1000, checkpoint=100, mu_factor=0.1, loss_type='l2')
    model = pcm_unit(loss_type=options.loss_type)  # Initialize the model with L2 loss
    W_est, jacobians = model.fit(X_tensor, options)  # Fit the model
    end = timer()  # End timer
    print(W_true)  # Print true weights
    print(W_est)  # Print estimated weights
    acc = eva.causal_structure_accuracy(B_true, W_est != 0)  # Calculate accuracy
    print("Causal Structure Accuracy:")
    print(acc)  # Print accuracy

    # Test causal_structure_consistency with Pearson method
    consistency_metrics_pearson = eva.causal_structure_consistency(W_true, W_est, method='pearson')
    print("Causal Structure Consistency (Pearson):")
    print(consistency_metrics_pearson)
    # Test causal_structure_consistency with Spearman method
    consistency_metrics_spearman = eva.causal_structure_consistency(W_true, W_est, method='spearman')
    print("Causal Structure Consistency (Spearman):")
    print(consistency_metrics_spearman)
    print(f'time: {end - start:.4f}s')  # Print elapsed time

    # Print the Jacobians
    print("Jacobian of the entire loss:")
    print(jacobians['loss_jacobian'])
    print("Jacobian of the score:")
    print(jacobians['score_jacobian'])
    print("Jacobian of the constraint:")
    print(jacobians['constraint_jacobian'])
    print("Jacobian of the regularization:")
    print(jacobians['reg_jacobian'])
    
if __name__ == '__main__':
    test()  # Run the test function