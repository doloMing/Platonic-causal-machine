import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import psutil 
from tqdm import tqdm
import auxiliary_libs as aux
import model_libs as mod_libs

def process_single_batch(batch, fit_opt: mod_libs.fit_options, batch_index):
    """
    Process a single batch to perform DAG discovery and compute Jacobians.

    Parameters:
    - batch: A batch of input data.
    - fit_opt: An object containing various options and hyperparameters for the learning process.
    - batch_index: The index of the current batch.
    Returns:
    - A tuple containing the learned adjacency matrix and the computed Jacobians.
    """
    model = mod_libs.pcm_unit(loss_type=fit_opt.loss_type)  # Initialize the model with the specified loss type
    sub_W, sub_jacobians = model.fit(batch, fit_opt, batch_index)  # Fit the model to the batch
    
    return sub_W, sub_jacobians

def directed_acyclic_graph_learning(X: torch.Tensor, fit_opt, ens_opt):
    """
    Learn a Directed Acyclic Graph (DAG) from the input data X using the specified options.
    """
    # Initialize lists to store results
    W_est_list = []  # List to store learned adjacency matrices
    loss_jacobian_list = []  # List to store loss Jacobians
    score_jacobian_list = []  # List to store score Jacobians
    constraint_jacobian_list = []  # List to store constraint Jacobians
    reg_jacobian_list = []  # List to store regularization Jacobians

    # Generate batches from the input tensor X
    batches = aux.batch_generator(X, ens_opt.batch_size, ens_opt.batch_num, ens_opt.continuum)

    # Determine the maximum number of workers
    max_workers = psutil.cpu_count()
    optimal_workers = max_workers - 1  # Start with max_workers - 1

    while optimal_workers > 0:
        print(f"Trying with {optimal_workers} workers...")
        futures = []
        
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            for batch_index, batch in enumerate(batches):
                # Check memory usage before submitting a new batch
                if aux.check_memory_usage(threshold=ens_opt.mem_threshold):
                    print(f"Memory usage exceeded threshold with {optimal_workers} workers. Reducing to {optimal_workers - 1}.")
                    optimal_workers -= 1  # Reduce the number of workers
                    break  # Exit the for loop to re-evaluate worker count

                # Submit the batch for processing
                futures.append(executor.submit(process_single_batch, batch, fit_opt, batch_index))

            # Collect results from futures
            for future in tqdm(futures, desc="Collecting results"):
                try:
                    sub_W, sub_jacobians = future.result()
                    W_est_list.append(sub_W)
                    loss_jacobian_list.append(sub_jacobians['loss_jacobian'].detach().numpy())
                    score_jacobian_list.append(sub_jacobians['score_jacobian'].detach().numpy())
                    constraint_jacobian_list.append(sub_jacobians['constraint_jacobian'].detach().numpy())
                    reg_jacobian_list.append(sub_jacobians['reg_jacobian'].detach().numpy())
                except Exception as e:
                    print(f"Error processing batch: {e}")
                

        # Check memory usage after processing
        if aux.check_memory_usage(threshold=ens_opt.mem_threshold):
            print(f"Memory usage exceeded threshold with {optimal_workers} workers. Reducing to {optimal_workers - 1}.")
            optimal_workers -= 1  # Reduce workers if memory exceeds threshold
        else:
            break  # If memory usage is acceptable, exit the while loop

    # If we exit the loop without processing, we can return the results collected so far
    return W_est_list, loss_jacobian_list, score_jacobian_list, constraint_jacobian_list, reg_jacobian_list



if __name__ == '__main__':
    """
    Test function to evaluate the directed_acyclic_graph_learning function in framework_libs.

    Generates synthetic data, fits the model, and prints the results.
    """
    from timeit import default_timer as timer
    import eval_libs as eva
    import simulation_libs as sim
    
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

    # Set fit options and ensemble options
    fit_opt = mod_libs.fit_options(lambda_v=0.01, w_threshold=0.3, T=7, iter_num=1000, checkpoint=500, mu_factor=0.1, loss_type='l2')
    ens_opt = aux.ensemble_options(use_parallel=True, mem_threshold=0.8, batch_size=400, batch_num=50, continuum= True)

    # Run directed acyclic graph learning
    W_est_list, loss_jacobian_list, score_jacobian_list, constraint_jacobian_list, reg_jacobian_list = directed_acyclic_graph_learning(X_tensor, fit_opt, ens_opt)

    end = timer()  # End timer
    print(f'Time taken for learning: {end - start:.4f}s')  # Print elapsed time

    # Evaluate each estimated weight matrix
    for idx, W_est in enumerate(W_est_list):
        acc = eva.causal_structure_accuracy(B_true, W_est != 0)  # Calculate accuracy
        print(f"Accuracy for W_est[{idx}]: {acc}")

        # Calculate consistency metrics
        consistency_metrics_pearson = eva.causal_structure_consistency(W_true, W_est != 0, method='pearson')
        print(f"Causal Structure Consistency (Pearson) for W_est[{idx}]: {consistency_metrics_pearson}")

        consistency_metrics_spearman = eva.causal_structure_consistency(W_true, W_est != 0, method='spearman')
        print(f"Causal Structure Consistency (Spearman) for W_est[{idx}]: {consistency_metrics_spearman}")
    
    expectation_W = aux.frequency_summary(W_est_list)

    acc = eva.causal_structure_accuracy(B_true, expectation_W != 0)  # Calculate accuracy
    print(f"Accuracy for expectation_W with masked_expectation: {acc}")

    for con_l in [0.95,0.9,0.8,0.7]:
        expectation_W, confidence_intervals = aux.conformal_inference(W_est_list, confidence_level=con_l)
        non_zero_indices = np.where(expectation_W !=0)[0]
        print(f"Confidence intervals of edge weights with confidence_level={con_l}:")
        print(confidence_intervals[non_zero_indices])

        acc = eva.causal_structure_accuracy(B_true, expectation_W != 0)  # Calculate accuracy
        print(f"Accuracy for expectation_W using conformal_inference with confidence_level={con_l}: {acc}")
        
        p_values, causal_effect_intervals = aux.causal_effects_with_significance(loss_jacobian_list, expectation_W, confidence_level=con_l)
        print(f"Causal effect intervals associated with loss_jacobian_list and confidence_level={con_l}:")
        print(causal_effect_intervals[non_zero_indices])
        print(np.sum(p_values[non_zero_indices]<0.05)/len(non_zero_indices))

        p_values, causal_effect_intervals = aux.causal_effects_with_significance(score_jacobian_list, expectation_W, confidence_level=con_l)
        print(f"Causal effect intervals associated with score_jacobian_list and confidence_level={con_l}:")
        print(causal_effect_intervals[non_zero_indices])
        print(np.sum(p_values[non_zero_indices]<0.05)/len(non_zero_indices))
