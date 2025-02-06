import numpy as np
import random
import copy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('../../main_code')))
from main_code.partitions_analysis import compute_entropy

def compute_mu_0(Y):
    return np.mean(Y, axis=0)   # default to mean of data, reasonable assumption

def compute_nu_0(D):
    return D    # default to D

def compute_inv_scale_mat_0(D):
    return np.eye(D)    # default to identity

def compute_lamb_0(Y):
    return 1    # default to 1

def compute_mahalanobis_penalty(X, cluster, new_obs, cov_matrix):
    """
    Computes the Mahalanobis distance penalty for a cluster.

    Args:
        X: 2D array of covariates.
        cluster: list of indices, representing the current cluster.
        new_obs: int, index of the new observation.
        cov_matrix: 2D array, covariance matrix of the cluster.

    Returns:
        penalty: float, Mahalanobis distance penalty for the cluster.
    """
    if len(cluster) == 0:
        return 0  # No penalty for empty clusters
    
    # Combine current cluster observations and the new observation
    cluster_data = np.array([X[idx] for idx in cluster] + [X[new_obs]])
    cluster_mean = np.mean(cluster_data, axis=0)
    
    # Compute Mahalanobis distance for the new observation
    penalty = mahalanobis(X[new_obs], cluster_mean, np.linalg.inv(cov_matrix))
    return penalty

def cluster_probabilities(i, clusters, Y, X, integral_func_1, integral_func_2, alpha,
                          mu_0, lamb_0, nu_0, inv_scale_mat_0, lambda_penalty):
    """
    Computes the probabilities of observation i joining each clusters or creating a new one. The output is not sctictly probabilities but weights.
    
    Args:
        i: int, index of the observation to add to the clustering
        clusters: list of list, current partition
        Y: 2D array, of observations. Each observation is a vetor of size D. So Y is of shape (n_observation, D)
        X: 2D array, covariate data for penalty computation.
        integral_func_1: function to compute the first integral in (3.7). Takes as argument the current observations in a cluster and the new observation.
        integral_func_2: function to compute the second integral in (3.7). Takes as argument the only observation of the new cluster
        alpha: float, concentration parameter. alpha > 0
        mu_0, lamb_0, nu_0, inv_scale_mat_0: prior parameters
        lambda_penalty: float, weight for Mahalanobis penalty.
    Returns:
        probabilities array, probabilities[j] = probability of joining cluster j. probabilities[-1] = probability of creating a new cluster
    """
    n = len(Y)
    n_clusters = len(clusters)
    probabilities = np.zeros(n_clusters+1)

    # Compute covariance matrix for Mahalanobis penalty on X
    cov_matrix_X = np.cov(X.T)

    # probabilities of joining existing cluster
    for c in range(n_clusters):
        probabilities[c] = integral_func_1(Y, clusters[c], i, mu_0, lamb_0, nu_0, inv_scale_mat_0)
        probabilities[c] *= (len(clusters[c]) / (n - 1 + alpha))

        # Apply Mahalanobis penalty based on X
        penalty = compute_mahalanobis_penalty(X, clusters[c], i, cov_matrix_X)
        probabilities[c] *= np.exp(-lambda_penalty * penalty)

    # probability of creating new cluster
    probabilities[-1] = integral_func_2(Y, i, mu_0, lamb_0, nu_0, inv_scale_mat_0)
    probabilities[-1] *= alpha / (n - 1 + alpha)

    return probabilities

def compute_summary_statistics(clustering, X):
    """
    Compute the mean of each cluster based on the provided clustering and covariates.
    
    Parameters:
        clustering (list of lists): A list where each sublist contains the indices of data points in a cluster.
        X (numpy.ndarray): A 2D array (n_samples, n_features) representing the covariates of the data.
        
    Returns:
        numpy.ndarray: A 2D array (n_clusters, n_features) containing the mean of each cluster.
    """
    cluster_means = []
    for cluster in clustering:
        if len(cluster) > 0:  # Avoid empty clusters
            cluster_points = X[cluster]  # Select points in the cluster
            cluster_mean = np.mean(cluster_points, axis=0)  # Compute the mean along features
            cluster_means.append(cluster_mean)
        else:
            cluster_means.append(np.zeros(X.shape[1]))  # Placeholder for empty clusters (all-zero vector)
    
    return np.array(cluster_means)

def cluster_summary_statistics(summary_statistics, n_clusters):
    """
    Perform clustering on the summary statistics of Layer 1 clusters.
    
    Parameters:
        summary_statistics (numpy.ndarray): A 2D array (n_clusters_layer1, n_features) 
                                             containing the mean of each cluster from Layer 1.
        n_clusters (int): The number of higher-level clusters to form.
        
    Returns:
        list of lists: A list where each sublist contains the indices of Layer 1 clusters
                       grouped into the same higher-level cluster.
    """
    # Perform clustering on the summary statistics
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(summary_statistics)
    
    # Group indices of Layer 1 clusters by their assigned higher-level cluster
    grouped_clusters = [[] for _ in range(n_clusters)]
    for cluster_idx, label in enumerate(labels):
        grouped_clusters[label].append(cluster_idx)
    
    return grouped_clusters

def algorithm_3(n_steps, Y, X, integral_func_1, integral_func_2, alpha=1, lambda_penalty=0.1,
                compute_mu_0=compute_mu_0, compute_lamb_0=compute_lamb_0, compute_nu_0=compute_nu_0, compute_inv_scale_mat_0=compute_inv_scale_mat_0,
                visualize_entropy=True, folder_path="../../results", file_name=None):
    """
    Performs a markov chain using algorithm 3 from Neal (2000). 

    Args:
        n_steps: int, number of step of the markov chain to do (one step is defined as randomly moving each observation 1 time)
        Y: 2D array, of observations. Each observation is a vector of size D. So Y is of shape (n_observation, D)
        integral_func_1: function to compute the first integral in (3.7). Takes as argument the current observations in a cluster and the new observation.
        integral_func_2: function to compute the second integral in (3.7). Takes as argument the only observation of the new cluster
        alpha: float, concentration parameter. alpha > 0
        compute_mu_0, compute_lamb_0, compute_nu_0, compute_inv_scale_mat_0: function to define prior assumptions
        visualize_entropy: boolean, to control if the entropy has to be shown at the end of the algorithm
        folder_path: str, folder path where results will be saved. Default works with our path layout
        file_name: str, name of the file (with extension) to save the results
    """
    D = len(Y[0])
    n_obs = len(Y)

    # Parameters fro the priors
    mu_0 = compute_mu_0(Y)
    lamb_0 = compute_lamb_0(Y)
    nu_0 = compute_nu_0(D)
    inv_scale_mat_0 = compute_inv_scale_mat_0(D)

    clusters = [[i] for i in range(n_obs)]

    history_l1 = [copy.deepcopy(clusters)]
    summary_statistics = compute_summary_statistics(history_l1[0], X)
    history_l2 = [cluster_summary_statistics(summary_statistics, 2)]

    # Compute entropy for the traceplot
    # entropies = [compute_entropy(clusters)]
    #! in the following way the entropy of the starting point (each point in their own cluster) is set to 0 
    #! so that we can more properly see the trace-plots without a huge value at the start
    entropies = [0] 

    # Initialize progress bar
    progress_bar = tqdm(total=n_steps, desc="Markov Chain Progress", unit="step")

    for step in range(n_steps):  # Markov chain
        for i in range(n_obs):  # 1 step of the Markov chain
            # 1. Find in which cluster the observation is
            c = 0
            for index in range(len(clusters)):  # one step 
                if i in clusters[index]:
                    c = index
                    break
            # 2. Remove observation i from clusters:
            if len(clusters[c]) == 1:  # case 1: i is the only element of the cluster -> remove cluster
                del clusters[c]
            else:  # case 2: cluster has more than 1 element -> remove i from the cluster
                clusters[c].remove(i)

            # 3. Compute probabilities of adding i to each cluster
            weights = cluster_probabilities(i, clusters, Y, X, integral_func_1, integral_func_2, alpha,
                                            mu_0, lamb_0, nu_0, inv_scale_mat_0, lambda_penalty)
            transitions = list(range(len(weights)))
            transition = random.choices(transitions, weights=weights)[0]

            # 4. Apply transition 
            if transition == len(clusters):  # add new cluster
                clusters.append([i])
            else:
                clusters[transition].append(i)

        # Apply second layer of clustering
        summary_statistics = compute_summary_statistics(clusters, X)

        # All elements have moved once -> one step of the Markov chain
        history_l1.append(copy.deepcopy(clusters))
        history_l2.append(cluster_summary_statistics(summary_statistics, 2))
        entropies.append(compute_entropy(clusters))

        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    if visualize_entropy:
            plt.plot(entropies)
            plt.title("Traceplot")
            plt.xlabel("Iteration")
            plt.ylabel("Entropy")

    # Save results to the specified file if both folder_path and file_name are provided
    if folder_path and file_name:
        full_path = os.path.join(folder_path, file_name)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
        with open(full_path, "wb") as f:
            pickle.dump({"history": history_l1, "entropies": entropies}, f)
        print(f"Results saved to {full_path}")

    return {"history_l1": history_l1, "history_l2": history_l2,"entropies": entropies, "save_path": full_path if folder_path and file_name else None}

def save_data(file_path, data, labels, MCMC_history, parameters):
    """
    Saves data, labels, output, and parameters to a file.

    Args:
        file_path (str): Path to the file where data will be stored.
        data (numpy.ndarray): Array of shape (n_samples, n_dimensions).
        labels (numpy.ndarray): Array of shape (n_samples).
        output (numpy.ndarray): Matrix of size (n_samples, n_samples).
        parameters (dict): Dictionary of parameters.
    """
    with open(file_path, 'wb') as file:
        pickle.dump({'data': data, 'labels': labels, 'MCMC_history': MCMC_history, 'parameters': parameters}, file)

def load_data(file_path):
    """
    Loads data, labels, output, and parameters from a file.

    Args:
        file_path (str): Path to the file to load data from.

    Returns:
        tuple: A tuple containing data, labels, output, and parameters.
    """
    with open(file_path, 'rb') as file:
        stored_data = pickle.load(file)
    return stored_data['data'], stored_data['labels'], stored_data['MCMC_history'], stored_data['parameters']