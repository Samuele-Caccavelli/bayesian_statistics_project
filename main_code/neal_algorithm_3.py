import numpy as np
import random
import copy
from tqdm import tqdm
import pickle

def cluster_probabilities(i, clusters, Y, integral_func_1, integral_func_2, alpha):
    """
    Computes the probabilities of observation i joining each clusters or creating a new one. The output is not sctictly probabilities but weights.
    
    Args:
        i: int, index of the observation to add to the clustering
        clusters: list of list, current partition
        Y: 2D array, of observations. Each observation is a vetor of size D. So Y is of shape (n_observation, D)
        integral_func_1: function to compute the first integral in (3.7). Takes as argument the current observations in a cluster and the new observation.
        integral_func_2: function to compute the second integral in (3.7). Takes as argument the only observation of the new cluster
        alpha: float, concentration parameter. alpha > 0
    Returns:
        probabilities array, probabilities[j] = probability of joining cluster j. probabilities[-1] = probability of creating a new cluster
    """
    n = len(Y)
    n_clusters = len(clusters)
    probabilities = np.zeros(n_clusters+1)

    # probabilities of joining existing cluster
    for c in range(n_clusters):
        probabilities[c] = integral_func_1(Y, clusters[c], i)
        probabilities[c] *= (len(clusters[c]) / (n - 1 + alpha))

    # probability of creating new cluster
    probabilities[-1] = integral_func_2(Y, i)
    probabilities[-1] *= alpha / (n - 1 + alpha)

    return probabilities


def algorithm_3(n_steps, Y, integral_func_1, integral_func_2, alpha=1):
    """
    Performs a markov chain using algorithm 3 from Neal (2000). 

    Args:
        n_steps: int, number of step of the markov chain to do (one step is defined as randomly moving each observation 1 time)
        Y: 2D array, of observations. Each observation is a vetor of size D. So Y is of shape (n_observation, D)
        integral_func_1: function to compute the first integral in (3.7). Takes as argument the current observations in a cluster and the new observation.
        integral_func_2: function to compute the second integral in (3.7). Takes as argument the only observation of the new cluster
        alpha: float, concentration parameter. alpha > 0
    """
    D = len(Y[0])
    n_obs = len(Y)

    clusters = [[i] for i in range(n_obs)]

    history = [copy.deepcopy(clusters)]

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
            weights = cluster_probabilities(i, clusters, Y, integral_func_1, integral_func_2, alpha)
            transitions = list(range(len(weights)))
            transition = random.choices(transitions, weights=weights)[0]

            # 4. Apply transition 
            if transition == len(clusters):  # add new cluster
                clusters.append([i])
            else:
                clusters[transition].append(i)

        # All elements have moved once -> one step of the Markov chain
        history.append(copy.deepcopy(clusters))

        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    return history

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