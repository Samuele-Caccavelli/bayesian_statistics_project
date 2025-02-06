import numpy as np

def generate_gaussian_mixture(means, covariances, n_samples):
    """
    Generate synthetic data from a mixture of Gaussians.

    Parameters:
        means (list of array-like): List of mean vectors for each Gaussian.
        covariances (list of array-like): List of covariance matrices for each Gaussian.
        n_samples (list of int): List of the number of samples for each Gaussian.

    Returns:
        data (numpy.ndarray): Array of generated data points.
        labels (list of lists): Cluster belonging information.
    """
    data = []
    labels = []

    start_idx = 0
    for i, (mean, cov, n) in enumerate(zip(means, covariances, n_samples)):
        points = np.random.multivariate_normal(mean, cov, n)
        data.append(points)
        cluster_indices = list(range(start_idx, start_idx + n))
        labels.append(cluster_indices)
        start_idx += n

    data = np.vstack(data)
    return data, labels


