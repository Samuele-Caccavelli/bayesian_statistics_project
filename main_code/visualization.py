import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(data, optimal_clustering, title, xlabel="X1", ylabel="X2"):
    """
    Plots clusters based on the given clustering information.

    Parameters:
        data (np.ndarray): The dataset, where rows are observations and columns are features.
        optimal_clustering (list of lists): The clustering information as a list of lists, 
                                            where each sublist contains indices of points in a cluster.
        title (str): The title of the graph.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(optimal_clustering)))
    plt.figure(figsize=(8, 6))

    for cluster_id, observations in enumerate(optimal_clustering):
        cluster_data = data[observations]
        plt.scatter(cluster_data[:, 0], 
                    cluster_data[:, 1], 
                    label=f"Cluster {cluster_id}", 
                    color=colors[cluster_id],
                    s=100,
                    alpha=0.8,
                    edgecolor="k")

    # Add labels, legend, and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_second_layer_clusters(data, first_layer_clustering, second_layer_clustering, title, xlabel="X1", ylabel="X2"):
    """
    Plots all the data points corresponding to each second-layer cluster, using different colors for each cluster.
    
    Parameters:
        data (np.ndarray): The dataset, where rows are observations and columns are features.
        first_layer_clustering (list of lists): The first-layer clustering, 
                                                where each sublist contains indices of points in a cluster.
        second_layer_clustering (list of lists): The second-layer clustering, 
                                                 where each sublist contains indices of first-layer clusters.
        title (str): The title of the graph.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    # Helper function to flatten nested lists
    def flatten(nested_list):
        """Recursively flattens a nested list."""
        for item in nested_list:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    plt.figure(figsize=(8, 6))
    
    # Generate a unique color for each second-layer cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, len(second_layer_clustering)))

    # Iterate over each second-layer cluster
    for cluster_id, second_layer_cluster in enumerate(second_layer_clustering):
        # Flatten the indices of first-layer clusters
        first_layer_cluster_indices = list(flatten(second_layer_cluster))

        # Collect all data points corresponding to this second-layer cluster
        all_points = []
        for first_layer_cluster_idx in first_layer_cluster_indices:
            all_points.extend(first_layer_clustering[first_layer_cluster_idx])

        # Extract the data points corresponding to the second-layer cluster
        cluster_data = data[all_points]

        # Plot the points for this second-layer cluster
        plt.scatter(cluster_data[:, 0],
                    cluster_data[:, 1],
                    color=colors[cluster_id],
                    label=f"Second-Layer Cluster {cluster_id}",
                    s=100,
                    alpha=0.8,
                    edgecolor="k")
    
    # Add labels, legend, and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()