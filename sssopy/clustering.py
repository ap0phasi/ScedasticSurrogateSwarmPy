# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 19:58:47 2023

@author: johnm
"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def cluster_points(inputs, num_clusters, min_points_per_cluster):
    """
    Cluster the input points using KMeans algorithm.

    Args:
        inputs (numpy.ndarray): Input points for clustering.
        num_clusters (int): Number of desired clusters.
        min_points_per_cluster (int): Minimum points required per cluster.

    Returns:
        numpy.ndarray: Cluster membership vector.
        numpy.ndarray: Cluster centers.
    """
    # Create KMeans instance
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the model and predict cluster labels
    cluster_labels = kmeans.fit_predict(inputs)

    # Get valid cluster centers
    cluster_centers = kmeans.cluster_centers_
        
    invalidity_mask = np.ones(num_clusters)
    for cluster_id in np.arange(num_clusters):
        # Check if cluster is invalid
        if np.bincount(cluster_labels)[cluster_id] < min_points_per_cluster:
            # Find next nearest cluster
            distance_matrix = cdist(inputs[cluster_labels==cluster_id], cluster_centers)
            # Ignore clusters already labeled as invalid
            rank_matrix = np.argsort(distance_matrix * invalidity_mask, axis=1).argsort(axis=1)
            # Assign to points to next nearest cluster
            cluster_labels[cluster_labels==cluster_id]=np.where(rank_matrix==1)[1]
            # Label cluster as invalid
            invalidity_mask[cluster_id] = None
            
    # Redetermine cluster centers
    cluster_centers = np.array([inputs[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])
    
    # Finally drop the invalid clusters and renumber our cluster labels
    keep_index = np.where(~np.isnan(cluster_centers[:,0]))[0]
    cluster_centers = cluster_centers[keep_index,:]
    new_labels = cluster_labels
    for keep_i, keep_num in enumerate(keep_index):
        new_labels[np.where(cluster_labels==keep_num)]=keep_i
    return new_labels, cluster_centers

if __name__ == "__main__":
    n = 13
    samp = 100
    inputs = np.random.rand(samp, n)

    num_clusters = 10
    min_points_per_cluster = 10

    clustering_vector, cluster_centers = cluster_points(inputs, num_clusters, min_points_per_cluster)
    print("Clustering Vector:")
    print(clustering_vector)
    print("\nCluster Centers:")
    print(cluster_centers)

    import matplotlib.pyplot as plt

    # Generate data
    n = 2
    samp = 150
    inputs = np.random.rand(samp, n)

    # Cluster the points
    num_clusters = 5
    min_points_per_cluster = 30
    clustering_vector, cluster_centers = cluster_points(inputs, num_clusters, min_points_per_cluster)

    # Plot the points and cluster centers
    plt.scatter(inputs[:, 0], inputs[:, 1], c=clustering_vector, cmap='viridis', marker='o', label='Points')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=np.arange(cluster_centers.shape[0]), cmap='viridis', marker='x', label='Cluster Centers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clustered Points and Cluster Centers')
    plt.legend()
    plt.show()