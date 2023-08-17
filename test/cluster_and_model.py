# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:26:27 2023

@author: johnm
"""

from clustering_tests import cluster_points
from surrogate_modelling_center_tests import *

from scipy.spatial.distance import cdist

def eval_surrogate(guess_point,surrogatesaves,centersaves):
    """
    Cluster the input points using KMeans algorithm.

    Args:
        guess_point (numpy.ndarray): Point at which to perform interpolation. 
        surrogatesaves (list): list of surrogates
        centersaves (list): list of centers

    Returns:
        numpy.ndarray: interpolated value
    """
    # Find which is the closest center to the guess point
    closest_center_index = np.argsort(cdist(guess_point,centersaves))[0][0]
    interpolated_value = surrogatesaves[closest_center_index][-1].predict(guess_point, centersaves[closest_center_index])
    return interpolated_value


def test_surrogate_on_function(fun,dim):
    n = dim  # Number of dimensions
    samp = 100  # Number of samples
    inputs = np.random.rand(samp, n)
    
    outputs = np.apply_along_axis(fun, axis=1, arr=inputs).reshape(-1, 1)
    
    # Determine clustering of inputs
    clustering_vector, cluster_centers = cluster_points(inputs, num_clusters = 3, min_points_per_cluster = 10)
    
    surrogatesaves=[]
    
    # Perform surrogate modelling per cluster
    for cluster_index in range(len(cluster_centers)):
        selector = SurrogateModelSelector(fit_threshold=0.0001)
        selector.fit_models(inputs[clustering_vector==cluster_index], outputs[clustering_vector==cluster_index], cluster_centers[cluster_index])
        surrogatesaves.append(selector.models)
    # Evaluate interpolation
    guess_point = np.random.rand(1, n)
    #guess_point = [inputs[1,:]]
    
    interpolated_value = eval_surrogate(guess_point,surrogatesaves,cluster_centers)
    actual_value = np.apply_along_axis(fun, axis=1, arr=guess_point).reshape(-1, 1)
    print(f"-----Evaluate Function: {fun.__name__}-------")
    print("Interpolated Value:")
    print(interpolated_value[0][0])
    print("\nActual Value:")
    print(actual_value[0][0])
    
# Calculate outputs using the Hartmann 6-dimensional function
def hartmann6d_function(x):
    # Hartmann 6-dimensional function coefficients
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])

    P = 10**-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])
    exp_term = np.exp(-np.sum(A * (x - P)**2, axis=1))
    return -np.sum(alpha * exp_term)

def quadratic_function(x):
    return x[0]**2 + x[1]**2 + x[0] * x[1]

test_surrogate_on_function(hartmann6d_function, 6)
test_surrogate_on_function(quadratic_function, 2)
