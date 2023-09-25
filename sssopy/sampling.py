import numpy as np
from scipy.stats import qmc

def calculate_sampling_range(center, lowlim, highlim, magnitude):
    # Calculate the range as a fraction of the difference between highlim and lowlim
    range_fraction = magnitude * (highlim - lowlim)
    
    # Calculate the lower and upper limits for sampling
    l_bounds = np.maximum(center - range_fraction / 2, lowlim)
    u_bounds = np.minimum(center + range_fraction / 2, highlim)
    
    return l_bounds, u_bounds

def latin_hypercube_within_range(l_bounds, u_bounds, num_samples):
    # Generate Latin hypercube samples within the calculated range
    num_dimensions = len(l_bounds)

    # Use qmc package to perform latin hypercube sampling
    sampler = qmc.LatinHypercube(d=num_dimensions)
    sample = sampler.random(n=num_samples)
    # Scale latin hypercube samples within provided bounds
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
    
    return scaled_samples

# Example Usage
if __name__ == "__main__":
    # Example usage:
    center = np.array([0.5, 0.5])
    lowlim = np.array([0.0, 0.0])
    highlim = np.array([1.0, 1.0])
    magnitude = np.array([0.1, 0.2])
    num_samples = 10

    # Calculate the sampling range
    lower_limit, upper_limit = calculate_sampling_range(center, lowlim, highlim, magnitude)
    print(lower_limit)
    print(upper_limit)
    
    lhs_samples = latin_hypercube_within_range(lower_limit,upper_limit,30)
    print(lhs_samples)
