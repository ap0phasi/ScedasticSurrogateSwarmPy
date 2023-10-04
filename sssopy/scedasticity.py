from surrogatemodel import *

from scipy.spatial.distance import cdist
import numpy as np

def backprop_error(center,surrogatesaves,centersaves,error):
    
    # Find which is the closest center to the center
    closest_center_index = np.argsort(cdist(center,centersaves))[0][0]
    # Get coefficients of first order surrogate model
    first_order_coefs = surrogatesaves[closest_center_index][0].model.coef_
    x, residuals, rank, s = np.linalg.lstsq(first_order_coefs, error, rcond=None)
    error_magitude = np.maximum(0.1,np.absolute(x))
    error_magitude = error_magitude / sum(error_magitude)
    return error_magitude

def error_by_point(modeled,actual):
    return (modeled-actual)**2