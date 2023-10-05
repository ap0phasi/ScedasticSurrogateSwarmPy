from surrogatemodel import *

from scipy.spatial.distance import cdist
import numpy as np

def backprop_error(center,surrogatesaves,centersaves,error):
    
    # Find which is the closest center to the center
    closest_center_index = np.argsort(cdist(center,centersaves))[0][0]
    # Get coefficients of first order surrogate model
    first_order_coefs = surrogatesaves[closest_center_index][0].model.coef_
    x, residuals, rank, s = np.linalg.lstsq(first_order_coefs, error, rcond=None)
    error_magnitude = np.absolute(x)
    error_magnitude = error_magnitude / sum(error_magnitude)
    return error_magnitude

def error_by_point(modeled,actual):
    return (modeled-actual)**2

def heteroscedastic_loss(desired_values,mean,std):
    return 1/(2*std**2)*np.abs(desired_values-mean)**2+1/2*np.log(std**2)