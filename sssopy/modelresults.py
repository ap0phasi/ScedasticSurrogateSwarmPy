from .surrogateeval import eval_surrogate

import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize

def evaluate_model_fun(pos_x,optproblem):
    """Evaluate user provided model function for sample set

    Args:
        pos_x (numpy array): sample set inputs
        optproblem (class): user defined SSSo optimization problem

    Returns:
        numpy array: model function results
    """
    results = np.array([optproblem.eval_function(row) for row in pos_x])
    return results
    
def surrogate_optimization_function(guess_point,*args):
    
    ### We don't need to develop surrogates for constraints, it is a waste of time. 
    desired_values, surrogatesaves, centersaves, error_measure, optproblem = args
    surrogate_out = eval_surrogate(np.array([guess_point]),surrogatesaves,centersaves)
    if error_measure == 'rmse':
        error_fun = np.sqrt(np.mean((surrogate_out - desired_values)**2))
        
    # Evaluate errors due to inequality constraints
    error_ineq = np.sqrt(np.mean(np.maximum(np.array(optproblem.eval_ineq(guess_point)),0)**2))
    
    # Evaluate errors due to equality constraints
    error_eq = np.sqrt(np.mean(np.array(optproblem.eval_eq(guess_point))**2))
        
    return(error_fun+error_ineq+error_eq)



def subopt(subopt_algo,optproblem,opt_bounds,error_measure,desired_vals,surrogatesaves,centersaves,center):
    if subopt_algo == "differential_evolution":
        subopt_result = differential_evolution(surrogate_optimization_function,
                                        opt_bounds, 
                                        popsize = 100,
                                        maxiter = 300,
                                        tol = 1e-9,
                                        args = (desired_vals,
                                                    surrogatesaves,
                                                    centersaves,
                                                    error_measure,
                                                    optproblem)
        )
        return subopt_result.x
    else:
        subopt_result = minimize(surrogate_optimization_function, 
                                x0 = center, 
                                #x0 = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))[0],
                                method=subopt_algo, 
                                bounds=opt_bounds, 
                                options = {"maxiter":1000},
                                args = (desired_vals,
                                                surrogatesaves,
                                                centersaves,
                                                error_measure,
                                                optproblem)
                                )
        return subopt_result.x
    