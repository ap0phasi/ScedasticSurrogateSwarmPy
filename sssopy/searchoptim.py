import numpy as np

from sampling import calculate_sampling_range, latin_hypercube_within_range
from surrogateeval import eval_surrogate
from surrogatemodel import *
from optplotter import plot_optimizer_results
from problem import SSSoProblem

from scipy.optimize import differential_evolution

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

class SurrogateSearch:
    """
    A class to perform surrogate searching
    
    Attributes:
        search_state: The current state of the class's search
    """
    def __init__(self,optproblem,desired_vals,param_len,lowlim,highlim,config=None):
        self.optproblem = optproblem
        self.desired_vals = desired_vals
        self.param_len = param_len
        self.lowlim = lowlim
        self.highlim = highlim
        self.config = config if config else self.default_config()
        self.search_state = self.initialize_search()
    
    def default_config(self):
        return {
            "search_mag": 0.2,
            "search_samples": self.param_len * 2,
            "revert_best": False,
            "error_measure": "rmse",
            "opt_mag": 2,
            "fit_threshold": 0.0
        }
    
    def initialize_search(self):
        """Initialize Search Group

        Returns:
            dict: search state of optimizer
        """
        pos_x = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
        
        return {
            "pos_x": pos_x,
            "gen" : False,
            "surrogatesaves": [],
            "centersaves": np.empty((0,self.param_len), int)
        }
    
    def step_search(self):
        pos_x = self.search_state["pos_x"]
        gen = self.search_state["gen"]
        surrogatesaves = self.search_state["surrogatesaves"]
        centersaves = self.search_state["centersaves"]
        
        # If gen state is true, generate a new search position
        if gen:
            newpos = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
            pos_x = np.vstack([pos_x, newpos])
        
        # Evaluate model function for each pos_x
        center_results = evaluate_model_fun(pos_x, self.optproblem)
        
        surrogate_recommendations = np.empty((0,self.param_len), int)
        
        # For each center and center result, search and optimize
        
        ### Are we okay with it always going in the same order?
        for center, center_result in zip(pos_x, center_results):
            # For each center, determine sampling range
            lower_limit, upper_limit = calculate_sampling_range(center, 
                                                                np.array(self.lowlim), 
                                                                np.array(self.highlim), 
                                                                self.config["search_mag"])
            # Perform Latin Hypercube Sampling within determined range
            lhs_samples = latin_hypercube_within_range(lower_limit,
                                                       upper_limit,
                                                       self.config["search_samples"])
            
            # Evaluate model function for samples
            lhs_results = evaluate_model_fun(lhs_samples, self.optproblem)
            
            # Append center result and centers to data for surrogate fitting
            function_inputs = np.vstack([lhs_samples,center])
            function_outputs = np.vstack([lhs_results,center_result])
            # Fit surrogates to sample results
            selector = SurrogateModelSelector(fit_threshold=self.config["fit_threshold"])
            selector.fit_models(function_inputs, function_outputs, center)
            
            # Centersaves must be the same length as surrogate saves
            centersaves = np.vstack([centersaves,center])
            surrogatesaves.append(selector.models)
            
            # Perform Optimization on Surrogate Model
            lower_opt, upper_opt = calculate_sampling_range(center, 
                                                                np.array(self.lowlim), 
                                                                np.array(self.highlim), 
                                                                self.config["opt_mag"])
            opt_bounds = list(zip(lower_opt, upper_opt))
            subopt_result = differential_evolution(surrogate_optimization_function,
                                               opt_bounds, 
                                               args = (self.desired_vals,
                                                        surrogatesaves,
                                                        centersaves,
                                                        self.config["error_measure"],
                                                        self.optproblem))
            surrogate_recommendations = np.vstack([surrogate_recommendations,subopt_result.x]) 
            
        self.search_state["pos_x"] = surrogate_recommendations
        self.search_state["gen"] = True
        self.search_state["surrogatesaves"] = surrogatesaves
        self.search_state["centersaves"] = centersaves
        
#Example Usage
if __name__ == "__main__":
    def model_function(params,args):
        input_vals, check = args
        modval = params[0] * np.cos(input_vals * params[1]) + params[1] * np.sin(input_vals * params[0])
        #modval = params[0] + input_vals * params[1]
        return modval
    
    def ineq_constraints(params):
        ineq1 = (params[0]-params[1])
        ineq2 = (params[0]-params[1])
        return [ineq1]
    
    def eq_constraints(params):
        eq1 = (params[0]+params[1])-0.72
        return [0]
        
    check = 1
    xdat = np.arange(1, 100.5, 0.5)
    params = np.array([0.32, 0.4])
    opt_problem = SSSoProblem(model_function = model_function,
                              ineq_constraints = ineq_constraints,
                              eq_constraints = eq_constraints,
                              args = (xdat,check)
                              )
    ydat = opt_problem.eval_function(params)
    
    searcher = SurrogateSearch(opt_problem,
                               ydat,
                               2,
                               [0,0],
                               [1,1])
    
    for itt in range(10):
        searcher.step_search()
        plot_optimizer_results(searcher.search_state["pos_x"],
                               xdat,
                               ydat,
                               searcher.search_state["surrogatesaves"],
                               searcher.search_state["centersaves"],
                               optproblem = opt_problem)
        print(searcher.search_state["pos_x"])
        
    print(searcher.search_state["surrogatesaves"])
