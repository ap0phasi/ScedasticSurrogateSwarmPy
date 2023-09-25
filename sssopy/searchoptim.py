import numpy as np

from sampling import calculate_sampling_range, latin_hypercube_within_range
from surrogateeval import eval_surrogate
from surrogatemodel import *

from scipy.optimize import differential_evolution

def evaluate_model_fun(pos_x,model_function):
    """Evaluate user provided model function for sample set

    Args:
        pos_x (numpy array): sample set inputs
        model_function (fun): user defined model function

    Returns:
        numpy array: model function results
    """
    results = np.array([model_function(row) for row in pos_x])
    return results
    
def surrogate_optimization_function(guess_point,*args):
    desired_values, surrogatesaves, pos_x, error_measure = args
    surrogate_out = eval_surrogate(np.array([guess_point]),surrogatesaves,pos_x)
    
    if error_measure == 'rmse':
        error = np.sqrt(np.mean((surrogate_out - desired_values)**2))
        
    return(error)

class SurrogateSearch:
    """
    A class to perform surrogate searching
    
    Attributes:
        search_state: The current state of the class's search
    """
    def __init__(self,modelfun,desired_vals,param_len,lowlim,highlim,config=None):
        self.modelfun = modelfun
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
            "opt_mag": 0.6
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
        center_results = evaluate_model_fun(pos_x, self.modelfun)
        
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
            lhs_results = evaluate_model_fun(lhs_samples, self.modelfun)
            
            # Append center result and centers to data for surrogate fitting
            function_inputs = np.vstack([lhs_samples,center])
            function_outputs = np.vstack([lhs_results,center_result])
            # Fit surrogates to sample results
            selector = SurrogateModelSelector(fit_threshold=0.0001)
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
                                                        self.config["error_measure"]))
            surrogate_recommendations = np.vstack([surrogate_recommendations,subopt_result.x]) 
            
        self.search_state["pos_x"] = surrogate_recommendations
        self.search_state["gen"] = True
        self.search_state["surrogatesaves"] = surrogatesaves
        self.search_state["centersaves"] = centersaves
        
#Example Usage
if __name__ == "__main__":
    def model_function(params):
        modval = params[0] * np.cos(xdat * params[1]) + params[1] * np.sin(xdat * params[0])
        # modval = params[0] + xdat * params[1]**2
        return modval
    
    xdat = np.arange(1, 100.5, 0.5)
    params = np.array([0.32, 0.4])
    ydat = model_function(params)
    
    import matplotlib.pyplot as plt
    # plt.scatter(xdat,ydat)
    # plt.show()
    
    searcher = SurrogateSearch(model_function,
                               ydat,
                               2,
                               [0,0],
                               [1,1])
    
    print(searcher.search_state["pos_x"])
    for itt in range(100):
        searcher.step_search()
        print(searcher.search_state["pos_x"])
