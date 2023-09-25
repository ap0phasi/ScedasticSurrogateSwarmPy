from multiprocessing import Pool
import numpy as np

from sampling import calculate_sampling_range, latin_hypercube_within_range
from surrogateeval import eval_surrogate
from surrogatemodel import *

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
            "revert_best": False
        }
    
    def initialize_search(self):
        """Initialize Search Group

        Returns:
            dict: search state of optimizer
        """
        pos_x = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
        
        return {
            "pos_x": pos_x,
            "gen" : False
        }
    
    def step_search(self):
        pos_x = self.search_state["pos_x"]
        gen = self.search_state["gen"]
        
        # If gen state is true, generate a new search position
        if gen:
            newpos = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
            pos_x = np.vstack([pos_x, newpos])
        
        # Evaluate model function for each pos_x
        center_results = evaluate_model_fun(pos_x, self.modelfun)

        surrogatesaves = []
        for center in pos_x:
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
            sample_results = evaluate_model_fun(lhs_samples, self.modelfun)
            
            # Fit surrogates to sample results
            selector = SurrogateModelSelector(fit_threshold=0.0001)
            selector.fit_models(lhs_samples, sample_results, center)
            surrogatesaves.append(selector.models)
            
        guess_point = np.array([lhs_samples[0,:]]) + np.random.uniform(-0.01, 0.01, size=(1, self.param_len))
        
        print(selector.models[0].model.coef_.shape)
        interpolated_value = eval_surrogate(guess_point,surrogatesaves,pos_x)
        actual_value = evaluate_model_fun(guess_point, self.modelfun)
        
        plt.scatter(interpolated_value,actual_value)
        plt.show()
            
        self.search_state["pos_x"] = pos_x
        self.search_state["gen"] = False
        
#Example Usage
if __name__ == "__main__":
    def model_function(params):
        modval = params[0] * np.cos(xdat * params[1]) + params[1] * np.sin(xdat * params[0])
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
                               [-1,-1],
                               [1,1])
    print(searcher.search_state["pos_x"])
    searcher.step_search()
    searcher.step_search()
    print(searcher.search_state["pos_x"])
