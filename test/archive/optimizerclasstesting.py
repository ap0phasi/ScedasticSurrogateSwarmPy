
import numpy as np

class SurrogateSearch:
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
            "deg": 2,
            "search_mag": 0.2,
            "search_samples": self.param_len * 2,
            "gen": False,
            "revert_best": False
        }
    
    def initialize_search(self):
        pos_x = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
        
        return {
            "pos_x": pos_x
        }
    
    def step_search(self):
        pos_x = self.search_state["pos_x"]
        pos_x = pos_x + np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))/100
        self.search_state["pos_x"]=pos_x
        
#Example Usage
if __name__ == "__main__":
    def model_function(params, xdat):
        modval = params[0] * np.cos(xdat * params[1]) + params[1] * np.sin(xdat * params[0])
        return modval
    
    xdat = np.arange(1, 100.5, 0.5)
    params = np.array([0.32, 0.4])
    ydat = model_function(params, xdat)
    
    import matplotlib.pyplot as plt
    plt.scatter(xdat,ydat)
    plt.show()
    
    searcher = SurrogateSearch(model_function,
                               ydat,
                               2,
                               [-1,-1],
                               [1,1])
    print(searcher.search_state["pos_x"])
    searcher.step_search()
    print(searcher.search_state["pos_x"])
