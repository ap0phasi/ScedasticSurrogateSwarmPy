import numpy as np

from .sampling import calculate_sampling_range, latin_hypercube_within_range
from .surrogatemodel import *
from .optplotter import plot_optimizer_results
from .problem import SSSoProblem
from .scedasticity import *
from .modelresults import *

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
            "search_samples": min([ 100, self.param_len * 2 ]),
            "revert_best": False,
            "error_measure": "rmse",
            "opt_mag": 2,
            "fit_threshold": 0.0, 
            "use_backprop": False,
            "subopt_algo": "differential_evolution",
            "always_gen" : False,
            "vel_w" : 0.1,
            "surro_w" : 0.9,
            "sced_w" : 1,
            "sced_mag" : 0.1,
            "supplement_historical": False
        }
    
    def initialize_search(self):
        """Initialize Search Group

        Returns:
            dict: search state of optimizer
        """
        pos_x = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
        vel_mag = np.abs(np.array(self.lowlim)-np.array(self.highlim))/1000
        vel = np.random.uniform(-vel_mag, vel_mag, size=(1, self.param_len))
        lowest_error = 1e9
        
        return {
            "pos_x": pos_x,
            "vel" : vel,
            "gen" : False,
            "surrogatesaves": [],
            "centersaves": np.empty((0,self.param_len), int),
            "lowest_error": lowest_error,
            "all_pos" : np.empty((0,self.param_len), int),
            "all_results" : []
        }
    
    def step_search(self):
        pos_x = self.search_state["pos_x"]
        vel = self.search_state["vel"]
        gen = self.search_state["gen"]
        surrogatesaves = self.search_state["surrogatesaves"]
        centersaves = self.search_state["centersaves"]
        all_pos = self.search_state["all_pos"]
        all_results = self.search_state["all_results"]
        
        # Update our positions with velocity, "bouncing" off of the bounds
        condition1 = (pos_x + vel) < self.lowlim
        condition2 = (pos_x + vel) > self.highlim

        vel[condition1] = -vel[condition1] / 100
        vel[condition2] = -vel[condition2] / 100
        
        # Update position
        pos_x = pos_x + vel
        
        # If gen state is true, generate a new search position
        if gen:
            newpos = np.random.uniform(self.lowlim, self.highlim, size=(1, self.param_len))
            pos_x = np.vstack([pos_x, newpos])
            vel_mag = np.abs(np.array(self.lowlim)-np.array(self.highlim))/1000
            newvel = np.random.uniform(-vel_mag, vel_mag, size=(1, self.param_len))
            vel = np.vstack([vel,newvel])

        # Evaluate model function for each pos_x
        center_results = evaluate_model_fun(pos_x, self.optproblem)
        
        #Initialize Empty Surrogate Recommendations
        surrogate_recommendations = np.empty((0,self.param_len), int)
        
        # Save center results for swarm reference
        all_pos = np.vstack([all_pos,pos_x])
        if len(all_results)>0:
            all_results = np.vstack([all_results, center_results])
        else:
            all_results = center_results
        
        # For each center and center result, search and optimize
        current_min_error = []
        ### Are we okay with it always going in the same order?
        for center, center_result in zip(pos_x, center_results):
            # For each center, determine sampling range
            model_error = error_by_point(center_result,self.desired_vals)
            current_min_error.append(min(model_error)) 
            
            if (len(surrogatesaves)>0) & self.config["use_backprop"]:
                search_mag = np.maximum(0.2,(backprop_error(np.array([center]),surrogatesaves,centersaves,model_error) * \
                    self.config["search_mag"] * self.param_len))
            else:
                search_mag = self.config["search_mag"]
            
            lower_limit, upper_limit = calculate_sampling_range(center, 
                                                                np.array(self.lowlim), 
                                                                np.array(self.highlim), 
                                                                search_mag)
            
            
            # Perform Latin Hypercube Sampling within determined range
            lhs_samples = latin_hypercube_within_range(lower_limit,
                                                       upper_limit,
                                                       self.config["search_samples"])
            
            # Evaluate model function for samples
            lhs_results = evaluate_model_fun(lhs_samples, self.optproblem)
            
            # Optionally, supplement with historical values that are within the LHS range
            if self.config["supplement_historical"]:
                # Create boolean masks
                mask_lower = np.all(all_pos >= lower_limit, axis=1)
                mask_upper = np.all(all_pos <= upper_limit, axis=1)

                # Combine masks
                final_mask = mask_lower & mask_upper
                
                # Filter the arrays
                supplemental_pos = all_pos[final_mask]
                supplemental_res = all_results[final_mask]
                
                # Append center result, supplementals, and centers to data for surrogate fitting
                function_inputs = np.unique(np.vstack([lhs_samples, supplemental_pos, center]),axis = 0)
                function_outputs = np.unique(np.vstack([lhs_results, supplemental_res, center_result]),axis = 0)
            else:
                # Append center result and centers to data for surrogate fitting
                function_inputs = np.vstack([lhs_samples, center])
                function_outputs = np.vstack([lhs_results, center_result])
                
            # Save lhs results for swarm reference
            all_pos = np.vstack([all_pos, lhs_samples])
            all_results = np.vstack([all_results, lhs_results])
            
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
            
            # Perform suboptimization with selected optimizer
            subopt_result = subopt(self.config["subopt_algo"],
                                   self.optproblem,
                                   opt_bounds,
                                   self.config["error_measure"],
                                   self.desired_vals,
                                   surrogatesaves,
                                   centersaves,
                                   center)
           
            surrogate_recommendations = np.vstack([surrogate_recommendations,subopt_result])
            
            # subopt_result = differential_evolution(surrogate_optimization_function,
            #                                    opt_bounds, 
            #                                    seed = 10,
            #                                    args = (self.desired_vals,
            #                                             surrogatesaves,
            #                                             centersaves,
            #                                             self.config["error_measure"],
            #                                             self.optproblem))
            # surrogate_recommendations = np.vstack([surrogate_recommendations,subopt_result.x])
        
        pos_mean = np.mean(center_results,axis = 0)
        pos_std = np.std(center_results,axis = 0)
        pos_het = heteroscedastic_loss(self.desired_vals,pos_mean,pos_std)
        
        # Backpropagate the heteroscedastic loss
        # Conditions for setting use_sced
        sced_conditions = [
            pos_x.shape[0] > 1,
            sum(np.maximum(0,pos_het))>0
        ]

        # Check if any condition is True
        use_sced = all(sced_conditions)
        if use_sced:
            het_samples = np.empty((0,self.param_len), int)
            for x in pos_x:
                het_backprop = backprop_error(center = np.array([x]),
                                                surrogatesaves=surrogatesaves,
                                                centersaves = centersaves,
                                                error = np.maximum(0,pos_het))
                het_range_lower, het_range_upper = calculate_sampling_range(x,
                                                            np.array(self.lowlim),
                                                            np.array(self.highlim),
                                                            magnitude = het_backprop * self.config["sced_mag"])
                het_sample = latin_hypercube_within_range(het_range_lower,het_range_upper,1)
                het_samples = np.vstack([het_samples,het_sample])

        # Determine if new centers should be generated
        if (self.search_state["lowest_error"] < min(current_min_error))|self.config["always_gen"]:
            self.search_state["gen"] = True
            self.search_state["lowest_error"] = min(current_min_error)
            print("Generating new center")
        else:
            self.search_state["gen"] = False
            self.search_state["lowest_error"] = min(current_min_error)
        
        vel = vel * self.config["vel_w"] + \
            self.config["surro_w"] * (surrogate_recommendations - pos_x)
            
        if use_sced:
            vel = vel + self.config["sced_w"] * (het_samples - pos_x) 
        
        self.search_state["pos_x"] = pos_x
        self.search_state["pos_results"] = center_results
        self.search_state["vel"] = vel
        self.search_state["surrogatesaves"] = surrogatesaves
        self.search_state["centersaves"] = centersaves
        self.search_state["surrogate_recommendations"] = surrogate_recommendations
        self.search_state["all_pos"] = all_pos
        self.search_state["all_results"] = all_results
        
        
#Example Usage
if __name__ == "__main__":
    
    objective_function_calls = 0
    
    def model_function(params,args):
        input_vals, check = args
        modval = params[0] * np.cos(input_vals * params[1]) + params[1] * np.sin(input_vals * params[0])
        # modval = params[0] + input_vals**2 * params[1]**2
        global objective_function_calls
        objective_function_calls += 1
        return modval
    
    def ineq_constraints(params):
        # ineq1 = (params[0]-params[1])
        # ineq2 = (params[0]-params[1])
        return [0]
    
    def eq_constraints(params):
        # eq1 = (params[0]+params[1])-0.72
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
    print(f"Objective Function Calls: {objective_function_calls}")
    
    searcher = SurrogateSearch(opt_problem,
                               ydat,
                               2,
                               [0,0],
                               [1,1])
    
    for itt in range(15):
        searcher.step_search()
        plot_optimizer_results(searcher.search_state["pos_x"],
                               xdat,
                               ydat,
                               searcher.search_state["surrogatesaves"],
                               searcher.search_state["centersaves"],
                               optproblem = opt_problem)
        #print(searcher.search_state["pos_x"])
        print(searcher.search_state["surrogate_recommendations"])
        print(f"Objective Function Calls: {objective_function_calls}")

