import numpy as np

from sampling import calculate_sampling_range, latin_hypercube_within_range
from surrogatemodel import *
from optplotter import plot_optimizer_results_with_rec
from problem import SSSoProblem
from scedasticity import *
from modelresults import *
from clustering import cluster_points

from scipy.optimize import differential_evolution
from scipy.optimize import minimize

def fitness_eval(modeled_values,desired_values,error_measure):
    if error_measure == 'rmse':
        error_fun = np.array([np.sqrt(np.mean((row - desired_values)**2)) for row in modeled_values])
    return error_fun

class SurrogateSwarm:
    
    def __init__(self,optproblem,desired_vals,param_len,lowlim,highlim,config=None,surrogatesaves=None,centersaves=None):
        self.optproblem = optproblem
        self.desired_vals = desired_vals
        self.param_len = param_len
        self.lowlim = lowlim
        self.highlim = highlim
        self.config = config if config else self.default_config()
        self.surrogatesaves = surrogatesaves if surrogatesaves else []
        self.centersaves = centersaves if centersaves else np.empty((0,self.param_len), int)
        self.swarm_state = self.initialize_swarm()

    
    def default_config(self):
        return {
            "swarm_size" : 100,
            "error_measure": "rmse",
            "opt_mag": 2,
            "fit_threshold": 0.0, 
            "subopt_algo": "differential_evolution",
            "cluster_num" : 3,
            "cluster_min" : 5,
            "locality" : 0.3,
            "sced_mag" : 0.1,
            "p_w" : 0.3,
            "l_w" : 0.6,
            "lo_w" : 0.6,
            "vel_w" : 0.5,
            "surro_w" : 0.6,
            "sced_w" : 1 
        }
    
    def initialize_swarm(self):
        """Initialize Swarm

        Returns:
            dict: search state of optimizer
        """
        pos_x = np.random.uniform(self.lowlim, self.highlim, size=(self.config["swarm_size"], self.param_len))
        pos_results = evaluate_model_fun(pos_x, self.optproblem)
        pos_mean = np.mean(pos_results,axis = 0)
        pos_std = np.std(pos_results,axis = 0)
        pos_het = heteroscedastic_loss(self.desired_vals,pos_mean,pos_std)
        
        best_individual_position = pos_x
        best_individual_score = fitness_eval(pos_results,self.desired_vals,"rmse")
        
        vel_mag = np.abs(np.array(self.lowlim)-np.array(self.highlim))/1000
        vel = np.random.uniform(-vel_mag, vel_mag, size=(self.config["swarm_size"], self.param_len))
        return {
            "pos_x": pos_x,
            "pos_results" : pos_results,
            "surrogatesaves" : self.surrogatesaves,
            "centersaves" : self.centersaves,
            "pos_het" : pos_het,
            "best_individual_position" : best_individual_position,
            "best_individual_score" : best_individual_score,
            "all_pos" : pos_x,
            "all_results" : pos_results,
            "vel" : vel
        }
        
    def step_swarm(self):
        pos_x = self.swarm_state["pos_x"]
        pos_results = self.swarm_state["pos_results"]
        surrogatesaves = self.swarm_state["surrogatesaves"]
        centersaves = self.swarm_state["centersaves"]
        pos_het = self.swarm_state["pos_het"]
        best_individual_position = self.swarm_state["best_individual_position"]
        best_individual_score = self.swarm_state["best_individual_score"]
        all_pos = self.swarm_state["all_pos"]
        all_results = self.swarm_state["all_results"]
        vel = self.swarm_state["vel"]
        
        locality = round(self.config["locality"]*self.config["swarm_size"])
        
        # Update our positions with velocity, "bouncing" off of the bounds
        condition1 = (pos_x + vel) < self.lowlim
        condition2 = (pos_x + vel) > self.highlim

        vel[condition1] = -vel[condition1] / 100
        vel[condition2] = -vel[condition2] / 100
        pos_x = pos_x + vel
        
        pos_results = evaluate_model_fun(pos_x, self.optproblem)
        pos_mean = np.mean(pos_results,axis = 0)
        pos_std = np.std(pos_results,axis = 0)
        pos_het = heteroscedastic_loss(self.desired_vals,pos_mean,pos_std)
        all_pos = np.vstack([all_pos,pos_x])
        all_results = np.vstack([all_results, pos_results])
        
        # Backpropagate the heteroscedastic loss
        # Conditions for setting use_sced
        sced_conditions = [
            centersaves.shape[0] > 1,
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
            
        pos_scores = fitness_eval(pos_results,self.desired_vals,"rmse")
        improved_locations = pos_scores < best_individual_score
        best_individual_position[improved_locations] = pos_x[improved_locations]
        best_individual_score[improved_locations] = pos_scores[improved_locations]
        
        # For each point, evaluate the <locality> closest points and determine which has the best score currently
        #   and the best score overall
        best_local_position_current = np.empty((0,self.param_len), int)
        best_local_position_overall = np.empty((0,self.param_len), int)
        for row in pos_x:
            closest_points = np.argsort(cdist(np.array([row]),pos_x))[0][1:locality]
            best_local_position_current = np.vstack([
                best_local_position_current,
                pos_x[closest_points[pos_scores[closest_points].argmin(0)],:] 
            ])
            best_local_position_overall = np.vstack([
                best_local_position_overall,
                pos_x[closest_points[best_individual_score[closest_points].argmin(0)],:]
            ])
            
        # Random Generation for Velocity Updates
        r_p = np.random.uniform(0, 1, size=(self.config["swarm_size"], self.param_len)) # position
        r_l = np.random.uniform(0, 1, size=(self.config["swarm_size"], self.param_len)) # local current
        r_lo = np.random.uniform(0, 1, size=(self.config["swarm_size"], self.param_len)) # local overall
        
        # Cluster locusts
        clustering_vector, cluster_centers = cluster_points(pos_x, self.config["cluster_num"], self.config["cluster_min"])
        
        # Determine which of all historical points are closest to the cluster centers
        distances = cdist(cluster_centers, all_pos)
        historical_index = np.argsort(distances, axis=1)[:, 0]
        cluster_centers_hist = all_pos[historical_index,:]
        cluster_centers_hist_result = all_results[historical_index,:]
        
        # For each cluster center, fit surrogate model
        for cluster_index in range(cluster_centers.shape[0]):
            center = cluster_centers_hist[cluster_index,:]
            center_result = cluster_centers_hist_result[cluster_index,:]
            swarm_samples = pos_x[clustering_vector==cluster_index,]
            swarm_results = pos_results[clustering_vector==cluster_index,]
        
            # Append center result and centers to data for surrogate fitting
            function_inputs = np.vstack([swarm_samples,center])
            function_outputs = np.vstack([swarm_results,center_result])
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
           
        surrogate_recommendations = subopt_result
        
        vel = vel * self.config["vel_w"] + \
            r_p * self.config["p_w"] * (best_individual_position - pos_x) + \
            r_l * self.config["l_w"] * (best_local_position_current - pos_x) + \
            r_lo * self.config["lo_w"] * (best_local_position_overall - pos_x) + \
            self.config["surro_w"] * (surrogate_recommendations - pos_x)
        
        if use_sced:
            vel = vel + self.config["sced_w"] * (het_samples - pos_x) 
            
        self.swarm_state["pos_x"] = pos_x
        self.swarm_state["pos_results"] = pos_results
        self.swarm_state["surrogatesaves"] = surrogatesaves
        self.swarm_state["centersaves"] = centersaves
        self.swarm_state["pos_het"] = pos_het
        self.swarm_state["best_individual_position"] = best_individual_position
        self.swarm_state["best_individual_score"] = best_individual_score
        self.swarm_state["all_pos"] = all_pos
        self.swarm_state["all_results"] = all_results
        self.swarm_state["vel"] = vel
        self.swarm_state["surrogate_recommendations"] = surrogate_recommendations
        
        
#Example Usage
if __name__ == "__main__":
    
    objective_function_calls = 0
    
    def model_function(params,args):
        input_vals, check = args
        modval = params[0] * np.cos(input_vals * params[1]) + params[1] * np.sin(input_vals * params[0])
        #modval = params[0] + input_vals**2 * params[1]**2
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
    
    swarmer = SurrogateSwarm(opt_problem,
                               ydat,
                               2,
                               [0,0],
                               [1,1])
    for itt in range(20):
        swarmer.step_swarm()
        swarmer.swarm_state["surrogate_recommendations"]
        plot_optimizer_results_with_rec(swarmer.swarm_state["pos_x"],
                               xdat,
                               ydat,
                               swarmer.swarm_state["surrogatesaves"],
                               swarmer.swarm_state["centersaves"],
                               swarmer.swarm_state["surrogate_recommendations"],
                               optproblem = opt_problem
                               )
        # print(np.mean(swarmer.swarm_state["pos_x"],axis = 0))
        print(swarmer.swarm_state["surrogate_recommendations"])
        print(f"Objective Function Calls: {objective_function_calls}")
