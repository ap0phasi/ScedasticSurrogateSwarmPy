import numpy as np

from sampling import calculate_sampling_range, latin_hypercube_within_range
from surrogatemodel import *
from optplotter import plot_optimizer_results, plot_optimizer_results_nosurrogate
from problem import SSSoProblem
from scedasticity import *
from modelresults import *

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
            "locality" : 0.3,
            "p_w" : 0.3,
            "l_w" : 0.6,
            "lo_w" : 0.6,
            "vel_w" : 0.5 
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
        
        vel = np.random.uniform(-0.001, 0.001, size=(self.config["swarm_size"], self.param_len))
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
        
        vel = vel * self.config["vel_w"] + \
            r_p * self.config["p_w"] * (best_individual_position - pos_x) + \
            r_l * self.config["l_w"] * (best_local_position_current - pos_x) + \
            r_lo * self.config["lo_w"] * (best_local_position_overall - pos_x)
            
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
        plot_optimizer_results_nosurrogate(swarmer.swarm_state["pos_x"],
                               xdat,
                               ydat,
                               optproblem = opt_problem)
        print(np.mean(swarmer.swarm_state["pos_x"],axis = 0))
        print(f"Objective Function Calls: {objective_function_calls}")
