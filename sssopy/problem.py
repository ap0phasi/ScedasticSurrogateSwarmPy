import numpy as np

class SSSoProblem:
    def __init__(self, model_function, eq_constraints=None, ineq_constraints=None, args=None):
        self.model_function = model_function
        self.ineq_constraints = ineq_constraints if ineq_constraints is not None else self.default_ineq_constraints
        self.eq_constraints = eq_constraints if eq_constraints is not None else self.default_eq_constraints
        self.args = args
        
    def eval_function(self, params):
        return self.model_function(params, self.args)
    
    def eval_ineq(self, params):
        return self.ineq_constraints(params)
    
    def eval_eq(self, params):
        return self.eq_constraints(params)
    
    @staticmethod
    def default_ineq_constraints(params):
        return [0]
    
    @staticmethod
    def default_eq_constraints(params):
        return [0]
    
if __name__ == "__main__":
    def model_function(params,args):
        input_vals = args
        modval = params[0] * np.cos(input_vals * params[1]) + params[1] * np.sin(input_vals * params[0])
        #modval = params[0] + xdat**2 * params[1]**2
        return modval
    
    def ineq_constraints(params):
        return [0]
    
    def eq_constraints(params):
        return [0]
    
    xdat = np.arange(1, 100.5, 0.5)
    params = np.array([0.32, 0.4])
    
    opt_problem = SSSoProblem(model_function = model_function,
                              ineq_constraints = ineq_constraints,
                              eq_constraints = eq_constraints,
                              args = (xdat)
                              )
    
    ydat = opt_problem.eval_function(params)
    print(ydat)
    ydat = model_function(params,xdat)
    print(ydat)
    
    opt_problem = SSSoProblem(model_function = model_function,
                              args = (xdat)
                              )