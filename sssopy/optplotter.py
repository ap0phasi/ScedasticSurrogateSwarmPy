import matplotlib.pyplot as plt
import numpy as np

from .surrogateeval import eval_surrogate

def plot_optimizer_results(pos_x,xdat,ydat,surrogatesaves,centersaves,optproblem):
    plt.scatter(xdat,ydat,color="black",label = "Observed")
    
    for i, row in enumerate(pos_x):
        testlines = optproblem.eval_function(row)
        # Label the first line in the loop only
        label = "Modeled" if i == 0 else None
        plt.plot(xdat, testlines, color="grey", label=label)
        
    predicted_out = eval_surrogate(np.array([pos_x[0,:]]),surrogatesaves,centersaves)
    plt.plot(xdat,predicted_out[0],color = "blue", label = "Predicted")
    actual_out = optproblem.eval_function(pos_x[0,:])
    plt.plot(xdat,actual_out,color = "red", label = "Actual")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),ncol = 4)
    plt.pause(0.01)
    plt.close()
    
def plot_optimizer_results_with_rec(pos_x,xdat,ydat,surrogatesaves,centersaves,surrogate_recommendation,optproblem):
    plt.scatter(xdat,ydat,color="black",label = "Observed")
    
    for i, row in enumerate(pos_x):
        testlines = optproblem.eval_function(row)
        # Label the first line in the loop only
        label = "Modeled" if i == 0 else None
        plt.plot(xdat, testlines, color="grey", label=label)
        
    surrogate_input = np.array([surrogate_recommendation])
    predicted_out = eval_surrogate(surrogate_input,surrogatesaves,centersaves)
    plt.plot(xdat,predicted_out[0],color = "blue", label = "Predicted")
    actual_out = optproblem.eval_function(surrogate_recommendation)
    plt.plot(xdat,actual_out,color = "red", label = "Actual")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),ncol = 4)
    plt.pause(0.01)
    plt.close()
        
def plot_optimizer_results_nosurrogate(pos_x,xdat,ydat,optproblem):
    
    for i, row in enumerate(pos_x):
        testlines = optproblem.eval_function(row)
        # Label the first line in the loop only
        label = "Modeled" if i == 0 else None
        plt.plot(xdat, testlines, color="grey", label=label, zorder = 1)
        
    plt.scatter(xdat,ydat,color="black",label = "Observed", zorder = 2)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),ncol = 4)
    plt.pause(0.01)
    plt.close()
