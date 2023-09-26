import matplotlib.pyplot as plt
import numpy as np

from surrogateeval import eval_surrogate

def plot_optimizer_results(pos_x,xdat,ydat,surrogatesaves,centersaves,model_function):
    plt.scatter(xdat,ydat,color="black",label = "Observed")
    for i, row in enumerate(pos_x):
        testlines = model_function(row)
        # Label the first line in the loop only
        label = "Modeled" if i == 0 else None
        plt.plot(xdat, testlines, color="grey", label=label)
    predicted_out = eval_surrogate(np.array([pos_x[0,:]]),surrogatesaves,centersaves)
    plt.plot(xdat,predicted_out[0],color = "blue", label = "Predicted")
    actual_out = model_function(pos_x[0,:])
    plt.plot(xdat,actual_out,color = "red", label = "Actual")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),ncol = 4)
    plt.pause(1)
    plt.close()