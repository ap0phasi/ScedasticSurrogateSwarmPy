
import os
os.chdir("..")
print(os.getcwd())
from sssopy.surrogateeval import *

if __name__ == "__main__":
    # Calculate outputs using the Hartmann 6-dimensional function
    def hartmann6d_function(x):
        # Hartmann 6-dimensional function coefficients
        alpha = np.array([1.0, 1.2, 3.0, 3.2])

        A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14]])

        P = 10**-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        exp_term = np.exp(-np.sum(A * (x - P)**2, axis=1))
        return -np.sum(alpha * exp_term)

    def quadratic_function(x):
        return x[0]**2 + x[1]**2 + x[0] * x[1]

    test_surrogate_on_function(hartmann6d_function, 6)
    test_surrogate_on_function(quadratic_function, 2)