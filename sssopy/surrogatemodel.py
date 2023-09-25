import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class SurrogateModel:
    """
    A class to create and use surrogate models.
    
    Attributes:
        model: The surrogate model.
    """
    def __init__(self):
        self.model = None

    def fit(self, inputs, outputs, center, model_type):
        """
        Fit the surrogate model to the provided data.
        
        Args:
            inputs (numpy.ndarray): Input data for fitting.
            outputs (numpy.ndarray): Output data for fitting.
            center (numpy.ndarray): Centering value for inputs.
            model_type (str): Type of model to fit ('rbf', 'quadratic', 'linear').
        
        Raises:
            ValueError: If an invalid model_type is provided.
        """
        # Center the inputs
        centered_inputs = inputs - center
        
        # Radial Basis Function Interpolation
        if model_type == 'rbf':
            self.model = RBFInterpolator(centered_inputs, outputs)
        
        # Second Order Polynomial Regression
        elif model_type == 'quadratic':
            order = 2 # 2nd order polynomial
            self.model = make_pipeline(PolynomialFeatures(order), LinearRegression())
            self.model.fit(centered_inputs, outputs)
            
        #Linear Regression
        elif model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(centered_inputs, outputs)
        else:
            raise ValueError("Invalid model_type. Supported types: 'rbf', 'quadratic', 'linear'")

    def predict(self, inputs, center):
        """
        Predict using the fitted surrogate model.

        Args:
            inputs (numpy.ndarray): Input data for prediction.
            center (numpy.ndarray): Centering value for inputs.

        Returns:
            numpy.ndarray: Predicted values.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        # Center the inputs
        centered_inputs = inputs - center
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        if hasattr(self.model, 'predict'): # Some models use predict, some don't
            return self.model.predict(centered_inputs)
        else:
            return self.model(centered_inputs)
    
class SurrogateModelSelector:
    """
    A class to select and fit surrogate models based on fitting criteria.
    
    Attributes:
        fitopts (list): List of model types to consider.
        fit_threshold (float): Fitting accuracy threshold.
        models (list): List of fitted surrogate models.
        error_metric (str): Error metric for model selection.
    """
    def __init__(self, fit_threshold=0.2, error_metric='absolute'):
        self.fitopts = ['linear', 'quadratic', 'rbf']
        self.fit_threshold = fit_threshold
        self.models = []
        self.error_metric = error_metric

    def calculate_error(self, predictions, actuals):
        """
        Calculate the error between predicted and actual values.
        
        Args:
            predictions (numpy.ndarray): Predicted values.
            actuals (numpy.ndarray): Actual values.
            error_metric (str): Type of error metric ('absolute', 'squared', 'rmse').
        
        Returns:
            float: Calculated error value.
        
        Raises:
            ValueError: If an invalid error_metric is provided.
        """
        return calculate_error(predictions, actuals, self.error_metric)

    def fit_models(self, inputs, outputs, center):
        """
        Fit surrogate models based on fitting criteria.
        
        Args:
            inputs (numpy.ndarray): Input data for fitting.
            outputs (numpy.ndarray): Output data for fitting.
            center (numpy.ndarray): Centering value for inputs.
        
        Raises:
            ValueError: If fitting fails for any model.
        """
        fit_errors = []
        for fit_type in self.fitopts:
            surrogate = SurrogateModel()
            try:
                surrogate.fit(inputs, outputs, center, fit_type)
                predictions = surrogate.predict(inputs, center)
                fit_errors.append(self.calculate_error(predictions, outputs))
                # Check the current error against the previous if it exists
                if fit_errors[-1] <= fit_errors[max(-2,-len(fit_errors))]:
                    self.models.append(surrogate)
                else:
                    break # If accuracy does not improve, do not continue
                if fit_errors[-1] < self.fit_threshold:
                    break # If accuracy within threshold, do not continue
            except Exception as e:
                print(f"Fitting failed for {fit_type} model: {e}")
                continue

def calculate_error(predictions, actuals, error_metric):
    """
    Calculate the error between predicted and actual values.

    Args:
        predictions (numpy.ndarray): Predicted values.
        actuals (numpy.ndarray): Actual values.
        error_metric (str): Type of error metric ('absolute', 'squared', 'rmse').

    Returns:
        float: Calculated error value.

    Raises:
        ValueError: If an invalid error_metric is provided.
    """
    if error_metric == 'absolute':
        return np.sum(np.abs(predictions - actuals))
    elif error_metric == 'squared':
        return np.sum(np.square(predictions - actuals))
    elif error_metric == 'rmse':
        return np.sqrt(np.mean(np.square(predictions - actuals)))
    else:
        raise ValueError("Invalid error metric.")

if __name__ == "__main__":
    # Example usage
    n = 10
    samp = 100
    inputs = np.random.rand(samp, n)
    outputs = np.random.rand(samp, 1)
    center = np.random.rand(1,n)

    # Create an instance of SurrogateModelSelector with different error metric
    selector = SurrogateModelSelector(fit_threshold=1)

    # Fit and store models based on accuracy threshold and error metric
    selector.fit_models(inputs, outputs, center)

    # Access and use the stored models as needed

    calculate_error(selector.models[2].predict(inputs, center),outputs,"rmse")