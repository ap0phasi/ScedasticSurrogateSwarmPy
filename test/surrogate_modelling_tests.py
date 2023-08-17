# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:52:28 2023

@author: johnm
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class SurrogateModel:
    def __init__(self):
        self.model = None

    def fit(self, inputs, outputs, model_type):
        if model_type == 'rbf':
            self.model = RBFInterpolator(inputs, outputs)
        elif model_type == 'quadratic':
            order = 2
            self.model = make_pipeline(PolynomialFeatures(order), LinearRegression())
            self.model.fit(inputs, outputs)
        elif model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(inputs, outputs)
        else:
            raise ValueError("Invalid model_type. Supported types: 'rbf', 'quadratic', 'linear'")

    def predict(self, inputs):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        if hasattr(self.model, 'predict'):
            return self.model.predict(inputs)
        else:
            return self.model(inputs)

# Example usage
n = 10
samp = 100
inputs = np.random.rand(samp, n)
outputs = np.random.rand(samp, 1)

# Create an instance of SurrogateModel
surrogate = SurrogateModel()

# Fit using RBF interpolation
surrogate.fit(inputs, outputs, 'rbf')

# Predict using the fitted model
predictions_rbf = surrogate.predict(inputs)

# Fit using quadratic regression
surrogate.fit(inputs, outputs, 'quadratic')

# Predict using the fitted model
predictions_quadratic = surrogate.predict(inputs)

# Fit using linear regression
surrogate.fit(inputs, outputs, 'linear')

# Predict using the fitted model
predictions_linear = surrogate.predict(inputs)

fitopts = ['linear','quadratic','rbf']
acc_thresh = 0.2
models = []
for ift in fitopts:
    surrogate = SurrogateModel()
    surrogate.fit(inputs, outputs, ift)
    fiterror = sum(abs(surrogate.predict(inputs)-outputs))[0]
    models.append(surrogate)
    if fiterror < acc_thresh:
        break
    
class SurrogateModelSelector:
    def __init__(self, fit_threshold=0.2, error_metric='absolute'):
        self.fitopts = ['linear', 'quadratic', 'rbf']
        self.fit_threshold = fit_threshold
        self.models = []
        self.error_metric = error_metric

    def calculate_error(self, predictions, actuals):
        return calculate_error(predictions, actuals, self.error_metric)

    def fit_models(self, inputs, outputs):
        for fit_type in self.fitopts:
            surrogate = SurrogateModel()
            surrogate.fit(inputs, outputs, fit_type)
            predictions = surrogate.predict(inputs)
            fit_error = self.calculate_error(predictions, outputs)
            self.models.append(surrogate)
            if fit_error < self.fit_threshold:
                break

def calculate_error(predictions, actuals, error_metric):
    if error_metric == 'absolute':
        return np.sum(np.abs(predictions - actuals))
    elif error_metric == 'squared':
        return np.sum(np.square(predictions - actuals))
    elif error_metric == 'rmse':
        return np.sqrt(np.mean(np.square(predictions - actuals)))
    else:
        raise ValueError("Invalid error metric.")

# Create an instance of SurrogateModelSelector with different error metric
selector = SurrogateModelSelector()

# Fit and store models based on accuracy threshold and error metric
selector.fit_models(inputs, outputs)

# Access and use the stored models as needed
model_0 = selector.models[0]
model_1 = selector.models[1]
model_2 = selector.models[2]

calculate_error(selector.models[2].predict(inputs),outputs,"rmse")


