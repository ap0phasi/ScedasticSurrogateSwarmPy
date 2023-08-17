# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:14:24 2023

@author: johnm
"""
import numpy as np
from scipy.interpolate import RBFInterpolator

n = 10
samp = 100

inputs = np.random.rand(samp,n)
outputs = np.random.rand(samp,1)

rbfinterp = RBFInterpolator(inputs, outputs)

rbfinterp(inputs)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Interpolation order (2nd order polynomial)
order = 2

# Create a pipeline with PolynomialFeatures and LinearRegression
quadmodel = make_pipeline(PolynomialFeatures(order), LinearRegression())

# Fit the model to the data
quadmodel.fit(inputs,outputs)

#Linear Model Regression
linmodel = LinearRegression()
linmodel.fit(inputs,outputs)
