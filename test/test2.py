import numpy as np
from scipy.interpolate import RBFInterpolator

n = 10
samp = 100

inputs = np.random.rand(samp,n)
outputs = np.random.rand(samp,1)

rbfinterp = RBFInterpolator(inputs, outputs)

rbfinterp(inputs)

from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

pft = PolynomialFeatures(degree=2).fit(inputs)