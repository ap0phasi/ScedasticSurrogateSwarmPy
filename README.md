# Scedastic Surrogate Swarm Optimization - Python Implementation

This package provides an R implementation of Scedastic Surrogate Swarm Optimization (SSSO), a powerful meta-heuristic optimizer designed to enhance social learning algorithms using surrogate optimization. SSSO leverages machine learning techniques to construct surrogate models, which approximate black-box functions. These surrogates enable efficient solving of sub-optimization problems while sharing learned behavior within swarm subgroups.

## Key Characteristics:

### Surrogate Model Approximation

As the swarm explores the model sample space, it constructs surrogate models to represent relationships between model inputs and outputs.

### Surrogate Sub-Optimization

The surrogate models are harnessed to solve sub-optimization problems at each step of the swarm's solution process. This provides recommendations for the swarm's next positions without requiring additional costly evaluations of the model function. 

### Scedastic Enforcement

Utilizing heteroscedastic loss of the swarm population, the algorithm enforces dispersion by backpropagating heteroscedastic loss through surrogate model-derived sensitivity matrices. This drives swarm agents to spread in dimensions that reduce heteroscedastic loss, ensuring adequate sampling of model responses and enabling uncertainty propagation.

### Random or Search Initialization

In lieu of using a full swarm, SSSO supports single- or multi- agent searches that use surrogate model approximation. 
For simple problems, this search can be used on its own, while in more complex applications it can be used to initialize the
swarm optimizer where all learned surrogates are provided to the swarm.