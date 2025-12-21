# CausalIQ Data API Reference

The CausalIQ Data API provides a unified interface for data handling in causal discovery workflows. The API is built around a plug-in architecture with concrete implementations for different data backends.

## Core Design

All data adapters implement the [`Data`](data.md) abstract base class, which extends the BNFit interface from causaliq-core. This ensures consistent behavior across different data sources while allowing backend-specific optimizations.

## Module Structure

### [`Data`](data.md) - Abstract Base Class

The foundational abstract class that defines the core interface for all data adapters. Provides:

- **Node Management**: Consistent handling of variable names and ordering
- **Randomisation Framework**: Built-in support for data and name randomisation 
- **BNFit Interface**: Full compatibility with Bayesian Network fitting operations
- **Type System**: Unified variable type handling across data sources

### [`Pandas`](pandas.md) - DataFrame-Based Adapter

A concrete implementation that wraps pandas DataFrames for flexible data handling:

- **Rich Type Support**: Native pandas categorical and numeric types
- **File I/O**: Direct CSV reading with compression support
- **Data Validation**: Comprehensive missing data and type checking
- **Memory Efficiency**: Smart sampling and subsetting without data duplication

### [`NumPy`](numpy.md) - High-Performance Array Adapter  

A high-performance implementation using NumPy arrays for computational efficiency:

- **Optimized Counting**: Fast categorical data counting using `bincount`
- **Memory Management**: Efficient handling of large datasets with minimal copying
- **Type Optimization**: Automatic selection of appropriate numeric types
- **Advanced Sampling**: Multiple strategies for data subset selection and randomization

### [`Oracle`](oracle.md) - Synthetic Data Generator

A specialized adapter for generating synthetic data from known Bayesian Networks:

- **BN Integration**: Direct integration with causaliq-core BN objects
- **Parameter Access**: Direct access to true conditional probability tables
- **Testing Support**: Ideal for algorithm validation and benchmarking
- **Simulation Control**: Flexible sample size management for experiments

### [`Score`](score.md) - Scoring Functions for Causal Structure Learning

A comprehensive module providing scoring functions for evaluating Bayesian networks and DAGs:

- **Multiple Score Types**: Support for entropy-based, Bayesian, and Gaussian scoring methods
- **Categorical Scoring**: BIC, AIC, log-likelihood, BDE, K2, and other Bayesian scores
- **Gaussian Scoring**: BGE, Gaussian BIC, and Gaussian log-likelihood for continuous data
- **Network Evaluation**: Complete DAG and Bayesian Network scoring with per-node breakdowns
- **Parameter Validation**: Automatic parameter checking and default value assignment

### [`Independence Testing`](indep.md) - Probabilistic Independence Tests

Statistical independence testing functionality for constraint-based causal discovery:

- **Multiple Test Statistics**: Chi-squared (X²) and Mutual Information (MI) tests
- **Conditional Independence**: Support for testing X ⊥ Y | Z with arbitrary conditioning sets
- **Flexible Data Sources**: Works with pandas DataFrames, data files, or Bayesian Network parameters
- **Robust Validation**: Comprehensive argument checking and error handling
- **Integration Ready**: Designed for use in PC, FCI, and other constraint-based algorithms

### [`Preprocess`](preprocess.md) - Data Preprocessing Utilities

Data cleaning and preparation utilities for Bayesian Network workflows:

- **Single-Valued Variable Removal**: Automatic detection and removal of constant variables
- **Network Restructuring**: Intelligent BN reconstruction after variable removal
- **Data Validation**: Ensures minimum variable requirements for meaningful analysis
- **Categorical Optimization**: Proper type handling for downstream operations
- **Integration Support**: Seamless workflow integration with data adapters and structure learning

## Common Patterns

### Data Loading
```python
from causaliq_data import Pandas, NumPy

# Load from CSV file
data = Pandas.read("dataset.csv", dstype="categorical")

# Convert to NumPy for performance
numpy_data = NumPy.from_df(data.as_df(), dstype="categorical")
```

### Randomisation Workflows
```python
# Randomize node names for sensitivity testing
data.randomise_names(seed=42)

# Randomize processing order
data.randomise_order(seed=123)

# Set working sample size with randomization
data.set_N(1000, seed=456, random_selection=True)
```

### Statistical Operations
```python
# Get marginal distributions
marginals = data.marginals("target_node", {"parent1": 0, "parent2": 1})

# Access value counts for categorical variables
counts = data.node_values["categorical_var"]

# Get unique value combinations
unique_vals, counts = data.unique(("var1", "var2"), num_vals)
```

## Type System

The API supports a comprehensive type system for different variable types:

- **Categorical**: `VariableType.CATEGORY` for discrete variables
- **Integers**: `INT16`, `INT32`, `INT64` for integer data
- **Floats**: `FLOAT32`, `FLOAT64` for continuous variables

Dataset types are automatically inferred:

- **Categorical**: All variables are categorical
- **Continuous**: All variables are numeric
- **Mixed**: Combination of categorical and numeric variables

## Performance Considerations

### Memory Efficiency

- Original data is preserved separately from working samples
- Lazy evaluation of expensive operations
- Strategic use of data views vs copies

### Computational Optimization

- NumPy adapter provides the best performance for large datasets
- Optimized algorithms for unique value detection and counting
- Efficient handling of categorical data through integer encoding

### Scalability

- Support for working with data subsets without loading entire datasets
- Memory-conscious type selection based on data characteristics
- Configurable thresholds for algorithm selection

## Error Handling

All adapters provide comprehensive error handling with descriptive messages:

- **Type Validation**: Strict checking of argument types and values
- **Data Validation**: Detection of missing data, invalid formats, and size constraints
- **State Consistency**: Validation of internal state consistency across operations