# Architecture Overview

## CausalIQ Ecosystem

causaliq-data is a component of the overall [CausalIQ ecosystem architecture](https://causaliq.org/projects/ecosystem_architecture/), providing the data layer foundation for causal discovery algorithms.

## Core Architecture: Plug-in Data Adapters

CausalIQ Data is built around a **plug-in data adapter architecture** that enables seamless integration of different data sources and formats through a unified interface. This design provides flexibility while maintaining consistent performance characteristics across different data backends.

### Abstract Base Class (`Data`)

The `Data` class defines the core BNFit interface that all data adapters must implement. This abstract base class:

- Extends the BNFit interface from causaliq-core for Bayesian Network fitting
- Defines standard methods for data access, manipulation, and randomisation
- Ensures consistent behavior across all concrete implementations
- Provides common functionality like node ordering and name randomisation

### Concrete Data Adapters

The architecture supports multiple data adapters, each optimized for different use cases:

1. **Pandas Adapter** - For standard tabular data with rich type support
2. **NumPy Adapter** - For high-performance numerical operations on large datasets
3. **Oracle Adapter** - For synthetic data generation from known Bayesian Networks

## Key Architectural Features

### In-Memory Counting and Optimization

The data adapters implement sophisticated **in-memory counting mechanisms** for efficient statistical operations:

- **Categorical Data Counting**: Optimized binning and counting for discrete variables using NumPy's `bincount` functionality
- **Value Combination Caching**: Intelligent caching of unique value combinations to avoid recomputation
- **Memory-Efficient Storage**: Strategic use of appropriate data types (int16, int32, float32, float64) to minimize memory footprint
- **Sample Subset Management**: Efficient handling of data subsets without copying underlying arrays

### Data Randomisation Capabilities

The architecture provides comprehensive **data randomisation features** essential for causal discovery validation:

#### Node Name Randomisation
- **Purpose**: Assess algorithm sensitivity to variable naming
- **Implementation**: Systematic generation of randomized node names while preserving data relationships
- **Reversibility**: Ability to revert to original names for result interpretation

#### Sample Order Randomisation
- **Purpose**: Test algorithm stability across different data presentations  
- **Methods**: Multiple randomization strategies (full shuffle, random selection, seeded ordering)
- **Seed Management**: Deterministic randomization for reproducible experiments

#### Node Processing Order Randomisation
- **Purpose**: Evaluate algorithm sensitivity to variable processing order
- **Flexibility**: Support for custom orderings or random permutations
- **Preservation**: Maintains data integrity while changing algorithmic perspectives

### Performance Optimizations

#### Lazy Evaluation
- Sample subsets are computed on-demand rather than pre-computed
- Type conversions happen only when necessary (e.g., float64 conversion for continuous data during scoring)

#### Memory Management
- Original data is preserved separately from working samples
- Efficient copy-on-write semantics where possible
- Strategic use of views vs copies to minimize memory overhead

#### Algorithmic Efficiency
- Optimized unique value detection using both numpy.unique and custom counting approaches
- Threshold-based algorithm selection for optimal performance across different data sizes
- In-place operations where safe and beneficial

## Integration Points

### CausalIQ Core Integration
- Implements BNFit interface for seamless integration with Bayesian Network fitting algorithms
- Provides marginal distributions and conditional independence testing capabilities
- Supports parameter estimation workflows

### CausalIQ Discovery Integration
- Supplies objective functions for score-based structure learning
- Provides conditional independence tests for constraint-based algorithms
- Enables stability testing through randomisation features

### CausalIQ Workflow Integration
- Supports experimental workflows requiring data randomization
- Provides consistent interfaces for batch processing
- Enables reproducible research through seed management

## Design Principles

1. **Separation of Concerns**: Data access, transformation, and algorithm logic are clearly separated
2. **Performance by Design**: Architecture prioritizes computational efficiency for large-scale causal discovery
3. **Extensibility**: New data adapters can be added without changing existing code
4. **Type Safety**: Comprehensive type checking and validation throughout the pipeline
5. **Reproducibility**: Built-in support for seeded randomization and deterministic operations