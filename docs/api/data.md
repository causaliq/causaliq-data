# Data - Abstract Base Class

The `Data` class serves as the abstract base class for all data adapters in CausalIQ Data. It extends the BNFit interface from causaliq-core and provides the foundation for the plug-in data adapter architecture.

## Class Definition

```python
class Data(BNFit):
    """Top level data object that implements BNFit interface.

    Extends BNFit interface with additional methods needed for
    causal structure learning algorithms.
    """
```

## Key Attributes

- **`elapsed`**: Elapsed time for operations
- **`order`**: Order in which nodes should be processed
- **`ext_to_orig`**: Mapping from external to original node names
- **`orig_to_ext`**: Mapping from original to external node names  
- **`dstype`**: Overall dataset type (categorical/continuous/mixed)

## Core Methods

### Node Order Management

#### `set_order(order: Tuple[str, ...]) -> None`
Sets the processing order of nodes to the specified sequence.

**Arguments:**
- `order`: New processing order as a tuple of external node names

**Raises:**
- `TypeError`: For invalid argument types
- `ValueError`: If order contains invalid node names

#### `get_order() -> Tuple[str, ...]`
Returns the current processing order using external node names.

#### `randomise_order(seed: int) -> None`
Randomizes the processing order of nodes using the specified seed.

**Arguments:**
- `seed`: Randomization seed (must be non-negative)

### Name Randomisation

#### `randomise_names(seed: Optional[int]) -> None`
Randomizes node names for sensitivity testing. When `seed=None`, reverts to original names.

**Arguments:**
- `seed`: Randomization seed or None to revert

## Abstract Methods

The following methods must be implemented by concrete subclasses:

### Sample Management

- **`set_N(N, seed, random_selection)`**: Set working sample size with optional randomization
- **`_update_sample(old_N, old_ext_to_orig)`**: Update sample after parameter changes

### Data Access

- **`marginals(node, parents, values_reqd)`**: Get marginal distributions
- **`values(nodes)`**: Return values for specified nodes  
- **`as_df()`**: Convert to pandas DataFrame representation

### Persistence

- **`write(filename)`**: Write data to file

## Properties (from BNFit)

### Node Information

- **`nodes`**: Internal (original) node names
- **`node_types`**: Node type mapping (e.g., `{node1: type1, ...}`)
- **`node_values`**: Value counts for categorical nodes

### Sample Information

- **`N`**: Current sample size being used
- **`sample`**: Access to underlying data sample

## Type System

### DatasetType Enumeration
```python
class DatasetType(StrEnum):
    CATEGORICAL = "categorical"  # All categorical variables
    CONTINUOUS = "continuous"    # All float variables  
    MIXED = "mixed"             # Mixed categorical and numeric
```

### VariableType Enumeration  
```python
class VariableType(StrEnum):
    INT16 = "int16"
    INT32 = "int32" 
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    CATEGORY = "category"
```

## Internal Methods

### `_set_dstype() -> None`
Determines the overall dataset type from individual node types. Sets `dstype` to:

- `CATEGORICAL`: If all variables are categorical
- `CONTINUOUS`: If all variables are numeric  
- `MIXED`: If variables are mixed types

### `_generate_random_names(seed: Optional[int]) -> None`
Generates randomized external names for nodes using the format `X###NNNNNN` where:

- `###` is a zero-padded random integer
- `NNNNNN` is the first 6 characters of the original name

When `seed=None`, reverts mappings back to original names.

## Usage Patterns

### Basic Inheritance
```python
class CustomData(Data):
    def __init__(self, source):
        super().__init__()
        # Initialize from source
        
    def set_N(self, N, seed=None, random_selection=False):
        # Implement sample size setting
        
    def marginals(self, node, parents, values_reqd=False):
        # Implement marginal computation
        
    # ... implement other abstract methods
```

### Name Randomization Workflow
```python
# Store original order for later restoration
original_order = data.get_order()

# Randomize names for sensitivity testing
data.randomise_names(seed=42)

# Run algorithm with randomized names
results_randomized = run_algorithm(data)

# Restore original names
data.randomise_names(seed=None)

# Compare results
compare_sensitivity(results_original, results_randomized)
```

## Integration with BNFit

The Data class fully implements the BNFit interface, enabling seamless integration with causaliq-core components:

- **Parameter Estimation**: Compatible with BN parameter fitting algorithms
- **Score Calculation**: Provides necessary marginals for score-based learning
- **Constraint Testing**: Supports conditional independence testing workflows