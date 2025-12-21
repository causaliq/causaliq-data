# Preprocess - Data Preprocessing Utilities

The `preprocess` module provides utilities for cleaning and preparing data for Bayesian Network operations and causal discovery workflows. These functions help address common data quality issues that can interfere with structure learning and parameter estimation.

## Overview

Data preprocessing is often a critical step before applying causal discovery algorithms or fitting Bayesian Networks. Real-world datasets and even synthetic data can contain variables that provide no useful information or can cause algorithmic issues. The preprocess module provides utilities to identify and remove such problematic variables.

## Functions

### `remove_single_valued(bn, data) -> Tuple[BN, DataFrame, List[str]]`

Removes nodes from a Bayesian Network that contain only a single unique value in the corresponding data. This is particularly useful when working with synthetic data or datasets where some variables have become constant due to filtering or other preprocessing steps.

**Arguments:**

- `bn`: Bayesian Network object to modify (BN)
- `data`: Pandas DataFrame containing the data corresponding to the BN

**Returns:**

Tuple containing:
- **Modified BN**: New Bayesian Network with single-valued variables removed
- **Cleaned DataFrame**: Data with single-valued columns removed and categorical types applied
- **Removed Variables**: List of variable names that were removed (sorted)

**Raises:**

- `TypeError`: If data is not a pandas DataFrame
- `ValueError`: If removing single-valued variables would leave fewer than 2 multi-valued variables

**Functionality:**

1. **Variable Analysis**: Identifies columns in the data that have fewer than 2 unique values
2. **Network Restructuring**: Creates a new BN with problematic nodes removed from:
   - Node list 
   - Edge relationships (removes edges involving removed nodes)
   - Network structure
3. **Data Cleaning**: Removes corresponding columns from the DataFrame and ensures categorical typing
4. **Validation**: Ensures at least 2 variables remain after cleaning

**Usage Examples:**

```python
from causaliq_data.preprocess import remove_single_valued
from causaliq_core.bn.io import read_bn
import pandas as pd

# Load a Bayesian Network and some data
original_bn = read_bn("network.dsc")
data = pd.read_csv("dataset.csv")

# Remove single-valued variables
cleaned_bn, cleaned_data, removed_vars = remove_single_valued(original_bn, data)

print(f"Original variables: {len(original_bn.nodes)}")
print(f"Cleaned variables: {len(cleaned_bn.nodes)}")
print(f"Removed variables: {removed_vars}")

# Use the cleaned network and data for further analysis
# ... proceed with structure learning or parameter estimation
```

**Common Use Cases:**

### Synthetic Data Cleaning
```python
# When generating synthetic data, some variables might accidentally become constant
from causaliq_data.oracle import Oracle

# Generate synthetic data from a BN
oracle = Oracle(bn=network, N=1000)
synthetic_data = oracle.sample

# Check for and remove any single-valued variables that might have emerged
if any(synthetic_data.nunique() < 2):
    cleaned_bn, cleaned_data, removed = remove_single_valued(network, synthetic_data)
    print(f"Removed constant variables: {removed}")
```

### Real Data Preprocessing
```python
# Real datasets often have variables that become single-valued after filtering
original_data = pd.read_csv("survey_data.csv")

# After applying filters, some variables might become constant
filtered_data = original_data[original_data['age'] > 65]  # Example filter

# Remove any variables that became single-valued after filtering
if any(filtered_data.nunique() < 2):
    # Fit initial BN from filtered data structure
    from causaliq_data.pandas import Pandas
    data_adapter = Pandas(filtered_data)
    initial_bn = BN.fit(complete_graph(filtered_data.columns), data_adapter)
    
    # Clean up single-valued variables
    final_bn, final_data, removed = remove_single_valued(initial_bn, filtered_data)
```

**Technical Details:**

### Variable Detection
The function uses pandas' `nunique()` method to count unique values per column:
```python
single_valued = [col for col, count in data.nunique().items() if count < 2]
```

This identifies variables that have:
- Exactly 1 unique value (constant variables)
- 0 unique values (empty variables, though rare)

### Network Reconstruction
The cleaned Bayesian Network is reconstructed by:

1. **Node Filtering**: Removing problematic nodes from the node list
2. **Edge Filtering**: Keeping only edges where both source and target nodes remain
3. **Structure Preservation**: Maintaining all valid conditional dependencies
4. **Parameter Re-estimation**: Fitting parameters using the cleaned data

### Data Type Management
The function ensures proper categorical typing on the cleaned data, which is important for:
- Consistent handling in downstream BN operations
- Memory efficiency with categorical variables
- Compatibility with causaliq-data adapters

## Error Handling

The function provides comprehensive validation:

**Type Validation:**
```python
if not isinstance(data, DataFrame):
    raise TypeError("remove_single_valued_variables() bad arg type")
```

**Minimum Variable Requirements:**
```python
if len(data.columns) - len(remove) < 2:
    raise ValueError("remove_single_valued_variables() - <2 multi-valued")
```

This ensures that the resulting dataset has enough variables for meaningful analysis, as Bayesian Networks require at least two variables to represent relationships.

## Integration with CausalIQ Workflow

The preprocess module integrates seamlessly with other causaliq-data components:

### With Data Adapters
```python
from causaliq_data.pandas import Pandas
from causaliq_data.preprocess import remove_single_valued

# Start with raw data
raw_data = pd.read_csv("dataset.csv")
data_adapter = Pandas(raw_data)

# If needed, clean up single-valued variables
if any(raw_data.nunique() < 2):
    # Create initial BN structure (e.g., complete graph for testing)
    initial_bn = BN.fit(complete_graph(raw_data.columns), data_adapter)
    cleaned_bn, cleaned_data, removed = remove_single_valued(initial_bn, raw_data)
    
    # Create new adapter with cleaned data
    final_adapter = Pandas(cleaned_data)
```

### With Structure Learning
```python
from causaliq_discovery import pc_algorithm  # Example

# Preprocess data before structure learning
cleaned_bn, cleaned_data, removed_vars = remove_single_valued(initial_bn, raw_data)

# Apply structure learning on cleaned data
learned_structure = pc_algorithm(cleaned_data)
```

### With Independence Testing
```python
from causaliq_data.indep import indep

# Clean data first to avoid issues with constant variables
cleaned_bn, cleaned_data, removed = remove_single_valued(bn, data)

# Now independence tests will work properly (no zero-variance variables)
test_result = indep("X", "Y", ["Z"], cleaned_data)
```

## Performance Considerations

- **Memory Efficiency**: Creates new objects rather than modifying in-place to preserve original data
- **Categorical Optimization**: Applies categorical typing for memory efficiency
- **Network Reconstruction**: Uses efficient BN.fit() method for parameter estimation
- **Minimal Overhead**: Only processes data if single-valued variables are detected

## Best Practices

1. **Early Preprocessing**: Apply before structure learning to avoid algorithmic issues
2. **Validation**: Always check the removed variable list to ensure expected variables weren't accidentally removed
3. **Documentation**: Record which variables were removed for reproducibility
4. **Threshold Consideration**: The function removes variables with <2 unique values, which is appropriate for discrete/categorical data

## See Also

- **[Data](data.md)**: Base data handling interface
- **[Pandas](pandas.md)**: DataFrame-based data adapter
- **[Independence Testing](indep.md)**: Statistical independence tests
- **CausalIQ Core**: Bayesian Network and DAG functionality