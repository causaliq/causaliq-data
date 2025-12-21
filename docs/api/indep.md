# Independence Testing - Probabilistic Independence Tests

The `indep` module provides comprehensive statistical independence testing functionality for causal discovery workflows. It implements multiple test statistics and supports both conditional and unconditional independence testing on data or Bayesian Network parameters.

## Overview

Independence tests are fundamental to constraint-based causal discovery algorithms (PC, FCI, etc.) and structure learning validation. The module supports:

- **Multiple Test Statistics**: Chi-squared (X²) and Mutual Information (MI) tests
- **Conditional Independence**: Testing X ⊥ Y | Z with arbitrary conditioning sets
- **Flexible Data Sources**: Works with pandas DataFrames, data files, or BN parameters
- **Comprehensive Validation**: Robust argument checking and error handling

## Constants

```python
TESTS = ["mi", "x2"]  # Supported test types
TEST_PARAMS = {"alpha": 0.05}  # Default test parameters  
MIN_P_VALUE = 1e-30  # Minimum p-value threshold
```

## Core Functions

### `indep(x, y, z, data, bn=None, N=1000000000, types="mi") -> DataFrame`

The main independence testing function that performs statistical tests to determine if variables x and y are independent, optionally conditional on variables z.

**Arguments:**

- `x`: Name of the first variable (string)
- `y`: Name of the second variable (string) 
- `z`: Name(s) of conditioning variables (string, list of strings, or None)
- `data`: Data source (pandas DataFrame, file path string, or None)
- `bn`: Bayesian Network for synthetic testing (BN object or None) 
- `N`: Sample size when using BN parameters (int, default: 1,000,000,000)
- `types`: Test statistic type(s) to compute (string or list, default: "mi")

**Returns:**

DataFrame with independence test results where:

- **Columns**: Test types (e.g., "mi", "x2")
- **Rows**: Test statistics ("statistic", "df", "p_value")

**Test Statistics:**

- **Chi-squared ("x2")**: Classical Pearson chi-squared test for independence
  - Formula: Σ((observed - expected)² / expected)
  - Asymptotically χ² distributed under null hypothesis of independence

- **Mutual Information ("mi")**: Information-theoretic measure of dependence  
  - Formula: 2 × Σ(observed × log(observed / expected))
  - Asymptotically χ² distributed (G-test statistic)

**Usage Examples:**

```python
from causaliq_data.indep import indep
import pandas as pd

# Load data
data = pd.read_csv("dataset.csv")

# Unconditional independence test
result = indep("X", "Y", None, data, types="x2")
print(f"p-value: {result.loc['p_value', 'x2']}")

# Conditional independence test  
result = indep("X", "Y", ["Z1", "Z2"], data, types=["mi", "x2"])

# Test using BN parameters
from causaliq_core.bn.io import read_bn
bn = read_bn("network.dsc")
result = indep("A", "B", "C", None, bn=bn, N=10000)
```

**Data Source Options:**

1. **DataFrame**: Direct pandas DataFrame input
2. **File Path**: Path to CSV file (loaded automatically)
3. **BN Parameters**: Uses conditional probability tables from Bayesian Network

### `check_test_params(params) -> Dict[str, Any]`

Validates and standardizes independence test parameters.

**Arguments:**

- `params`: Dictionary of test parameters to validate

**Supported Parameters:**

- `alpha`: Significance level for tests (float, 0 < alpha < 1, default: 0.05)

**Returns:**

Dictionary of validated parameters with defaults applied.

**Raises:**

- `TypeError`: If parameters have incorrect types
- `ValueError`: If parameter values are invalid

**Example:**

```python
from causaliq_data.indep import check_test_params

# Validate custom parameters
params = check_test_params({"alpha": 0.01})
print(params)  # {"alpha": 0.01}

# Apply defaults
params = check_test_params({})  
print(params)  # {"alpha": 0.05}
```

## Internal Functions

### `check_indep_args(...) -> Tuple[...]`

Internal function that validates and standardizes all arguments for independence tests.

**Key Validations:**

- Type checking for all arguments
- Variable name uniqueness
- Data/BN consistency checks
- Column/node existence validation  
- Sample size validation

### `_statistic(actuals, type) -> Tuple[int, float]`

Internal function that computes test statistics from contingency tables.

**Arguments:**

- `actuals`: 2D list representing contingency table counts
- `type`: Test statistic type ("x2" or "mi")

**Returns:**

Tuple of (degrees of freedom, test statistic value)

**Features:**

- Handles zero-count tables (returns 0.0 statistic)
- Robust error handling for malformed inputs
- Optimized computation for both test types

## Statistical Details

### Test Assumptions

Both chi-squared and mutual information tests assume:

1. **Categorical Variables**: All variables must be discrete/categorical
2. **Sufficient Sample Size**: Large enough samples for asymptotic properties
3. **Independent Observations**: Rows represent independent samples
4. **No Missing Data**: Complete case analysis only

### Degrees of Freedom

For contingency tables with dimensions r × c:

- **Degrees of Freedom**: (r-1) × (c-1)
- **Conditional Tests**: Sum across conditioning set combinations

### P-value Computation

P-values are computed using the chi-squared distribution:
```python
p_value = 1.0 - stats.chi2.cdf(statistic, degrees_of_freedom)
```

Values below `MIN_P_VALUE` (1e-30) are set to 0.0 for numerical stability.

## Error Handling

The module provides comprehensive error checking:

**Type Errors:**

- Non-string variable names
- Invalid data types for DataFrame/BN arguments
- Malformed conditioning sets or test type specifications

**Value Errors:**

- Duplicate variable names
- Variables not present in data/BN
- Negative sample sizes
- Unsupported or duplicate test types
- Empty test specifications

**File Errors:**

- Missing data files
- Malformed CSV data

## Integration with Causal Discovery

Independence tests are essential for:

### Constraint-Based Algorithms
```python
# PC Algorithm skeleton discovery
if indep("X", "Y", [], data)["mi"]["p_value"] > 0.05:
    # Remove edge X-Y
    pass

# Conditional independence for orientation  
if indep("X", "Y", ["Z"], data)["mi"]["p_value"] <= 0.05:
    # Orient edge based on dependence
    pass
```

### Structure Learning Validation
```python
# Validate learned structure
learned_bn = learn_structure(data)
for x, y in learned_bn.edges():
    parents_xy = list(set(learned_bn.parents(x) + learned_bn.parents(y)))
    test_result = indep(x, y, parents_xy, data)
    if test_result["mi"]["p_value"] > 0.05:
        print(f"Warning: {x}-{y} may be spurious")
```

### Synthetic Data Validation
```python  
# Test independence properties in generated data
true_bn = read_bn("true_network.dsc")
synthetic_data = generate_data(true_bn, N=5000)

# Verify independence assumptions hold
for node in true_bn.nodes:
    non_descendants = true_bn.non_descendants(node)
    for nd in non_descendants:
        parents = true_bn.parents(node)
        result = indep(node, nd, parents, synthetic_data)
        assert result["mi"]["p_value"] > 0.05, f"{node} should be independent of {nd}"
```

## Performance Considerations

### Computational Complexity

- **Contingency Table Construction**: O(n × k) where n = sample size, k = variables
- **Statistic Computation**: O(r × c) where r, c are table dimensions  
- **Conditional Tests**: Multiplicative in conditioning set size

### Memory Usage

- **Sparse Tables**: Efficient handling of sparse contingency tables
- **Batch Processing**: Processes all conditioning combinations efficiently
- **Memory Reuse**: Minimal copying in DataFrame operations

### Optimization Tips

1. **Batch Multiple Tests**: Use `types=["mi", "x2"]` for multiple statistics
2. **Limit Conditioning Sets**: Large conditioning sets increase computational cost
3. **Sample Size Management**: Use appropriate N values with BN parameters
4. **Data Preprocessing**: Pre-filter and clean data before testing

## See Also

- **[Data](data.md)**: Base data handling interface
- **[Pandas](pandas.md)**: DataFrame-based data adapter  
- **[Score](score.md)**: Structure scoring functions
- **CausalIQ Core**: Bayesian Network and DAG functionality