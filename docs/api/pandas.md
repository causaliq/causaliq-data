# Pandas - DataFrame-Based Data Adapter

The `Pandas` class provides a concrete implementation of the Data interface using pandas DataFrames as the underlying data storage. This adapter is ideal for exploratory data analysis and moderate-sized datasets where pandas' rich functionality is beneficial.

## Class Definition

```python
class Pandas(Data):
    """Data subclass which holds data in a Pandas dataframe.

    Args:
        df: Data provided as a Pandas dataframe.

    Attributes:
        df: Original Pandas dataframe providing data.
        dstype: Type of dataset (categorical/numeric/mixed).
    """
```

## Constructor

### `__init__(df: DataFrame) -> None`

Creates a new Pandas data adapter from a DataFrame.

**Arguments:**

- `df`: Pandas DataFrame containing the data

**Validation:**

- Minimum 2 rows and 2 columns required
- No missing data (NaN values) allowed
- All column names must be strings

**Raises:**

- `TypeError`: If df is not a pandas DataFrame
- `ValueError`: If DataFrame size or data validation fails

## Class Methods

### `read(filename, dstype, N=None, **kwargs) -> 'Pandas'`

Factory method to create a Pandas instance by reading data from a file.

**Arguments:**

- `filename`: Path to data file (supports .csv, .gz compression)
- `dstype`: Dataset type ("categorical", "continuous", or "mixed")  
- `N`: Optional sample size limit
- `**kwargs`: Additional arguments for pandas.read_csv()

**Features:**
- Automatic compression detection for .gz files
- Intelligent type inference and conversion
- Categorical variable encoding
- Memory-efficient loading for large files

**Example:**
```python
# Load categorical data
data = Pandas.read("dataset.csv", dstype="categorical")

# Load with custom separator and sample size
data = Pandas.read("data.tsv", dstype="mixed", N=10000, sep='\t')
```

## Data Management

### `set_N(N, seed=None, random_selection=False) -> None`

Sets the working sample size with optional randomization.

**Arguments:**

- `N`: Target sample size (must be ≤ original data size)
- `seed`: Randomization seed for reproducible sampling
- `random_selection`: If True, randomly selects rows; if False, uses first N rows

**Behavior:**

- Updates internal `_sample` DataFrame with the specified subset
- Preserves original data in `df` attribute
- Recomputes categorical value counts for the new sample

### `randomise_names(seed=None) -> None`

Randomizes node names for algorithm sensitivity testing.

**Arguments:**

- `seed`: Randomization seed, or None to revert to original names

**Implementation:**

- Generates randomized column names using format `X###NNNNNN`
- Updates DataFrame column names in-place
- Maintains mappings between original and external names
- Updates sample DataFrame to reflect name changes

## Statistical Operations

### `marginals(node, parents, values_reqd=False) -> Tuple`

Computes marginal distributions for a node given its parents.

**Arguments:**

- `node`: Target node name (external name)
- `parents`: Dictionary of parent values `{parent_name: value}`
- `values_reqd`: If True, returns actual values; if False, returns counts only

**Returns:**

- Tuple of (marginal_counts, unique_values) for categorical data
- For continuous data, returns appropriate statistical summaries

**Implementation:**

- Uses pandas crosstab for efficient categorical marginalization
- Handles continuous variables with binning strategies
- Optimized for sparse parent configurations

### `values(nodes: Tuple[str, ...]) -> np.ndarray`

Returns the actual data values for specified nodes.

**Arguments:**

- `nodes`: Tuple of node names (external names)

**Returns:**

- NumPy array with shape (N, len(nodes)) containing the data values

**Usage:**
```python
# Get values for specific variables
subset = data.values(("var1", "var2", "var3"))
print(subset.shape)  # (N, 3)
```

## Properties

### Node Information
- **`nodes`**: Original column names from DataFrame
- **`node_types`**: Mapping of node names to their data types
- **`node_values`**: Value counts for categorical variables only

### Sample Access
- **`N`**: Current working sample size
- **`sample`**: Current working sample as DataFrame

## Data Conversion

### `as_df() -> DataFrame`

Returns the current working sample as a pandas DataFrame.

**Returns:**

- DataFrame with external column names and current sample data

**Usage:**
```python
# Access current sample
current_df = data.as_df()

# Convert to NumPy for performance-critical operations
numpy_data = NumPy.from_df(current_df, dstype=data.dstype)
```

## File I/O

### `write(filename, compress=False, sf=10, zero=None, preserve=True) -> None`

Writes the current sample to a CSV file.

**Arguments:**

- `filename`: Output file path
- `compress`: Whether to gzip compress the output
- `sf`: Significant figures for floating-point data
- `zero`: Value to replace zeros with (for numerical stability)
- `preserve`: Whether to preserve original formatting

**Features:**

- Automatic compression if filename ends with .gz
- Configurable precision for floating-point output
- Handles categorical data appropriately
- Preserves data integrity during round-trip operations

## Type Handling

### Automatic Type Inference

The Pandas adapter automatically infers and converts data types:

**Categorical Data:**

- String columns are converted to pandas categorical type
- Integer columns with limited unique values become categorical
- Maintains category ordering where applicable

**Numeric Data:**

- Floating-point columns preserve precision
- Integer columns use appropriate NumPy integer types
- Mixed columns are handled according to dstype parameter

### Type Validation

```python
# Dataset type is automatically determined
if data.dstype == "categorical":
    print("All variables are categorical")
elif data.dstype == "continuous":
    print("All variables are numeric")  
else:  # "mixed"
    print("Mixed variable types detected")
```

## Memory Management

### Efficient Sampling

- Original DataFrame is preserved in `df` attribute
- Working sample stored separately in `_sample`
- View-based operations where possible to minimize copying

### Lazy Operations
- Type conversions performed only when necessary
- Value counts computed on-demand for categorical variables
- Sample updates triggered only when needed

## Performance Characteristics

### Best Use Cases

- Exploratory data analysis and prototyping
- Moderate-sized datasets (< 100K rows typically)
- Mixed data types requiring pandas functionality
- File I/O with various formats and options

### Performance Considerations

- Memory overhead due to pandas metadata
- String operations can be slow for large categorical data
- DataFrame operations generally slower than pure NumPy

## Integration Examples

### With NumPy Adapter
```python
# Load and explore with Pandas
pandas_data = Pandas.read("data.csv", dstype="mixed")
print(pandas_data.as_df().describe())

# Convert to NumPy for computational efficiency
numpy_data = NumPy.from_df(pandas_data.as_df(), 
                          dstype=pandas_data.dstype,
                          keep_df=True)
```

### Workflow Integration
```python
# Load data
data = Pandas.read("experiment.csv", dstype="categorical")

# Set up experiment parameters
data.set_N(5000, seed=42, random_selection=True)
data.randomise_names(seed=123)

# Run causal discovery algorithm
results = discovery_algorithm.run(data)

# Save results
data.write("experiment_sample.csv", compress=True)
```