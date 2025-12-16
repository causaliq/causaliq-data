# Oracle - Synthetic Data Generator

The `Oracle` class provides a specialized data adapter that generates synthetic data from known Bayesian Networks. This adapter is primarily used for algorithm validation, benchmarking, and controlled experiments where the true underlying causal structure is known.

## Class Definition

```python
class Oracle(Data):
    """Oracle data adapter for synthetic data generation from Bayesian Networks.
    
    Args:
        bn: A BN (Bayesian Network) object from causaliq-core.
        
    Attributes:
        bn: The underlying Bayesian Network object.
    """
```

## Constructor

### `__init__(bn) -> None`

Creates an Oracle data adapter from a Bayesian Network.

**Arguments:**
- `bn`: BN object from causaliq-core containing DAG structure and CPTs

**Validation:**
- Input must be a valid BN object
- BN must contain both DAG structure and conditional probability tables
- All nodes must have associated conditional distributions

**Initialization:**
- Extracts node names from BN DAG structure
- Determines variable types from conditional distributions (CPT vs continuous)
- Sets initial sample size to 1 (can be changed with `set_N()`)

**Example:**
```python
from causaliq_core.bn.io import read_bn
from causaliq_data import Oracle

# Load BN from file
bn = read_bn("cancer.dsc")

# Create Oracle adapter
oracle = Oracle(bn)
print(f"Nodes: {oracle.nodes}")
print(f"Types: {oracle.node_types}")
```

## Synthetic Data Generation

### `set_N(N, seed=None, random_selection=False) -> None`

Sets the effective sample size for synthetic data operations.

**Arguments:**

- `N`: Target sample size for synthetic data generation
- `seed`: Must be None (not supported for Oracle)
- `random_selection`: Must be False (not applicable)

**Behavior:**

- Updates internal sample size counter
- Does not actually generate data (Oracle provides analytical answers)
- Used by algorithms to determine confidence/precision of estimates

**Validation:**

- `N` must be positive integer
- `seed` parameter must be None (raises TypeError if provided)
- `random_selection` must be False

**Usage:**
```python
oracle.set_N(10000)  # Set effective sample size
print(f"Sample size: {oracle.N}")
```

## Statistical Operations

### `marginals(node, parents, values_reqd=False) -> Tuple`

Provides exact marginal distributions from the Bayesian Network.

**Arguments:**

- `node`: Target node name (internal name)
- `parents`: Dictionary specifying parent values `{parent: value}`
- `values_reqd`: Whether to return value labels (always False for Oracle)

**Returns:**

- Exact conditional probability distribution for the node given parents
- For categorical nodes: probability vector over possible values
- For continuous nodes: parameters of the conditional distribution

**Features:**

- **Exact Results**: Returns true probabilities, not empirical estimates
- **No Sampling Error**: Results are analytical, not subject to sampling variation
- **Efficient Computation**: Leverages BN's internal probability representations

**Example:**
```python
# Get P(Cancer | Smoker=True, Pollution=High)
marginal = oracle.marginals("Cancer", 
                           {"Smoker": "True", "Pollution": "High"})
print(f"P(Cancer=True|evidence): {marginal[0][1]}")
```

### `values(nodes) -> np.ndarray`

**Not Implemented**: Oracle does not store actual data values.

**Raises:**

- `TypeError`: Always raised with message "Oracle.values() not implemented"

**Rationale:**

- Oracle provides analytical results, not sampled data
- Use concrete adapters (Pandas/NumPy) for data value access
- Consistent with Oracle's role as synthetic probability source

## Specialized Oracle Features

### True Parameter Access

Oracle provides direct access to the true parameters of the Bayesian Network:

**Conditional Probability Tables:**
```python
# Access true CPT for a categorical node
cpt = oracle.bn.cnds["Disease"]
print("True conditional probabilities:")
for parent_config in cpt.parents_configs():
    for value in cpt.values:
        prob = cpt.get_prob(parent_config, value)
        print(f"P({value}|{parent_config}) = {prob}")
```

**Network Structure:**
```python
# Access true DAG structure
dag = oracle.bn.dag
print(f"True edges: {dag.edges}")
print(f"True parents of X: {dag.parents('X')}")
```

### Algorithm Validation

Oracle is ideal for validating causal discovery algorithms:

**Score Validation:**
```python
# Compare algorithm scores with true model
true_score = oracle.score(learned_dag)
oracle_score = oracle.score(oracle.bn.dag)  # True structure score
print(f"Score difference: {abs(true_score - oracle_score)}")
```

**Conditional Independence Testing:**
```python
# Test algorithm's CI conclusions against true model
for x, y, z in ci_tests:
    true_independent = oracle.bn.d_separated(x, y, z)
    algorithm_independent = algorithm.ci_test(oracle, x, y, z)
    accuracy = (true_independent == algorithm_independent)
```

## Limitations and Constraints

### Unsupported Operations

**Data Value Access:**
- `values()` method raises TypeError
- No actual data samples available
- Use for probability queries only

**Randomization Restrictions:**

- `randomise_names()` raises NotImplementedError  
- Name randomization not meaningful for Oracle
- Node names tied to BN structure

**Sampling Limitations:**

- No row-level sampling or shuffling
- `set_N()` only affects effective sample size for algorithms
- No actual data generation performed

### Data Type Constraints

**Variable Types:**

- Categorical variables: Must have finite discrete values
- Continuous variables: Limited to supported distribution types
- Mixed networks: Handled according to individual node types

## Integration with CausalIQ Ecosystem

### Algorithm Testing Framework
```python
def test_algorithm_accuracy(algorithm, test_bns):
    results = []
    for bn_file in test_bns:
        # Load true BN
        bn = read_bn(bn_file)
        oracle = Oracle(bn)
        
        # Run algorithm
        oracle.set_N(10000)  # Large effective sample size
        learned_structure = algorithm.run(oracle)
        
        # Compare with true structure  
        accuracy = compare_structures(bn.dag, learned_structure)
        results.append(accuracy)
    
    return results
```

### Benchmark Experiments
```python
def benchmark_scoring_functions(oracle, scoring_functions):
    true_score = {}
    for score_fn in scoring_functions:
        # Get score for true structure
        true_score[score_fn.name] = score_fn.calculate(oracle, oracle.bn.dag)
        
        # Test alternative structures
        for alt_structure in generate_alternatives(oracle.bn.dag):
            alt_score = score_fn.calculate(oracle, alt_structure)
            print(f"{score_fn.name}: True={true_score[score_fn.name]:.3f}, "
                  f"Alt={alt_score:.3f}")
```

### Stability Analysis
```python
def analyze_algorithm_stability(algorithm, oracle, trials=100):
    # Oracle provides consistent "data" across trials
    results = []
    for trial in range(trials):
        oracle.randomise_order(trial)  # Change processing order
        result = algorithm.run(oracle)
        results.append(result)
    
    # Analyze consistency of results
    return assess_stability(results)
```

## Performance Characteristics

### Computational Efficiency
- **Analytical Operations**: No sampling or counting required
- **Exact Computations**: Probability queries return exact values
- **Memory Efficient**: No large data arrays stored
- **Fast Initialization**: Only stores BN structure and parameters

### Scalability
- **Network Size**: Performance depends on BN complexity, not sample size
- **Query Complexity**: Marginal queries scale with network connectivity
- **Memory Usage**: Minimal, proportional to BN size only

## Best Practices

### When to Use Oracle
- **Algorithm Validation**: Testing against known ground truth
- **Benchmarking**: Comparing algorithm performance across known structures
- **Method Development**: Developing new algorithms with reliable test cases
- **Educational Use**: Demonstrating causal discovery concepts

### Usage Patterns
```python
# Validation workflow
oracle = Oracle(known_bn)
oracle.set_N(sample_size)

# Test your algorithm
learned_result = your_algorithm.discover(oracle)

# Compare with truth
accuracy_metrics = evaluate_against_truth(learned_result, oracle.bn)
```

### Integration Tips
- Use Oracle early in algorithm development for debugging
- Combine with Pandas/NumPy adapters for comprehensive testing
- Leverage exact probabilities for theoretical analysis
- Document true structure properties for result interpretation