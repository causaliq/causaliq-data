# Score - Scoring Functions for Causal Structure Learning

The `score` module provides comprehensive scoring functions for evaluating Bayesian networks and DAGs. It implements various entropy-based and Bayesian scoring methods for both categorical and continuous (Gaussian) data.

## Score Types

### Categorical Data Scores

#### Entropy-Based Scores
- **`loglik`**: Log-likelihood score
- **`bic`**: Bayesian Information Criterion
- **`aic`**: Akaike Information Criterion

#### Bayesian Scores
- **`bde`**: Bayesian Dirichlet Equivalent (with hyperparameters)
- **`k2`**: K2 score
- **`bdj`**: Bayesian Dirichlet with Jeffreys prior
- **`bds`**: Bayesian Dirichlet Sparse

### Gaussian Data Scores
- **`bic-g`**: Gaussian BIC
- **`bge`**: Bayesian Gaussian Equivalent
- **`loglik-g`**: Gaussian log-likelihood

## Constants

### Available Scores

```python
ENTROPY_SCORES = ["loglik", "bic", "aic"]
BAYESIAN_SCORES = ["bde", "k2", "bdj", "bds"] 
GAUSSIAN_SCORES = ["bic-g", "bge", "loglik-g"]
```

### Score Parameters

```python
SCORES = {
    "loglik": {"base"},
    "loglik-g": {"base"}, 
    "aic": {"base", "k"},
    "bic": {"base", "k"},
    "bic-g": {"base", "k"},
    "bge": {},
    "bde": {"iss", "prior"},
    "bds": {"iss", "prior"},
    "bdj": {},
    "k2": {},
}

SCORE_PARAMS = {
    "base": "e",
    "k": 1,
    "iss": 1, 
    "prior": "uniform",
    "unistate_ok": True,
}
```

## Core Functions

### Node Scoring

#### `node_score(node, parents, types, params, data) -> Dict[str, float]`

Computes specified score types for a single node given its parents.

**Arguments:**

- `node`: Node name to score
- `parents`: Dictionary mapping nodes to their parent lists
- `types`: Score type(s) to compute (string or list)
- `params`: Score parameters dictionary
- `data`: Data object implementing the Data interface

**Returns:**

- Dictionary mapping score types to computed values

#### `categorical_node_score(node, parents, types, params, data, counts_reqd=False)`

Returns decomposable scores for a categorical node with specified parents.

**Arguments:**

- `node`: Node name to score
- `parents`: Parents dictionary
- `types`: List of score types required
- `params`: Score parameters
- `data`: Data object
- `counts_reqd`: Whether to return marginal count information

**Returns:**

- Dictionary of scores, or tuple of (scores, counts_info) if counts_reqd=True

#### `gaussian_node_score(node, parents, types, params, data) -> Dict[str, float]`

Computes Gaussian scores for continuous nodes.

**Arguments:**

- `node`: Node name to score
- `parents`: Parents dictionary  
- `types`: List of Gaussian score types
- `params`: Score parameters
- `data`: Data object

**Returns:**

- Dictionary mapping score types to values

### DAG and Network Scoring

#### `dag_score(dag, data, types, params) -> DataFrame`

Returns per-node scores for a complete DAG given data.

**Arguments:**

- `dag`: DAG object to score
- `data`: Data object (not Oracle type)
- `types`: Score type(s) required (string or list)
- `params`: Score parameters dictionary

**Returns:**

- DataFrame with nodes as rows and score types as columns

**Raises:**

- `TypeError`: For invalid argument types
- `ValueError`: For invalid argument values or data/DAG mismatch

#### `bn_score(bn, N, types, params) -> DataFrame`

Returns per-node scores for a Bayesian Network using oracle scoring.

**Arguments:**

- `bn`: BN object to score
- `N`: Dataset size to assume
- `types`: Score type(s) required 
- `params`: Score parameters dictionary

**Returns:**

- DataFrame with nodes as rows and score types as columns

**Raises:**

- `TypeError`: For invalid argument types
- `ValueError`: For invalid score types or non-positive N

### Specialized Score Functions

#### `bayesian_score(N_ijk, q_i, type, params) -> float`

Computes Bayesian-based scores for marginal counts of a single node.

**Arguments:**

- `N_ijk`: 2D array of instance counts for node i, parental combo j, and node value k
- `q_i`: Maximum possible number of parental value combinations
- `type`: Bayesian score type (bde, k2, bdj, bds)
- `params`: Parameters dictionary including 'iss' (imaginary sample size)

**Returns:**

- Computed Bayesian score value

#### `entropy_scores(Nijk, types, params, N, free_params) -> Dict[str, float]`

Returns entropy-based scores for marginal counts.

**Arguments:**

- `Nijk`: 2D array of instance counts
- `types`: Entropy score types required
- `params`: Parameters including logarithm 'base'
- `N`: Number of cases (instances)  
- `free_params`: Number of free parameters

**Returns:**

- Dictionary of requested entropy scores

#### `bayesian_gaussian_score(node, parents, params, data) -> float`

Computes Bayesian Gaussian Equivalent (BGE) score for a node.

**Arguments:**

- `node`: Node to score
- `parents`: Parents dictionary
- `params`: Score parameters
- `data`: Data object

**Returns:**

- BGE score value

#### `entropy_gaussian_score(node, parents, params, data) -> Dict[str, float]`

Returns entropy-based scores for Gaussian nodes.

**Arguments:**

- `node`: Node to score
- `parents`: Parents dictionary
- `params`: Score parameters  
- `data`: Data object

**Returns:**

- Dictionary with 'bic-g' and 'loglik-g' scores

### Utility Functions

#### `check_score_params(params, scores=None) -> Dict[str, Any]`

Validates and completes score parameters with defaults.

**Arguments:**

- `params`: Parameters dictionary to validate
- `scores`: Optional list of score types to validate against

**Returns:**

- Validated and completed parameters dictionary

**Raises:**

- `TypeError`: For invalid parameter types
- `ValueError`: For invalid parameter values

#### `free_params(graph, data, debug=False) -> int`

Calculates the total number of free parameters in a graph structure.

**Arguments:**

- `graph`: Graph object (DAG or PDAG)
- `data`: Data object  
- `debug`: Whether to print debug information

**Returns:**

- Total number of free parameters

## Usage Examples

### Basic Node Scoring

```python
from causaliq_data.score import node_score

# Score a node with categorical data
scores = node_score(
    node="X1", 
    parents={"X1": ["X0", "X2"]},
    types=["bic", "bde"],
    params={"base": "e", "iss": 1},
    data=data_obj
)
```

### DAG Scoring

```python
from causaliq_data.score import dag_score

# Score entire DAG
df_scores = dag_score(
    dag=my_dag,
    data=my_data, 
    types=["bic", "loglik"],
    params={"base": "e"}
)
```

### Gaussian Scoring

```python
from causaliq_data.score import gaussian_node_score

# Score continuous node
gaussian_scores = gaussian_node_score(
    node="Y1",
    parents={"Y1": ["Y0"]}, 
    types=["bic-g", "bge"],
    params={"base": "e"},
    data=continuous_data
)
```

## Notes

- Score functions automatically determine appropriate score types based on data type
- Parameters are validated and defaults applied via `check_score_params()`  
- BGE implementation follows bnlearn defaults with some simplifications
- Entropy scores require sufficient sample sizes for reliable estimates
- Single-valued variables will raise errors unless `unistate_ok=True`