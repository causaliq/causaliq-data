# Score parameter constants and checks for causaliq_data.score
#
# Extracted from causaliq_data.score so that the module stays within the
# module size limits. The names here are re-exported by causaliq_data.score,
# so their import path and behaviour are unchanged. The score constants live
# here because the score kernels need them and importing them from
# causaliq_data.score would create a circular import.

from typing import Any, Dict, List, Optional

from pandas import DataFrame

ENTROPY_SCORES = ["loglik", "bic", "aic"]  # categorical entropy scores
BAYESIAN_SCORES = ["bde", "k2", "bdj", "bds"]  # categorical Bayesian
GAUSSIAN_SCORES = ["bic-g", "bge", "loglik-g"]  # Gaussian scores

SCORES = {
    "loglik": {"base"},  # Supported scores and their allowed parameters
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
    "base": "e",  # score parameters and default values
    "k": 1,
    "iss": 1,
    "prior": "uniform",
    "unistate_ok": True,
}


def _check_score_params_args(
    params: Dict[str, Any], scores: Optional[List[str]]
) -> None:
    """Check the argument types and values for `check_score_params`."""
    if not isinstance(params, dict) or (
        scores is not None and not isinstance(scores, list)
    ):
        raise TypeError("Bad arg type for _check_score_params")

    if any([key not in SCORE_PARAMS for key in params]):
        raise ValueError("Unknown score parameter")


def _score_param_defaults(
    params: Dict[str, Any], scores: Optional[List[str]]
) -> Dict[str, Any]:
    """Return values for the score parameters that were not specified."""
    reqd = (
        list(SCORE_PARAMS)
        if scores is None
        else [p for s in scores for p in list(SCORES[s])]
    )
    return {p: SCORE_PARAMS[p] for p in reqd if p not in params}


def _check_score_param_types(params: Dict[str, Any]) -> None:
    """Check the types of the score parameters specified by the user."""
    if "base" in params and (
        (
            not isinstance(params["base"], int)
            and not isinstance(params["base"], str)
        )
        or isinstance(params["base"], bool)
    ):
        raise TypeError("Wrong type for base score parameter")
    if "prior" in params and not isinstance(params["prior"], str):
        raise TypeError("Wrong type for prior base parameter")
    if (
        "iss" in params
        and not isinstance(params["iss"], int)
        and not isinstance(params["iss"], float)
    ):
        raise TypeError("Wrong type for prior base parameter")
    if (
        "k" in params
        and not isinstance(params["k"], int)
        and not isinstance(params["k"], float)
    ):
        raise TypeError("Wrong type for penalty multiplier")


def _check_score_param_values(params: Dict[str, Any]) -> None:
    """Check the values of the score parameters specified by the user."""
    if "base" in params and params["base"] not in [2, 10, "e"]:
        raise ValueError("Bad value for base score param")
    if "prior" in params and params["prior"] not in ["uniform", "marginal"]:
        raise ValueError("Bad value for priot score param")
    if "iss" in params and (params["iss"] < 1e-6 or params["iss"] > 1e6):
        raise ValueError("Bad value for prior score param")
    if "k" in params and (params["k"] < 1e-6 or params["k"] > 1e6):
        raise ValueError("Bad value for penalty multiplier")


def check_score_params(
    params: Dict[str, Any], scores: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Check score parameters specified by user and supply default values.

    Parameters supported are:
      * base: logarithm base used, 2, 10 or 'e'
      * iss: imaginary sample size, positive real
      * prior: structural prior, string 'uniform' or 'marginal'
      * k: complexity penalty multiplier

    Args:
        params (dict): Parameters for scoring metrics.
        scores (list/None): Score/list of scores which params used for.

    Returns:
        dict: Of parameters {param: value}.

    Raises:
        TypeError: If invalid types used.
        ValueError: If any invalid values, combinations specified.
    """
    _check_score_params_args(params, scores)

    params.update(_score_param_defaults(params, scores))

    _check_score_param_types(params)

    _check_score_param_values(params)

    return params


def _value_counts(data: DataFrame) -> Dict[Any, Dict[Any, int]]:
    """Return the count of each value for each column in the data."""
    value_counts: Dict[Any, Dict[Any, int]] = {}
    for col in data.columns:
        counts = data[col].value_counts()
        value_counts[col] = {
            value: count for value, count in zip(counts.index, counts.values)
        }

    return value_counts


def _count_free_params(
    node: str, value_counts: Dict[Any, Dict[Any, int]], graph: Any
) -> int:
    """Return the number of free parameters for a single node."""
    cardinality = len(value_counts[node])
    n_free_params = cardinality - 1
    if node in graph.parents:
        for parent in graph.parents[node]:
            n_free_params *= len(value_counts[parent])

    return n_free_params


def _print_node_params(
    node: str,
    cardinality: int,
    num_parents: int,
    n_free_params: int,
) -> None:
    """Print the number of states, parents and free params for a node."""
    print(
        "{} has {} state(s), {} parent(s)"
        " and {} free param(s)".format(
            node, cardinality, num_parents, n_free_params
        )
    )


def free_params(graph: Any, data: DataFrame, debug: bool = False) -> int:
    """Compute number of free parameters for graph and data.

    Args:
        graph (DAG): The dag to return value for.
        data (DataFrame): Data for the graph.
        debug (bool): Whether details of node parameters printed.

    Returns:
        int: Number of free params for graph with data.
    """
    if debug:
        print("")

    value_counts = _value_counts(data)

    total_free_params = 0
    for node in graph.nodes:
        cardinality = len(value_counts[node])
        n_free_params = _count_free_params(node, value_counts, graph)
        total_free_params += n_free_params
        if debug:
            num_parents = (
                len(graph.parents[node]) if node in graph.parents else 0
            )
            _print_node_params(node, cardinality, num_parents, n_free_params)

    if debug:
        print("Graph has {} free params".format(total_free_params))

    return total_free_params
