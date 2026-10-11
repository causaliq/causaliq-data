#
#   Class for score (objective) functions
#
# The score kernels live in sibling modules which are re-exported below so
# that the public API (causaliq_data.score) and the functions substituted in
# tests are unchanged: categorical scores in score_categorical, Gaussian
# scores in score_gaussian and parameter handling in score_params. The
# dispatch layer (node_score) and the network scores stay in this module.

from typing import Any, Dict, Iterable, List, Tuple, Union

from pandas import DataFrame

from causaliq_data import Data, Oracle
from causaliq_data.score_categorical import bayesian_score, entropy_scores
from causaliq_data.score_gaussian import (
    bayesian_gaussian_score,
    entropy_gaussian_score,
)
from causaliq_data.score_params import (
    BAYESIAN_SCORES,
    ENTROPY_SCORES,
    GAUSSIAN_SCORES,
    SCORE_PARAMS,
    SCORES,
    check_score_params,
    free_params,
)

__all__ = [
    "ENTROPY_SCORES",
    "BAYESIAN_SCORES",
    "GAUSSIAN_SCORES",
    "SCORES",
    "SCORE_PARAMS",
    "node_score",
    "gaussian_node_score",
    "bayesian_gaussian_score",
    "entropy_gaussian_score",
    "categorical_node_score",
    "dag_score",
    "bn_score",
    "bayesian_score",
    "entropy_scores",
    "check_score_params",
    "free_params",
]


def node_score(
    node: str,
    parents: Dict[str, List[str]],
    types: List[str],
    params: Dict[str, Any],
    data: Data,
    counts_reqd: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """Return specified types of decomposable scores for a node.

    Args:
        node (str): Node scores are required for.
        parents (dict): Parents of non-orphan nodes {node: [parents]}.
        types (list): Scores required e.g. ['loglik', 'aic', 'bde'].
        params (dict): Parameters for scores e.g. logarithm base.
        data (Data): Data to learn graph from.
        counts_reqd (bool): Whether info about marginal counts returned.

    Returns:
        dict/tuple: Scores {score_type: score} if counts not required
            otherwise (scores, counts).
    """
    if data.dstype == "categorical":
        return categorical_node_score(
            node, parents, types, params, data, counts_reqd
        )
    else:
        return gaussian_node_score(
            node, parents, types, params, data, counts_reqd
        )


def gaussian_node_score(
    node: str,
    parents: Dict[str, List[str]],
    types: List[str],
    params: Dict[str, Any],
    data: Data,
    counts_reqd: bool,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """Return specified types of decomposable scores for a Gaussian node.

    Args:
        node (str): Node scores are required for.
        parents (dict): Parents of non-orphan nodes {node: [parents]}.
        types (list): Scores required e.g. ['bic-g', 'bge'].
        params (dict): Parameters for scores e.g. logarithm base.
        data (Data): Data to learn graph from.
        counts_reqd (bool): Not relevant for Gaussian node, but retained
            so returned data has structure consistent with categorical nodes.

    Returns:
        dict/tuple: Score {score_type: score} if counts not required
            otherwise (scores, counts).
    """
    score = {}
    if "bic-g" in types or "loglik-g" in types:
        _scores = entropy_gaussian_score(node, parents, params, data)
        if "bic-g" in types:
            score["bic-g"] = _scores["bic-g"]
        if "loglik-g" in types:
            score["loglik-g"] = _scores["loglik-g"]
    if "bge" in types:
        score["bge"] = bayesian_gaussian_score(node, parents, params, data)
    return (
        score
        if counts_reqd is False
        else (score, {"mean": 0, "max": 0, "min": 0, "lt5": 0, "fpa": 0})
    )


def categorical_node_score(
    node: str,
    parents: Dict[str, List[str]],
    types: List[str],
    params: Dict[str, Any],
    data: Data,
    counts_reqd: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """Return specified types of decomposable scores for categorical node.

    Args:
        node (str): Node scores are required for.
        parents (dict): Parents of non-orphan nodes {node: [parents]}.
        types (list): Scores required e.g. ['loglik', 'aic', 'bde'].
        params (dict): Parameters for scores e.g. logarithm base.
        data (Data): Data to learn graph from.
        counts_reqd (bool): Whether info about marginal counts returned.

    Returns:
        dict/tuple: Scores {score_type: score} if counts not required
            otherwise (scores, counts).
    """

    # get narginal counts from the data

    counts, q_i, _, _ = data.marginals(node, parents)

    # get statistics on contingency table counts

    if counts_reqd is True:
        counts_info: Dict[str, Any] = {
            "mean": counts.mean().item(),
            "max": counts.max().item(),
            "min": counts.min().item(),
            "lt5": (counts < 5).sum() / counts.size,
            "fpa": (counts.shape[0] - 1) * counts.shape[1],
        }
    else:
        counts_info = {"mean": 0, "max": 0, "min": 0, "lt5": 0, "fpa": 0}

    scores: Dict[str, float] = {}
    reqd_types = [t for t in types if t in ENTROPY_SCORES]
    if reqd_types:
        scores = entropy_scores(
            counts, reqd_types, params, data.N, q_i * (counts.shape[0] - 1)
        )

    for type in [t for t in types if t in BAYESIAN_SCORES]:
        scores[type] = bayesian_score(counts, q_i, type, params)

    return scores if counts_reqd is False else (scores, counts_info)


def _check_dag_score_args(
    dag: Any,
    data: Data,
    types: Iterable[str],
    params: Dict[str, Any],
) -> None:
    """Check the argument types and values for `dag_score`."""
    if (
        type(dag).__name__ != "DAG"
        or not isinstance(types, list)
        or any([not isinstance(t, str) for t in types])
        or not isinstance(params, dict)
        or not isinstance(data, Data)
        or isinstance(data, Oracle)
    ):
        raise TypeError("bad arg types for dag_score")

    if set(data.get_order()) != set(dag.nodes):
        raise ValueError("data/dag variable mismatch for dag_score")

    allowed = (
        ENTROPY_SCORES + BAYESIAN_SCORES
        if data.dstype == "categorical"
        else GAUSSIAN_SCORES
    )
    if not len(types) or any(t not in allowed for t in types):
        raise ValueError("bad score types for dag_score")


def _check_unistate(data: Data, params: Dict[str, Any]) -> None:
    """Check no variables are single state, unless it is allowed."""
    if (
        "unistate_ok" not in params or params["unistate_ok"] is not True
    ) and any([len(v.keys()) < 2 for n, v in data.node_values.items()]):
        raise ValueError("Some variables are single valued")


def _node_scores(
    nodes: Iterable[str],
    parents: Dict[str, List[str]],
    types: List[str],
    params: Dict[str, Any],
    data: Data,
) -> Dict[str, List[Any]]:
    """Return per-node scores with a column per score type.

    Loop over all nodes computing all required scores.
    """
    scores: Dict[str, List[Any]] = {
        score_type: [] for score_type in types + ["node"]
    }
    for node in nodes:
        scores["node"].append(node)
        node_scores = node_score(node, parents, types, params, data=data)
        if isinstance(node_scores, tuple):
            node_scores = node_scores[0]  # Extract just the scores dict
        for score_type, value in node_scores.items():
            scores[score_type].append(value)

    return scores


def dag_score(
    dag: Any, data: Data, types: Union[str, List[str]], params: Dict[str, Any]
) -> DataFrame:
    """Return per-node DAG+data network scores of specified types.

    Args:
        dag (DAG): The DAG to score.
        data (Data): Data to be scored.
        types (list/str): Type(s) of score required e.g. 'bic', 'bde'.
        params (dict): Parameters for scores e.g. logarithm base.

    Returns:
        DataFrame: Scores (column=type) per node (rows).

    Raises:
        TypeError: If args have bad types.
        ValueError: If args have bad values.
    """
    if isinstance(types, str):
        types = [types]

    _check_dag_score_args(dag, data, types, params)

    params = params.copy()
    params = check_score_params(params, types)  # check params, set defaults

    _check_unistate(data, params)

    scores = _node_scores(dag.nodes, dag.parents, types, params, data)

    return DataFrame(scores).set_index("node")  # type: ignore[no-any-return]


def _check_bn_score_args(
    bn: Any,
    N: int,
    types: Iterable[str],
    params: Dict[str, Any],
) -> None:
    """Check the argument types and values for `bn_score`."""
    if (
        type(bn).__name__ != "BN"
        or not isinstance(types, list)
        or any([not isinstance(t, str) for t in types])
        or not isinstance(params, dict)
        or not isinstance(N, int)
        or isinstance(N, bool)
    ):
        raise TypeError("bad arg types for bn_score")

    if not len(types) or any(
        t not in ENTROPY_SCORES + BAYESIAN_SCORES for t in types
    ):
        raise ValueError("bad score types for dag_score")

    if N < 1:
        raise ValueError("non-positive N for bn_score")


def bn_score(
    bn: Any, N: int, types: Union[str, List[str]], params: Dict[str, Any]
) -> DataFrame:
    """Return per-node scores of specified types for Bayesian Network.

    Assumes specified dataset size. Computes scores based on CPTs -
    hence is an "oracle" score.

    Args:
        bn (BN): The BN to score.
        N (int): Dataset size to use.
        types (list/str): Type(s) of score required e.g. 'bic', 'bde'.
        params (dict): Parameters for scores e.g. logarithm base.

    Returns:
        DataFrame: Scores (column=type) per node (rows).

    Raises:
        TypeError: If args have bad types.
        ValueError: If args have bad values.
    """
    if isinstance(types, str):
        types = [types]

    _check_bn_score_args(bn, N, types, params)

    params = params.copy()
    params = check_score_params(params, types)  # check params and set defaults

    # Loop over all nodes computing all required scores

    data = Oracle(bn=bn)
    data.set_N(N)

    scores = _node_scores(
        bn.dag.ordered_nodes(), bn.dag.parents, types, params, data
    )

    return DataFrame(scores).set_index("node")  # type: ignore[no-any-return]
