# Categorical scoring kernels for causaliq_data.score
#
# Extracted from causaliq_data.score so that the module stays within the
# module size limits. The names here are re-exported by causaliq_data.score,
# so their import path and behaviour are unchanged.

from typing import Any, Dict, List, Tuple, Union

from causaliq_core.utils import ln
from numpy import log as nplog
from numpy import ndarray
from numpy import sum as npsum
from numpy import zeros
from scipy.special import gammaln  # type: ignore

from causaliq_data.score_params import BAYESIAN_SCORES, ENTROPY_SCORES


def _check_bayesian_score_args(
    N_ijk: ndarray, q_i: int, type: str, params: Dict[str, Any]
) -> None:
    """Check the argument types and values for `bayesian_score`."""
    if (
        not isinstance(N_ijk, ndarray)
        or not isinstance(q_i, int)
        or isinstance(q_i, bool)
        or not isinstance(type, str)
        or not isinstance(params, dict)
    ):
        raise TypeError("Bad arg types for bayesian_score")

    if type not in BAYESIAN_SCORES or "iss" not in params:
        raise ValueError("Bad arg values for bayesian_score")


def _prior_counts(
    type: str, q_i: int, r_i: int, iss: float
) -> Tuple[float, float]:
    """Return the prior counts NP_ijk and NP_ij.

    Use variable names to match literature: NP_ijk is N-prime subscript ijk,
    the prior count for kth state and jth parental value combination.
    """
    if type in ["bde", "bds"]:
        return (iss / (q_i * r_i), iss / q_i)

    NP_ijk = 1.0 if type == "k2" else 0.5  # 1 or 0.5 for K2 or BDJ
    NP_ij = r_i if q_i == 1 or type == "k2" else 0.5 * r_i
    return (NP_ijk, NP_ij)


def bayesian_score(
    N_ijk: ndarray, q_i: int, type: str, params: Dict[str, Any]
) -> float:
    """Return Bayesian based scores for marginal counts (for a node).

    Variable names follow standard names used for the relevant formula in
    the literature.

    Args:
        N_ijk (ndarray): 2-D array of instance counts for node i,
            parental combo j and node value k.
        q_i (int): Maximum possible number of parental value combinations
            (even if they don't all necessarily occur in the data).
        type (str): Bayesian score type required e.g. bde, k2.
        params (dict): Parameters used in score computation: 'base' is
            logarithm base.

    Returns:
        dict: Of requested scores {score: value}.

    Raises:
        TypeError: If any argument has invalid type.
        ValueError: If any argument has invalid value.
    """
    _check_bayesian_score_args(N_ijk, q_i, type, params)

    iss = params["iss"]  # imaginary sample size
    score = 0.0  # will accrue the score value

    r_i = N_ijk.shape[0]

    if type == "bds":  # for BDS use actual number of parental combos in data
        q_i = N_ijk.shape[1]

    NP_ijk, NP_ij = _prior_counts(type, q_i, r_i, iss)

    # Element of score derived from marginal count for each parental combo

    N_ij = zeros((1, N_ijk.shape[1]))
    N_ij = N_ijk.sum(axis=0)
    score = npsum(gammaln(NP_ij) - gammaln(NP_ij + N_ij))

    # Element of score derived from individual j, k counts

    score += npsum(gammaln(NP_ijk + N_ijk) - gammaln(NP_ijk))

    return float(score)


def _check_entropy_scores_args(
    Nijk: ndarray,
    types: List[str],
    params: Dict[str, Any],
    N: int,
    free_params: int,
) -> None:
    """Check the argument types for `entropy_scores`."""
    if (
        not isinstance(Nijk, ndarray)
        or not isinstance(params, dict)
        or not isinstance(types, list)
        or not isinstance(N, int)
        or isinstance(N, bool)
        or not isinstance(free_params, int)
        or isinstance(free_params, bool)
        or any([not isinstance(t, str) for t in types])
    ):
        raise TypeError("Bad arg types for entropy_scores")


def _loglik_value(Nijk: ndarray, base: Union[int, str]) -> float:
    """Return the log likelihood of the marginal counts."""
    # compute column sums and replicate them for each row in Nijk so it
    # is same shape as Nijk ready for vectorised operations. Note other
    # approaches used like repeat and tile were slower than Nij[:] brpadcast
    # method.

    Nij = zeros(Nijk.shape)
    Nij[:] = Nijk.sum(axis=0)

    # use vectorised operation to compute log likelihood. "nonzero" is a mask
    # preventing operations on zero counts

    nonzero = Nijk > 0
    loglik = zeros(Nijk.shape)
    loglik[nonzero] = Nijk[nonzero] * nplog(Nijk[nonzero] / Nij[nonzero])
    loglik_val = float(loglik.sum().item())
    if base == 2 or base == 10:
        loglik_val = float(loglik_val / nplog(base))

    return loglik_val


def _entropy_score_values(
    loglik_val: float,
    types: List[str],
    params: Dict[str, Any],
    N: int,
    free_params: int,
) -> Dict[str, float]:
    """Return the requested entropy scores with complexity penalties."""
    base: Union[int, str] = (
        params["base"] if "base" in params else 10
    )  # default to base 10
    k = params["k"] if "k" in params else 1.0  # default to 1.0

    # Compute specific scores with particular complexity penalties

    scores: Dict[str, float] = {}
    if "loglik" in types:
        scores["loglik"] = loglik_val

    if "bic" in types:
        scores["bic"] = loglik_val - (k * 0.5 * free_params * ln(N, base))

    if "aic" in types:
        scores["aic"] = loglik_val - params["k"] * free_params

    return scores


def entropy_scores(
    Nijk: ndarray,
    types: Union[str, List[str]],
    params: Dict[str, Any],
    N: int,
    free_params: int,
) -> Dict[str, float]:
    """Return entropy based scores for marginal counts (for a node).

    Args:
        Nijk (ndarray): 2-D array of instance counts for node i,
            parental combo j and node value k.
        types (str/list): Entropy based score(s) required.
        params (dict): Parameters used in score computation: 'base' is
            logarithm base.
        N (int): Number of cases (instances).
        free_params (int): Number of free parameters.

    Returns:
        dict: Of requested scores {score: value}.

    Raises:
        TypeError: If any argument has invalid type.
        ValueError: If any argument has invalid value.
    """
    if isinstance(types, str):
        types = [types]

    _check_entropy_scores_args(Nijk, types, params, N, free_params)

    base: Union[int, str] = (
        params["base"] if "base" in params else 10
    )  # default to base 10

    if (
        any([t not in ENTROPY_SCORES for t in types])
        or N < 1
        or free_params < 0
        or base not in [2, 10, "e"]
    ):
        raise ValueError("Bad arg values for entropy_scores")

    loglik_val = _loglik_value(Nijk, base)

    return _entropy_score_values(loglik_val, types, params, N, free_params)
