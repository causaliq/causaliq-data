# Gaussian scoring kernels for causaliq_data.score
#
# Extracted from causaliq_data.score so that the module stays within the
# module size limits. The names here are re-exported by causaliq_data.score,
# so their import path and behaviour are unchanged.

from typing import Any, Dict, List, Optional, Tuple

import scipy.stats as stats  # type: ignore
from causaliq_core.utils import ln
from causaliq_core.utils.timing import Timing
from numpy import cov, fill_diagonal, float64
from numpy import log as nplog
from numpy import mean as npmean
from numpy import pi
from numpy import sqrt as npsqrt
from numpy import sum as npsum
from numpy.linalg import det
from scipy.special import gammaln  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from causaliq_data import Data


def _bge_dimensions(data: Data) -> Tuple[float64, float64, float64]:
    """Return the BGE sample size, node count and prior weight."""
    N = float64(data.N)
    n = float64(len(data.nodes))

    return N, n, float64(n + 2)


def _gamma_ratio_term(
    N: float64, n: float64, alpha_w: float64, p: int
) -> float64:
    """Return the term 2 ratio of multivariate Gamma functions."""
    return float64(
        gammaln(0.5 * (N + alpha_w - n + p + 1))
        - gammaln(0.5 * (alpha_w - n + p + 1))
        - 0.5 * N * nplog(pi)
    )


def _orphan_bge_term(
    data: Data,
    node: str,
    N: float64,
    n: float64,
    alpha_w: float64,
    prior: float64,
) -> float64:
    """Return the BGE term 3 complexity penalty for an orphan node."""
    values = data.values((node,))[:, 0]
    mean = npmean(values)
    sse: float = npsum((values - mean) ** 2)  # Sum of squared errors
    term = 0.5 * (
        (alpha_w - n + 1) * nplog(prior)
        - (N + alpha_w - n + 1) * nplog(prior + sse)
    )

    return float64(term)


def _prior_det_term(
    n: float64, alpha_w: float64, p: int, prior: float64
) -> float64:
    """Return the BGE term 3 ratio of determinants of the priors."""
    return float64(
        0.5
        * ((alpha_w - n + p + 1) * (p + 1) - (alpha_w - n + p) * (p))
        * nplog(prior)
    )


def _posterior_dets(
    data: Data,
    node: str,
    parents: Dict[str, List[str]],
    N: float64,
    prior: float64,
) -> Tuple[float64, float64]:
    """Return the BGE posterior determinants with and without child.

    The covariance matrix of the parents and child is scaled by (N - 1),
    the prior is added to all diagonal elements, and then the determinant
    is computed; the second determinant omits the child variable.
    """
    values = data.values(tuple([node] + parents[node])).astype(float64)

    Ryy_cov = cov(values, rowvar=False) * (N - 1)
    fill_diagonal(Ryy_cov, Ryy_cov.diagonal() + prior)
    Ryy_det = float64(det(Ryy_cov))
    Tyy_det = float64(det(Ryy_cov[1:, 1:]))

    return Ryy_det, Tyy_det


def _apply_log_base(bge: float64, params: Dict[str, Any]) -> float64:
    """Return the BGE converted to the requested logarithm base."""
    if params is not None and "base" in params and params["base"] != "e":
        return float64(bge / nplog(params["base"]))

    return float64(bge)


def bayesian_gaussian_score(
    node: str,
    parents: Dict[str, List[str]],
    params: Dict[str, Any],
    data: Data,
) -> float:
    """Compute Bayesian Gaussian Equivalent (BGE) for a node.

    Computes BGE for a node with specified parents. Based on the
    cwpost/wpost functions in bnlearn which are in turn derived from
    Kuipers et al, 2014, Annals of Statistics which corrects the orginal
    formulation by Geiger and Heckerman (2002). Variable names follow
    those in equation (2) of Kuipers et al. This implementation is
    simplified so that some hyperparameters have the bnlearn default
    values:
      - alpha_mu = 1.0 - weight assigned to prior means
      - nu = observed means - prior means for each variable
      - alpha_w = n + 2 - weight assigned to prior variances

    Args:
        node (str): Node scores are required for.
        parents (dict): Parents of non-orphan nodes {node: [parents]}.
        params (dict): Parameters for scores e.g. logarithm base.
        data (Data): Data to learn graph from.

    Returns:
        float: BGE score.
    """
    N, n, alpha_w = _bge_dimensions(data)

    # Term 1 is [alpha_mu / (alpha_mu + N)] ** (p / 2), where p is the number
    # of parents + 1. Here we follow bnlearn which seems to INCORRECTLY set p
    # to be 1 (in the orginal equation p is actually the letter "l", but we
    # use p here for readability). alpha_mu takes its default value of 1.0.

    bge = -0.5 * nplog(N + 1)

    # Term 2 is the ratio of multivariate Gamma functions. Here again, we
    # follow bnlearn where p is set to 1 (or 0 for orphan nodes) instead of
    # actual number of parents, and additionally univariate Gamma functions
    # are used. This seems a MISTAKE.

    p = len(parents[node]) if node in parents else 0
    bge += _gamma_ratio_term(N, n, alpha_w, p)

    # Term 3 is a complexity penalty which differs between orphan and
    # non-orphan nodes.

    prior = 0.5 * (alpha_w - n - 1)
    if node not in parents:
        bge += _orphan_bge_term(data, node, N, n, alpha_w, prior)
    else:
        bge += _prior_det_term(n, alpha_w, p, prior)
        Ryy_det, Tyy_det = _posterior_dets(data, node, parents, N, prior)

        # Add the determinant ratio into BGE score

        bge += 0.5 * (N + alpha_w - n + p) * nplog(Tyy_det)
        bge -= 0.5 * (N + alpha_w - n + p + 1) * nplog(Ryy_det)

    return float(_apply_log_base(bge, params).item())


def _orphan_gaussian_loglik(
    data: Data, node: str, scale: int, start: Optional[float]
) -> Tuple[float, int, Optional[float]]:
    """Return the orphan node log likelihood and free parameter count.

    Computes the mean and corrected SD of the values and sums the
    log-likelihood of each value assuming a Gaussian with those values.
    """
    values = data.values((node,))[:, 0]
    mean = npmean(values)
    sd = npsqrt(npsum((values - mean) ** 2) / (data.N - 1))
    start = Timing.record("fit", scale, start)

    loglik: float = npsum(stats.norm.logpdf(values, loc=mean, scale=sd))
    start = Timing.record("loglik", scale, start)

    return loglik, 2, start


def _child_gaussian_loglik(
    data: Data,
    node: str,
    parents: Dict[str, List[str]],
    scale: int,
    start: Optional[float],
) -> Tuple[float, int, Optional[float]]:
    """Return the child node log likelihood and free parameter count.

    Fits a linear regression predicting child values from parents. The
    likelihood of the predicted value is the likelihood of the residual
    using a Gaussian with zero mean and the computed SD of residuals.
    """
    values = data.values(tuple([node] + parents[node]))
    model = LinearRegression().fit(values[:, 1:], values[:, 0])
    start = Timing.record("fit", scale, start)

    residuals = values[:, 0] - model.predict(values[:, 1:])
    sd = npsqrt(npsum(residuals**2) / (data.N - 1))
    start = Timing.record("residuals", scale, start)

    loglik: float = npsum(stats.norm.logpdf(residuals, loc=0.0, scale=sd))
    start = Timing.record("loglik", scale, start)

    return loglik, 2 + len(parents[node]), start


def entropy_gaussian_score(
    node: str,
    parents: Dict[str, List[str]],
    params: Dict[str, Any],
    data: Data,
) -> Dict[str, float]:
    """Return entropy-based scores for a Gaussian node.

    Args:
        node (str): Node scores are required for.
        parents (dict): Parents of non-orphan nodes {node: [parents]}.
        params (dict): Parameters for scores e.g. logarithm base.
        data (Data): Data to learn graph from.

    Returns:
        dict: {'bic-g': float, 'loglik-g': float}.
    """
    start = Timing.now()
    scale = len(parents[node]) + 1 if node in parents else 1

    if node not in parents:
        loglik, free_params_count, start = _orphan_gaussian_loglik(
            data, node, scale, start
        )
    else:
        loglik, free_params_count, start = _child_gaussian_loglik(
            data, node, parents, scale, start
        )

    # Subtract penalty value

    penalty = free_params_count * 0.5 * ln(data.N)
    bic_g = loglik - penalty

    return {"bic-g": bic_g, "loglik-g": loglik}
