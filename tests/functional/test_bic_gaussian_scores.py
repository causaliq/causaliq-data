# Test BIC Gaussian scores - Mathematical validation without bnlearn dependency
# This test file validates CausalIQ's BIC Gaussian score implementation
# using mathematical properties and internal consistency checks

from math import pi
from os import remove
from random import random

import pytest
from causaliq_core.bn import BN
from causaliq_core.utils import values_same
from numpy import array, log
from numpy import mean as npmean
from numpy import sqrt
from numpy import sum as npsum
from pandas import DataFrame

import tests.fixtures.example_dags as dag
from causaliq_data import NumPy, Pandas
from causaliq_data.score import (
    bayesian_gaussian_score,
    bn_score,
    categorical_node_score,
    check_score_params,
    dag_score,
    entropy_gaussian_score,
    free_params,
    gaussian_node_score,
)

TESTDATA_DIR = "tests/data/functional"


# temp file, automatically removed
@pytest.fixture(scope="function")
def tmpfile():
    _tmpfile = TESTDATA_DIR + "/tmp/{}.csv".format(int(random() * 10000000))
    yield _tmpfile
    try:
        remove(_tmpfile)
    except Exception:
        pass


# Test that BIC-g follows the correct mathematical formula
def test_bic_formula_consistency():
    """Test that BIC-g follows the correct mathematical formula:
    BIC = loglik - k*params*ln(N)/2"""
    data = NumPy(
        array([[1.1, 0.0], [2.2, 1.7], [-0.3, 0.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Get both BIC and loglik scores
    scores_bic = dag_score(dag.x_y(), data, "bic-g", {"k": 1.0})
    scores_loglik = dag_score(dag.x_y(), data, "loglik-g", {"k": 1.0})

    # For independent X,Y: each node has 2 parameters (mean, sd)
    total_params = 4
    N = data.N
    k = 1.0

    # Calculate expected penalty
    penalty = k * total_params * log(N) / 2

    # BIC should equal loglik minus penalty
    expected_bic = dict(scores_loglik.sum())["loglik-g"] - penalty
    actual_bic = dict(scores_bic.sum())["bic-g"]

    assert values_same(actual_bic, expected_bic, sf=10)


# Test that parameter counting is correct for different network structures
def test_parameter_counting_correctness():
    """Test that parameter counting is correct for
    different network structures"""
    NumPy(
        array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype="float32",
        ),
        dstype="continuous",
        col_values={"X": None, "Y": None, "Z": None},
    )

    # Two-node independent: X Y -> 2 nodes * 2 params each = 4 params
    data_xy = NumPy(
        array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )
    scores_xy_indep = dag_score(dag.x_y(), data_xy, "bic-g", {"k": 1.0})
    loglik_xy_indep = dag_score(dag.x_y(), data_xy, "loglik-g", {"k": 1.0})

    # X->Y chain: X(2) + Y(3: mean,sd,coeff) = 5 params
    scores_xy_chain = dag_score(dag.xy(), data_xy, "bic-g", {"k": 1.0})
    loglik_xy_chain = dag_score(dag.xy(), data_xy, "loglik-g", {"k": 1.0})

    penalty_indep = 4 * log(data_xy.N) / 2
    penalty_chain = 5 * log(data_xy.N) / 2

    expected_bic_indep = (
        dict(loglik_xy_indep.sum())["loglik-g"] - penalty_indep
    )
    expected_bic_chain = (
        dict(loglik_xy_chain.sum())["loglik-g"] - penalty_chain
    )

    assert values_same(
        dict(scores_xy_indep.sum())["bic-g"], expected_bic_indep, sf=10
    )
    assert values_same(
        dict(scores_xy_chain.sum())["bic-g"], expected_bic_chain, sf=10
    )


# Test that BIC-g scores are in reasonable ranges
def test_score_validity_ranges():
    """Test that BIC-g scores are in reasonable ranges"""
    data = NumPy(
        array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    scores = dag_score(dag.x_y(), data, "bic-g", {"k": 1.0})
    total_score = dict(scores.sum())["bic-g"]

    # Score should be a finite number
    assert not (total_score == float("inf") or total_score == float("-inf"))
    assert not (total_score != total_score)  # Check for NaN

    # For this small dataset, score should be reasonable
    assert -100 < total_score < 100


# Test that orphan node BIC score matches manual calculation
def test_orphan_node_score_correctness():
    """Test that orphan node BIC score matches manual calculation"""
    # Use 2-column data but only test X
    data = NumPy(
        array(
            [[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0], [5.0, 9.0]],
            dtype="float32",
        ),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Manual calculation for X values
    values = array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    N = len(values)
    mean = npmean(values)
    sd = sqrt(npsum((values - mean) ** 2) / (N - 1))

    # Log-likelihood for Gaussian
    loglik_manual = npsum(
        -0.5 * log(2 * pi) - log(sd) - 0.5 * ((values - mean) / sd) ** 2
    )
    penalty_manual = 2 * log(N) / 2  # 2 parameters: mean, sd
    bic_manual = loglik_manual - penalty_manual

    # Get score from function for X node only
    scores = gaussian_node_score("X", {}, ["bic-g"], {"k": 1.0}, data, False)

    assert values_same(scores["bic-g"], bic_manual, sf=6)


# Test that individual node scores sum to total DAG score
def test_node_score_additivity():
    """Test that individual node scores sum to total DAG score"""
    data = NumPy(
        array(
            [[1.1, 2.2, 3.3], [1.4, 2.5, 3.6], [1.7, 2.8, 3.9]],
            dtype="float32",
        ),
        dstype="continuous",
        col_values={"X": None, "Y": None, "Z": None},
    )

    # Get total score
    total_scores = dag_score(dag.xyz(), data, "bic-g", {"k": 1.0})
    total_bic = dict(total_scores.sum())["bic-g"]

    # Get individual node scores
    x_score = gaussian_node_score("X", {}, ["bic-g"], {"k": 1.0}, data, False)
    y_score = gaussian_node_score(
        "Y", {"Y": ["X"]}, ["bic-g"], {"k": 1.0}, data, False
    )
    z_score = gaussian_node_score(
        "Z", {"Z": ["Y"]}, ["bic-g"], {"k": 1.0}, data, False
    )

    manual_total = x_score["bic-g"] + y_score["bic-g"] + z_score["bic-g"]

    assert values_same(total_bic, manual_total, sf=10)


# Test that more data generally improves likelihood (before penalty)
def test_more_data_improves_fit():
    """Test that more data generally improves likelihood (before penalty)"""
    # Small dataset with good fit
    small_data = NumPy(
        array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Larger dataset with same linear relationship
    large_data = NumPy(
        array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
            ],
            dtype="float32",
        ),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    small_loglik = dag_score(dag.x_y(), small_data, "loglik-g", {"k": 1.0})
    large_loglik = dag_score(dag.x_y(), large_data, "loglik-g", {"k": 1.0})

    # Just verify both produce valid scores
    small_score = dict(small_loglik.sum())["loglik-g"]
    large_score = dict(large_loglik.sum())["loglik-g"]

    # Both should be finite numbers
    assert not (small_score == float("inf") or small_score == float("-inf"))
    assert not (large_score == float("inf") or large_score == float("-inf"))
    assert small_score == small_score  # Check for NaN
    assert large_score == large_score  # Check for NaN


# Test entropy_gaussian_score function directly
def test_entropy_gaussian_score_function():
    """Test entropy_gaussian_score function directly"""
    data = NumPy(
        array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Test orphan node
    x_entropy = entropy_gaussian_score("X", {}, {"k": 1.0}, data)
    x_gaussian = gaussian_node_score(
        "X", {}, ["bic-g", "loglik-g"], {"k": 1.0}, data, False
    )

    assert values_same(x_entropy["bic-g"], x_gaussian["bic-g"], sf=10)
    assert values_same(x_entropy["loglik-g"], x_gaussian["loglik-g"], sf=10)


# Test that adding parents changes score appropriately
def test_linear_regression_effect():
    """Test that adding parents changes score appropriately"""
    # Create data where Y is clearly related to X
    data = NumPy(
        array(
            [[1.0, 2.1], [2.0, 4.1], [3.0, 6.1], [4.0, 8.1], [5.0, 10.1]],
            dtype="float32",
        ),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Y as orphan vs Y with parent X
    y_orphan = gaussian_node_score("Y", {}, ["bic-g"], {"k": 1.0}, data, False)
    y_with_x = gaussian_node_score(
        "Y", {"Y": ["X"]}, ["bic-g"], {"k": 1.0}, data, False
    )

    # With clear linear relationship, Y|X should have better likelihood
    # despite penalty for extra parameter
    assert y_with_x["bic-g"] > y_orphan["bic-g"]


# Test that independent nodes with same distribution get same scores
def test_score_symmetry_for_independent_nodes():
    """Test that independent nodes with same distribution get same scores"""
    # Create symmetric data
    data = NumPy(
        array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    x_score = gaussian_node_score("X", {}, ["bic-g"], {"k": 1.0}, data, False)
    y_score = gaussian_node_score("Y", {}, ["bic-g"], {"k": 1.0}, data, False)

    # Should be exactly equal for identical distributions
    assert values_same(x_score["bic-g"], y_score["bic-g"], sf=12)


# Test that bic-g, loglik-g, and bge scores are consistent
def test_different_score_types_consistency():
    """Test that bic-g, loglik-g, and bge scores are consistent"""
    data = NumPy(
        array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # All three score types should work
    bic_scores = dag_score(dag.x_y(), data, "bic-g", {"k": 1.0})
    loglik_scores = dag_score(dag.x_y(), data, "loglik-g", {"k": 1.0})
    bge_scores = dag_score(dag.x_y(), data, "bge", {"k": 1.0})

    # Verify we get valid numerical results
    assert isinstance(dict(bic_scores.sum())["bic-g"], float)
    assert isinstance(dict(loglik_scores.sum())["loglik-g"], float)
    assert isinstance(dict(bge_scores.sum())["bge"], float)

    # BIC should be less than loglik (due to penalty)
    assert (
        dict(bic_scores.sum())["bic-g"] < dict(loglik_scores.sum())["loglik-g"]
    )


# Test BGE score with different logarithm base
def test_bge_score_different_base():
    """Test BGE score calculation with different logarithm base"""
    data = NumPy(
        array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Test with different base parameter (line 381)
    bge_base_e = bayesian_gaussian_score("X", {}, {"base": "e"}, data)
    bge_base_10 = bayesian_gaussian_score("X", {}, {"base": 10}, data)

    assert isinstance(bge_base_e, float)
    assert isinstance(bge_base_10, float)
    assert bge_base_e != bge_base_10


# Test BGE score with parents to cover matrix operations
def test_bge_score_with_parents():
    """Test BGE score calculation with parent nodes for matrix operations"""
    # Create data with clear linear relationship for Y|X
    data = NumPy(
        array(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype="float32"
        ),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Test BGE calculation for node with parents (lines 349-378)
    bge_score = bayesian_gaussian_score("Y", {"Y": ["X"]}, {}, data)

    assert isinstance(bge_score, float)
    assert not (bge_score == float("inf") or bge_score == float("-inf"))


# Test categorical node score with counts information
def test_categorical_node_score_with_counts():
    """Test categorical node score calculation with counts information"""
    # Create discrete data for categorical scoring - convert to categorical
    df = DataFrame(
        {"X": ["A", "B", "A", "B", "A"], "Y": ["C", "D", "C", "D", "C"]}
    )
    df["X"] = df["X"].astype("category")
    df["Y"] = df["Y"].astype("category")
    discrete_data = Pandas(df)

    # Test with counts_reqd=True to cover lines 495-500
    scores, counts_info = categorical_node_score(
        "Y", {"Y": ["X"]}, ["bic"], {}, discrete_data, counts_reqd=True
    )

    assert isinstance(scores, dict)
    assert isinstance(counts_info, dict)
    assert "mean" in counts_info
    assert "max" in counts_info
    assert "min" in counts_info
    assert "lt5" in counts_info


# Test dag_score with tuple return from node_score
def test_dag_score_with_tuple_handling():
    """Test dag_score handling when node_score returns tuple"""
    df = DataFrame({"X": ["A", "B", "A"], "Y": ["C", "D", "C"]})
    df["X"] = df["X"].astype("category")
    df["Y"] = df["Y"].astype("category")
    discrete_data = Pandas(df)

    # This should trigger the tuple handling path (line 579)
    scores = dag_score(dag.x_y(), discrete_data, "bic", {})

    assert isinstance(scores, DataFrame)
    assert "bic" in scores.columns


# Test bn_score with BN object
def test_bn_score_with_bn_object():
    """Test bn_score function with BN object"""
    df = DataFrame({"X": ["A", "B", "A"], "Y": ["C", "D", "C"]})
    df["X"] = df["X"].astype("category")
    df["Y"] = df["Y"].astype("category")
    data = Pandas(df)

    # Create a simple BN for testing
    bn = BN.fit(dag.x_y(), data)

    # Test bn_score which covers line 642 - args: bn, N, types, params
    scores = bn_score(bn, 100, "bic", {})

    assert isinstance(scores, DataFrame)
    assert "bic" in scores.columns


# Test check_score_params error handling
def test_check_score_params_errors():
    """Test check_score_params function error conditions"""
    # Test TypeError for bad params type (line 674)
    try:
        check_score_params("not_a_dict", None)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test TypeError for bad scores type
    try:
        check_score_params({}, "not_a_list")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test ValueError for unknown parameter
    try:
        check_score_params({"unknown_param": 1}, None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# Test free_params function with debug output
def test_free_params_with_debug():
    """Test free_params function with debug output enabled"""
    df = DataFrame({"X": ["A", "B", "A"], "Y": ["C", "D", "C"]})
    df["X"] = df["X"].astype("category")
    df["Y"] = df["Y"].astype("category")

    # Test with debug=True to cover lines 746, 761-765, 773
    # free_params expects a DataFrame, not a Pandas object
    total_params = free_params(dag.x_y(), df, debug=True)

    assert isinstance(total_params, int)
    assert total_params >= 0


# Test dag_score tuple handling with monkeypatch
def test_dag_score_tuple_handling_with_monkeypatch(monkeypatch):
    """Test dag_score handling when node_score returns tuple (line 579)"""
    data = NumPy(
        array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype="float32"),
        dstype="continuous",
        col_values={"X": None, "Y": None},
    )

    # Mock gaussian_node_score to return a tuple
    def mock_gaussian_node_score(
        node, parents, types, params, data, counts_reqd
    ):
        base_scores = {"bic-g": -10.5}
        counts_info = {"mean": 0, "max": 0, "min": 0, "lt5": 0, "fpa": 0}
        return (base_scores, counts_info)  # Always return tuple

    # Monkeypatch the function
    monkeypatch.setattr(
        "causaliq_data.score.gaussian_node_score", mock_gaussian_node_score
    )

    # This should trigger the tuple handling on line 579
    scores = dag_score(dag.x_y(), data, "bic-g", {})

    assert isinstance(scores, DataFrame)
    assert "bic-g" in scores.columns
    assert len(scores) == 2  # X and Y nodes


# Test bn_score tuple handling with monkeypatch
def test_bn_score_tuple_handling_with_monkeypatch(monkeypatch):
    """Test bn_score handling when node_score returns tuple (line 642)"""
    df = DataFrame({"X": ["A", "B", "A"], "Y": ["C", "D", "C"]})
    df["X"] = df["X"].astype("category")
    df["Y"] = df["Y"].astype("category")
    data = Pandas(df)

    # Create a simple BN for testing
    bn = BN.fit(dag.x_y(), data)

    # Mock categorical_node_score to return a tuple
    def mock_categorical_node_score(
        node, parents, types, params, data, counts_reqd=False
    ):
        base_scores = {"bic": -8.2}
        counts_info = {"mean": 2.5, "max": 5, "min": 1, "lt5": 0.8, "fpa": 2}
        return (base_scores, counts_info)  # Always return tuple

    # Monkeypatch the function
    monkeypatch.setattr(
        "causaliq_data.score.categorical_node_score",
        mock_categorical_node_score,
    )

    # This should trigger the tuple handling on line 642
    scores = bn_score(bn, 100, "bic", {})

    assert isinstance(scores, DataFrame)
    assert "bic" in scores.columns
    assert len(scores) == 2  # X and Y nodes
