import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.utils import dicts_same
from pandas import DataFrame

import tests.fixtures.example_dags as dag
from causaliq_data import NumPy, Oracle, Pandas
from causaliq_data.score import (
    BAYESIAN_SCORES,
    ENTROPY_SCORES,
    dag_score,
    free_params,
)

TESTDATA_DIR = "tests/data/functional"
ENTROPY_PARAMS = {"base": "e", "k": 1.0}
BAYESIAN_PARAMS = {"iss": 1.0, "prior": "uniform"}


# --- Failure cases


# Test dag_score with bad primary arg types for DAG.score
def test_graph_score_type_error_1():
    graph = dag.ab()
    with pytest.raises(TypeError):
        dag_score(graph)
    with pytest.raises(TypeError):
        dag_score(graph, 10, "bic", {})
    with pytest.raises(TypeError):
        dag_score(graph, 10, "bic", {})
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "0"], "B": ["1", "1"]}, dtype="category"),
            37,
        )
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "0"], "B": ["1", "1"]}, dtype="category"),
            "bic",
            True,
        )


# Test dag_score with bad score type
def test_graph_score_type_error_2():
    graph = dag.ab()
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "1"], "B": ["1", "1"]}, dtype="category"),
            [37],
        )


# Test dag_score with bad 'base' score param type
def test_graph_score_type_error_3():
    graph = dag.ab()
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category"),
            "bic",
            {"base": 2.2},
        )
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category"),
            "bic",
            {"base": True},
        )


# Test dag_score with bad 'prior' score param type
def test_graph_score_type_error_4():
    graph = dag.ab()
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category"),
            "bde",
            {"prior": 12},
        )


# Test dag_score with bad 'iss' score param type
def test_graph_score_type_error_5():
    graph = dag.ab()
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category"),
            "bds",
            {"prior": "uniform", "iss": "should be num"},
        )


# Test dag_score with bad 'k' score param type
def test_graph_score_type_error_6():
    graph = dag.ab()
    with pytest.raises(TypeError):
        dag_score(
            graph,
            DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category"),
            "bic",
            {"k": "should be num"},
        )


# Test dag_score with DAG/Data column mismatch
def test_graph_score_value_error_7():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["0", "1"], "C": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "aic", {})


# Test dag_score with single-valued variables
def test_graph_score_value_error_8():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["1", "0"], "B": ["0", "0"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "aic", {})


# Test dag_score with bad arg types
def test_dag_score_type_error_1():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["1", "0"], "B": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        dag_score({"A": ["1", "0"], "B": ["0", "1"]}, data, "bic", {})
    with pytest.raises(TypeError):
        dag_score("graph", data, "bic", {})
    with pytest.raises(TypeError):
        dag_score("graph", data, 37, {})
    with pytest.raises(TypeError):
        dag_score(graph, data, "bic", ["base"])
    with pytest.raises(TypeError):
        dag_score(graph, None, "bic", {})


# Test dag_score with bad base type
def test_dag_score_type_error_2():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["1", "0"], "B": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        dag_score(graph, data, "bic", {"base": []})


# Test dag_score with bad k type
def test_dag_score_type_error_3():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["1", "0"], "B": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        dag_score(graph, data, "bic", {"k": "should be int/float"})
    with pytest.raises(TypeError):
        dag_score(graph, data, "bic", {"k": {}})


# Test dag_score with bad prior type
def test_dag_score_type_error_4():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["1", "0"], "B": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        dag_score(graph, data, "bde", {"prior": 1})
    with pytest.raises(TypeError):
        dag_score(graph, data, "bde", {"prior": {}})


# Test dag_score with bad iss type
def test_dag_score_type_error_5():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["1", "0"], "B": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        dag_score(graph, data, "bds", {"iss": "should be int/float"})
    with pytest.raises(TypeError):
        dag_score(graph, data, "bds", {"iss": {}})


# Test dag_score cannot score an oracle type
def test_dag_score_type_error_6():
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    with pytest.raises(TypeError):
        dag_score(bn.dag, Oracle(bn), "bic", {})


# Test dag_score with unsupported score types
def test_dag_score_value_error_2():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["0", "1"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, [], {})
    with pytest.raises(ValueError):
        dag_score(graph, data, "unsupported", {})
    with pytest.raises(ValueError):
        dag_score(graph, data, ["unsupported", "bic"], {})


# Test dag_score with single-valued data type
def test_dag_score_value_error_3():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "1"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {})


# Test dag_score with data/dag column mismatch
def test_dag_score_value_error_4():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "C": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {})


# Test dag_score with unknown score parameter
def test_dag_score_value_error_5():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"unsupported": 3})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"base": 2, "unsupported": 3})


# Test dag_score with bad "base" score param value
def test_dag_score_value_error_6():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"base": 7})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"base": "2"})


# Test dag_score with bad "prior" score param value
def test_dag_score_value_error_7():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"prior": "unsupported"})


# Test dag_score with bad "iss" score param value
def test_dag_score_value_error_8():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"iss": 0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"iss": 0.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"iss": -1.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"iss": 1e-10})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"iss": 10000000})


# Test dag_score with bad "k" score param value
def test_dag_score_value_error_9():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"k": 0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"k": 0.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"k": -1.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"k": 1e-10})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"k": 10000000})


# Test for irrelevant score parameters disabled for now (error_11 & 8)


# irrelevant parameters for entropy
def xtest_dag_score_value_error_10():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"prior": "uniform"})
    with pytest.raises(ValueError):
        dag_score(graph, data, "aic", {"prior": "uniform"})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bic", {"iss": 2.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, "aic", {"iss": 10})


# irrelevant parameters for bayesian
def xtest_dag_score_value_error_11():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        dag_score(graph, data, "bde", {"k": 4})
    with pytest.raises(ValueError):
        dag_score(graph, data, "k2", {"k": 4})
    with pytest.raises(ValueError):
        dag_score(graph, data, "bde", {"k": 10.0})


# --- Successful score cases


# Test A -> B graph with 2 rows
def test_dag_score_ab1():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    )
    assert free_params(graph, data.as_df()) == 3
    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "loglik": -1.38629436111989,
        "bic": -2.42601513195981,
        "aic": -4.38629436111989,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))
    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"bic": -3.5, "loglik": -2, "aic": -5}
    )


# Test A -> B graph with 2 rows, set k = 2
def test_dag_score_ab2():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    )
    assert free_params(graph, data.as_df()) == 3
    params = dict(ENTROPY_PARAMS)
    params.update({"k": 2})
    scores = dag_score(graph, data, ENTROPY_SCORES, params)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "loglik": -1.38629436111989,
        "bic": -3.46573590279973,
        "aic": -7.38629436111989,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))
    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2, "k": 2})
    assert dicts_same(dict(scores.sum()), {"bic": -5, "loglik": -2, "aic": -8})


# Test A -> B graph with 4 rows
def test_dag_score_ab3():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "0", "1"]},
            dtype="category",
        )
    )
    assert free_params(graph, data.as_df()) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "loglik": -5.54517744447956,
        "bic": -7.6246189861594,
        "aic": -8.54517744447956,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"bic": -11, "loglik": -8, "aic": -11}
    )


# Test A -> B graph with 4 rows (variation)
def test_dag_score_ab4():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "1", "1"]},
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "loglik": -4.15888308335967,
        "bic": -6.23832462503951,
        "aic": -7.15888308335967,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(dict(scores.sum()), {"bic": -9, "loglik": -6, "aic": -9})


# Test A -> B graph with 4 rows (another variation)
def test_dag_score_ab5():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {"A": ["0", "1", "1", "1"], "B": ["0", "1", "1", "1"]},
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "loglik": -2.24934057847523,
        "bic": -4.32878212015507,
        "aic": -5.24934057847523,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()),
        {"bic": -6.245112498, "loglik": -3.245112498, "aic": -6.245112498},
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"bic": -1.879966188, "loglik": -0.9768762012, "aic": -3.976876201},
    )


# Test A -> B graph with 4 rows, 3 states
def test_dag_score_ab6():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "1", "2"]},
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 5

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn
    # (captured 2025-12-17 with correct test data)
    bnlearn_reference = {
        "loglik": -5.54517744447956,
        "bic": -9.01091334727929,
        "aic": -10.5451774444796,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"bic": -13, "loglik": -8, "aic": -13}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"bic": -3.913389944, "loglik": -2.408239965, "aic": -7.408239965},
    )


# Test A -> B graph with 4 rows, 3 states (extended)
def test_dag_score_ab7():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "2", "2", "2"],
                "B": ["0", "1", "1", "2", "0", "1", "1"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 8

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn
    # (captured 2025-12-17 with correct test data)
    bnlearn_reference = {
        "loglik": -12.2350766822673,
        "bic": -20.0187172784886,
        "aic": -20.2350766822673,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()),
        {"bic": -28.88090414, "loglik": -17.65148445, "aic": -25.65148445},
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    print(dict(scores.sum()))
    assert dicts_same(
        dict(scores.sum()),
        {"bic": -8.694018449, "loglik": -5.313626289, "aic": -13.31362629},
    )


# Test Bayesian scores for A -> B, 2 rows
def test_dag_score_ab8():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    )
    assert free_params(graph, data.as_df()) == 3
    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "bde": -3.46573590279973,
        "k2": -3.17805383034795,
        "bdj": -4.56434819146784,
        "bds": -3.46573590279973,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))
    assert dicts_same(
        dict(scores.sum()),
        {
            "bde": -3.465735903,
            "bdj": -4.564348191,
            "bds": -3.465735903,
            "k2": -3.178053830,
        },
    )


# Test Bayesian scores for A -> B, 2 rows, ISS =5
def test_dag_score_ab9():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    )
    assert free_params(graph, data.as_df()) == 3
    params = BAYESIAN_PARAMS.copy()
    params.update({"iss": 5})
    scores = dag_score(graph, data, BAYESIAN_SCORES, params)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "bde": -2.95491027903374,
        "k2": -3.17805383034795,
        "bdj": -4.56434819146784,
        "bds": -2.95491027903374,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))
    assert dicts_same(
        dict(scores.sum()),
        {
            "bde": -2.954910279,
            "bdj": -4.564348191,
            "bds": -2.954910279,
            "k2": -3.178053830,
        },
    )


# Test Bayesian scores, A -> B, 8 rows
def test_dag_score_ab10():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "2", "2", "2"],
                "B": ["0", "1", "1", "2", "0", "1", "1"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 8

    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "bde": -21.6031483094249,
        "k2": -17.3016371504117,
        "bdj": -21.0268842486965,
        "bds": -21.6031483094249,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))


# Test Bayesian scores, A -> B, 8 rows, ISS=10.0
def test_dag_score_ab11():
    graph = dag.ab()
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "2", "2", "2"],
                "B": ["0", "1", "1", "2", "0", "1", "1"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 8

    params = BAYESIAN_PARAMS.copy()
    params.update({"iss": 10.0})
    scores = dag_score(graph, data, BAYESIAN_SCORES, params)
    # Reference values from bnlearn (captured 2025-12-17)
    bnlearn_reference = {
        "bde": -16.490655130222,
        "k2": -17.3016371504117,
        "bdj": -21.0268842486965,
        "bds": -16.490655130222,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))


# Test single-valued data type allowed
def test_dag_score_ab12():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["2", "0"], "B": ["1", "1"]}, dtype="category")
    )
    bic = dag_score(graph, data, "bic", {"unistate_ok": True}).to_dict()["bic"]
    assert dicts_same({"A": -1.732867951, "B": 0}, bic)


# Test A -> B -> C graph, 7 rows
def test_dag_score_abc1():
    graph = dag.abc()
    data = Pandas(
        DataFrame(
            {"A": ["0", "1"], "B": ["0", "1"], "C": ["0", "1"]},
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 5

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from bnlearn
    # (captured 2025-12-17 with correct test data)
    bnlearn_reference = {
        "loglik": -1.38629436111989,
        "bic": -3.11916231251975,
        "aic": -6.38629436111989,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"aic": -7.0, "bic": -4.5, "loglik": -2.0}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {
            "aic": -5.6020599913279625,
            "bic": -1.3546349804879154,
            "loglik": -0.6020599913279623,
        },
    )


# Test A -> B -> C, Bayesian scores
def test_dag_score_abc2():
    graph = dag.abc()
    data = Pandas(
        DataFrame(
            {"A": ["0", "1"], "B": ["0", "1"], "C": ["0", "1"]},
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 5

    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    # Reference values from bnlearn
    # (captured 2025-12-17 with correct test data)
    bnlearn_reference = {
        "bde": -4.85203026391962,
        "k2": -4.56434819146784,
        "bdj": -5.95064255258773,
        "bds": -4.85203026391962,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"aic": -7.0, "bic": -4.5, "loglik": -2.0}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {
            "aic": -5.6020599913279625,
            "bic": -1.3546349804879154,
            "loglik": -0.6020599913279623,
        },
    )


# Test A -> C <- B graph, 3 rows
def test_dag_score_ac_bc1():
    graph = dag.ac_bc()
    data = Pandas(
        DataFrame(
            {"A": ["1", "0"], "B": ["0", "1"], "C": ["0", "1"]},
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from causaliq computation (captured 2025-12-18)
    bnlearn_reference = {
        "loglik": -2.772588722239781,
        "bic": -4.852030263919617,
        "aic": -8.772588722239782,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"aic": -10, "bic": -7, "loglik": -4}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -7.204119983, "bic": -2.107209970, "loglik": -1.204119983},
    )


# Test A -> C <- B graph, 4 rows
def test_dag_score_ac_bc2():
    graph = dag.ac_bc()
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1"],
                "B": ["0", "1", "0", "1"],
                "C": ["0", "0", "0", "1"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from causaliq computation (captured 2025-12-18)
    bnlearn_reference = {
        "loglik": -5.545177444479562,
        "bic": -9.704060527839234,
        "aic": -11.545177444479563,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"aic": -14, "bic": -14, "loglik": -8}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -8.408239965, "bic": -4.214419939, "loglik": -2.408239965},
    )


# Test A -> C <- B graph, 5 rows
def test_dag_score_ac_bc3():
    graph = dag.ac_bc()
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "1"],
                "B": ["0", "1", "0", "1", "1"],
                "C": ["0", "0", "0", "1", "0"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from causaliq computation (captured 2025-12-18)
    bnlearn_reference = {
        "loglik": -8.116411031212454,
        "bic": -12.944724768514757,
        "aic": -14.116411031212454,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -17.70950594, "bic": -18.67529023, "loglik": -11.70950594},
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -9.524912524, "bic": -5.621822537, "loglik": -3.524912524},
    )


# Test A -> C <- B graph, 10 rows
def test_dag_score_ac_bc4():
    graph = dag.ac_bc()
    data = {
        "A": ["0", "0", "1", "1", "2", "2", "2", "3", "3", "3"],
        "B": ["0", "1", "1", "2", "0", "1", "1", "3", "3", "3"],
        "C": ["0", "1", "1", "1", "0", "1", "1", "2", "2", "1"],
    }
    data = Pandas(DataFrame(data, dtype="category"))

    assert free_params(graph, data.as_df()) == 38

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from causaliq computation (captured 2025-12-18)
    bnlearn_reference = {
        "loglik": -28.36967323891313,
        "bic": -72.11879000580001,
        "aic": -66.36967323891314,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -78.92878689, "bic": -104.0454207, "loglik": -40.92878689},
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -50.32079254, "bic": -31.32079254, "loglik": -12.32079254},
    )


# Test Cancer graph, 4 rows
def test_dag_score_cancer_1():
    graph = dag.cancer()
    data = Pandas(
        DataFrame(
            {
                "Smoker": ["no", "no", "yes", "yes"],
                "Pollution": ["low", "high", "low", "high"],
                "Cancer": ["no", "no", "yes", "yes"],
                "Dyspnoea": ["no", "yes", "no", "yes"],
                "Xray": ["clear", "clear", "dark", "dark"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 10

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    # Reference values from causaliq computation (captured 2025-12-18)
    bnlearn_reference = {
        "loglik": -8.317766166719343,
        "bic": -15.249237972318797,
        "aic": -18.317766166719345,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2})
    assert dicts_same(
        dict(scores.sum()), {"aic": -22, "bic": -22, "loglik": -12}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -13.61235995, "bic": -6.622659905, "loglik": -3.612359948},
    )


# Test Cancer graph, 4 rows, k = 0.1
def test_dag_score_cancer_2():
    graph = dag.cancer()
    data = Pandas(
        DataFrame(
            {
                "Smoker": ["no", "no", "yes", "yes"],
                "Pollution": ["low", "high", "low", "high"],
                "Cancer": ["no", "no", "yes", "yes"],
                "Dyspnoea": ["no", "yes", "no", "yes"],
                "Xray": ["clear", "clear", "dark", "dark"],
            },
            dtype="category",
        )
    )

    assert free_params(graph, data.as_df()) == 10

    params = ENTROPY_PARAMS.copy()
    print(params)
    scores = dag_score(graph, data, ENTROPY_SCORES, params)
    # Reference values from causaliq computation (captured 2025-12-18)
    bnlearn_reference = {
        "loglik": -8.317766166719343,
        "bic": -15.249237972318797,
        "aic": -18.317766166719345,
    }
    assert dicts_same(bnlearn_reference, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 2, "k": 0.1})
    assert dicts_same(
        dict(scores.sum()), {"aic": -13, "bic": -13, "loglik": -12}
    )

    scores = dag_score(graph, data, ENTROPY_SCORES, {"base": 10, "k": 0.1})
    assert dicts_same(
        dict(scores.sum()),
        {"aic": -4.612359948, "bic": -3.913389944, "loglik": -3.612359948},
    )


# Test Covid reference, 1K rows
def test_dag_score_covid_ref_1():
    data = NumPy.read(
        TESTDATA_DIR + "/experiments/datasets/covid.data.gz",
        dstype="categorical",
        N=1000,
    )
    print(data.as_df().tail())

    ref = read_bn(TESTDATA_DIR + "/discrete/medium/covid.dsc").dag
    params = {"unistate_ok": True, "base": "e"}

    scores = dag_score(ref, data, "bic", params)
    print(scores)
    print((scores["bic"].sum()))

    # Reference value from causaliq computation (captured 2025-12-18)
    assert abs(scores["bic"].sum() - (-37438.61029956387)) < 1e-6
