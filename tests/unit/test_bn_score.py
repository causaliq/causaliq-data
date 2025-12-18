import pytest
from causaliq_core.bn import BN
from causaliq_core.utils import dicts_same
from pandas import DataFrame

import tests.fixtures.example_dags as dag
from causaliq_data.pandas import Pandas
from causaliq_data.score import bn_score, dag_score, free_params

TYPES = ["aic", "bic", "loglik"]
DEFAULT_PARAMS = {"iss": 1.0, "prior": "uniform", "base": "e"}


# Test bn_score with bad primary arg types for BN.score
def test_bn_score_type_error_1():
    data = DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(TypeError):
        bn_score(bn)
    with pytest.raises(TypeError):
        bn_score(bn, data, "bic", {})
    with pytest.raises(TypeError):
        bn_score(bn, True, "bic", {})
    with pytest.raises(TypeError):
        bn_score(bn, 10.7, "bic", {})
    with pytest.raises(TypeError):
        bn_score(bn, 1000, 37, {})
    with pytest.raises(TypeError):
        bn_score(bn, 200, "bic", True)


# Test bn_score with bad score type
def test_bn_score_type_error_2():
    data = DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(TypeError):
        bn_score(bn, 1000, [37])


# Test bn_score with bad 'base' score param type
def test_bn_score_type_error_3():
    data = DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(TypeError):
        bn_score(bn, 800, "bic", {"base": 2.2})
    with pytest.raises(TypeError):
        bn_score(bn, 40, "bic", {"base": True})


# Test bn_score with bad 'prior' score param type
def test_bn_score_type_error_4():
    data = DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(TypeError):
        bn_score(bn, 10000, "bds", {"prior": 12})


# Test bn_score with bad 'iss' score param type
def test_bn_score_type_error_5():
    data = DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(TypeError):
        bn_score(bn, 12, "bde", {"prior": "uniform", "iss": "should be num"})


# Test bn_score with non-positive sample size
def test_bn_score_value_error_6():
    data = DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 0, "bic", {})
    with pytest.raises(ValueError):
        bn_score(bn, -1, "bic", {})


# Test bn_score with bad arg types
def test_bn_score_type_error_7():
    graph = dag.ab()
    data = DataFrame({"A": ["1", "0"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(graph, Pandas(df=data))
    with pytest.raises(TypeError):
        bn_score(graph, 12, "bic", {})
    with pytest.raises(TypeError):
        bn_score("bn", 2, "bic", {})
    with pytest.raises(TypeError):
        bn_score(data, 2, "bic", {})
    with pytest.raises(TypeError):
        bn_score(bn, data, "bic", {})
    with pytest.raises(TypeError):
        bn_score(bn, 37, {})
    with pytest.raises(TypeError):
        bn_score(bn, 42, "bic", ["base"])
    with pytest.raises(TypeError):
        bn_score(bn, None, "bic", {})


# Test bn_score with unsupported score types
def test_bn_score_value_error_8():
    data = DataFrame({"A": ["2", "0"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 2, [], {})
    with pytest.raises(ValueError):
        bn_score(bn, 10000000, "unsupported", {})
    with pytest.raises(ValueError):
        bn_score(bn, 1, ["unsupported", "bic"], {})


# Test bn_score with bad N value
def test_bn_score_value_error_9():
    data = DataFrame({"A": ["2", "0"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 0, "bic", {})
    with pytest.raises(ValueError):
        bn_score(bn, -10, "bic", {})


# Test bn_score with unknown score parameter
def test_bn_score_value_error_12():
    data = DataFrame({"A": ["2", "0"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"unsupported": 3})
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"base": 2, "unsupported": 3})


# Test bn_score with bad "base" score param value
def test_bn_score_value_error_13():
    data = DataFrame({"A": ["2", "0"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"base": 7})
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"base": "2"})


# Test bn_score with bad "prior" score param value
def test_bn_score_value_error_14():
    data = DataFrame({"A": ["2", "0"], "B": ["0", "1"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"prior": "unsupported"})


# Test bn_score with bad "iss" score param value
def test_bn_score_value_error_15():
    data = DataFrame({"A": ["2", "0"], "B": ["1", "2"]}, dtype="category")
    bn = BN.fit(dag.ab(), Pandas(df=data))
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"iss": 0})
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"iss": 0.0})
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"iss": -1.0})
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"iss": 1e-10})
    with pytest.raises(ValueError):
        bn_score(bn, 2, "bic", {"iss": 10000000})


# Test simple A->B graph with two rows of data
def test_bn_score_ab2():
    graph = dag.ab()
    data = Pandas(
        DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    )
    bn = BN.fit(graph, data)

    assert free_params(bn.dag, data.sample) == 3  # 1 free param at A, 2 at B
    scores = bn_score(bn, 2, TYPES, {"base": 2})

    # probability of first row is P(A=0).P(B=0|A=0) = 0.5
    # probability of second row is P(A=1).P(B=1|A=0) = 0.5
    # Likelihood is 0.25, log(base2) likelihood is -2
    # BIC adjustment = 0.5 * free_params * ln(N), A=-0.5, B=-1, -1.5 in all
    # AIC adjustment = -free_params = -3 in all

    assert dicts_same(
        dict(scores.sum()), {"bic": -3.5, "loglik": -2, "aic": -5}
    )

    # all parental combos present in data so DAG & BN scores will match

    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test A->B graph with 4 rows of data
def test_bn_score_ab3():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "0", "1"]},
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 4, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 3  # 1 free param at A, 2 at B

    # all parental combos present in data so DAG & BN scores will match

    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test all possible binary cases for A->B graph
def test_bn_score_ab4():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "0", "1"]},
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 4, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 3  # 1 free param at A, 2 at B

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test 2 unbalanced binary cases for A->B graph
def test_bn_score_ab5():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame(
            {"A": ["0", "1", "1", "1"], "B": ["0", "1", "1", "1"]},
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 4, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 3  # 1 free param at A, 2 at B

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))

    # different bases

    scores = bn_score(bn, 4, TYPES, {"base": 10})
    dag_scores = dag_score(graph, data, TYPES, {"base": 10})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    scores = bn_score(bn, 4, TYPES, {"base": "e"})
    dag_scores = dag_score(graph, data, TYPES, {"base": "e"})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test A binary, B categorical for A->B graph
def test_bn_score_ab6():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "1", "2"]},
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 4, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 5  # 1 free param at A, 4 at B

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))

    # different bases

    scores = bn_score(bn, 4, TYPES, {"base": 10})
    dag_scores = dag_score(graph, data, TYPES, {"base": 10})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    scores = bn_score(bn, 4, TYPES, {"base": "e"})
    dag_scores = dag_score(graph, data, TYPES, {"base": "e"})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test A and B categorical for A->B graph
def test_bn_score_ab7():
    graph = dag.ab()  # A --> B
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "2", "2", "2"],
                "B": ["0", "1", "1", "2", "0", "1", "1"],
            },
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 7, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 8  # 2 free param at A, 6 at B

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))

    # different bases

    scores = bn_score(bn, 7, TYPES, {"base": 10})
    dag_scores = dag_score(graph, data, TYPES, {"base": 10})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    scores = bn_score(bn, 7, TYPES, {"base": "e"})
    dag_scores = dag_score(graph, data, TYPES, {"base": "e"})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test A-->B-->C, A&B categorical, C binary
def test_bn_score_abc1():
    graph = dag.abc()  # A-->B-->C
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "2", "2", "2"],
                "B": ["0", "1", "1", "2", "0", "1", "1"],
                "C": ["0", "1", "1", "1", "0", "1", "1"],
            },
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 7, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 11  # free params: A=2, B=6, C=3

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    print(scores)

    # different bases

    scores = bn_score(bn, 7, TYPES, {"base": 10})
    dag_scores = dag_score(graph, data, TYPES, {"base": 10})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    scores = bn_score(bn, 7, TYPES, {"base": "e"})
    dag_scores = dag_score(graph, data, TYPES, {"base": "e"})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test A-->B-->C, A, B & C categorical
def test_bn_score_abc2():
    graph = dag.abc()  # A-->B-->C
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "0", "1", "1", "2", "2", "2", "3", "3", "3"],
                "B": ["0", "1", "1", "2", "0", "1", "1", "3", "3", "3"],
                "C": ["0", "1", "1", "1", "0", "1", "1", "2", "2", "1"],
            },
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 10, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 23  # free params: A=3,B=12,C=8

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    print(scores)

    # different bases

    scores = bn_score(bn, 10, TYPES, {"base": 10})
    dag_scores = dag_score(graph, data, TYPES, {"base": 10})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    scores = bn_score(bn, 10, TYPES, {"base": "e"})
    dag_scores = dag_score(graph, data, TYPES, {"base": "e"})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test A-->C<--B, A, B & C binary, ABSENT pvs
def test_bn_score_ac_bc1():
    graph = dag.ac_bc()  # A-->C<--B
    data = Pandas(
        DataFrame(
            {"A": ["1", "0"], "B": ["0", "1"], "C": ["0", "1"]},
            dtype="category",
        )
    )
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 2, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 6  # free params: A=1, B=1, C=4

    assert bn.estimated_pmfs == {"C": 2}  # 2 estimated PMFs for node C
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})

    # BN and DAG scores will be same for nodes A and B

    assert dicts_same(scores.loc["A"].to_dict(), dag_scores.loc["A"].to_dict())
    assert dicts_same(scores.loc["B"].to_dict(), dag_scores.loc["B"].to_dict())

    # BN and DAG scores for node C different, since estimated PMFs for BN
    # introduce new entries into CPT for node C for (A=0,B=0) and (A=1,B=2)
    # with PMF ('0': 0.5, '1': 0.5) which introduces entropy into node C

    assert dicts_same(
        scores.loc["C"].to_dict(), {"aic": -5.0, "bic": -3.0, "loglik": -1.0}
    )


# Test A-->C<--B, A, B & C binary, all pvs present
def test_bn_score_ac_bc2():
    graph = dag.ac_bc()  # A-->C<--B
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
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 4, TYPES, {"base": 2})

    assert free_params(bn.dag, data.sample) == 6  # free params: A=1, B=1, C=4

    assert bn.estimated_pmfs == {}  # no estimated PMFs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))


# Test Cancer BN, all PVs present
def test_bn_score_cancer():
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
    bn = BN.fit(graph, data)
    scores = bn_score(bn, 4, TYPES, {"base": 2})
    print(scores)

    assert free_params(bn.dag, data.sample) == 10  # S=1,P=1,C=4,D=2,X=2

    # all parental combos present in data so DAG & BN scores will match

    assert bn.estimated_pmfs == {}  # no estimated pmfs
    dag_scores = dag_score(graph, data, TYPES, {"base": 2})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))

    # different bases

    scores = bn_score(bn, 4, TYPES, {"base": 10})
    dag_scores = dag_score(graph, data, TYPES, {"base": 10})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
    scores = bn_score(bn, 4, TYPES, {"base": "e"})
    dag_scores = dag_score(graph, data, TYPES, {"base": "e"})
    assert dicts_same(dict(scores.sum()), dict(dag_scores.sum()))
