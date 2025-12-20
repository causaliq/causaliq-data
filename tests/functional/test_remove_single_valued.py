#   Test remove_single_valued() which removes single-valued variables
#   from a BN.

import pytest
from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import DAG
from pandas import DataFrame

import tests.fixtures.example_dags as ex_dag
from causaliq_data.pandas import Pandas
from causaliq_data.preprocess import remove_single_valued

TESTDATA_DIR = "tests/data/functional"


# Test TypeError conditions for remove_single_valued function
def test_bn_remove_single_valued_type_error_1():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    with pytest.raises(TypeError):
        remove_single_valued(
            bn,
        )
    with pytest.raises(TypeError):
        remove_single_valued(bn, 52.1)
    with pytest.raises(TypeError):
        remove_single_valued(bn, {"A": ["0"], "B": ["1"]})


# Test ValueError when data has only one row
def test_bn_remove_single_valued_value_error_1():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    data = DataFrame({"A": ["0"], "B": ["1"], "C": ["1"]}, dtype="category")
    with pytest.raises(ValueError):
        remove_single_valued(bn, data)


# Test ValueError when all variables are single-valued
def test_bn_remove_single_valued_value_error_2():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    data = DataFrame(
        {"A": ["0", "0"], "B": ["1", "1"], "C": ["1", "1"]}, dtype="category"
    )
    with pytest.raises(ValueError):
        remove_single_valued(bn, data)


# Test ValueError when only one variable is multi-valued
def test_bn_remove_single_valued_value_error_3():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    data = DataFrame(
        {"A": ["0", "0"], "B": ["0", "1"], "C": ["1", "1"]}, dtype="category"
    )
    with pytest.raises(ValueError):
        remove_single_valued(bn, data)


# Test successful case when no variables need removing
def test_bn_remove_single_valued_abc_ok_1():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    data = DataFrame(
        {"A": ["0", "1"], "B": ["1", "0"], "C": ["1", "0"]}, dtype="category"
    )
    new_bn, new_data, removed = remove_single_valued(bn, data)
    assert removed == []
    assert bn == new_bn
    assert not len(new_data.compare(data).columns)


# Test successful case when one variable needs removing
def test_bn_remove_single_valued_abc_ok_2():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    data = DataFrame(
        {"A": ["0", "1"], "B": ["1", "0"], "C": ["1", "1"]}, dtype="category"
    )
    new_bn, new_data, removed = remove_single_valued(bn, data)
    assert removed == ["C"]
    expected_data = DataFrame(
        {"A": ["0", "1"], "B": ["1", "0"]}, dtype="category"
    )
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    print(type(expected_data))
    return
    expected_bn = BN.fit(ex_dag.ab(), expected_data)
    assert expected_bn == new_bn


# Test successful case when variable A needs removing
def test_bn_remove_single_valued_abc_ok_3():
    bn = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")
    data = DataFrame(
        {"A": ["0", "0"], "B": ["1", "0"], "C": ["1", "0"]}, dtype="category"
    )
    new_bn, new_data, removed = remove_single_valued(bn, data)
    assert removed == ["A"]
    expected_data = DataFrame(
        {"B": ["1", "0"], "C": ["1", "0"]}, dtype="category"
    )
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    expected_bn = BN.fit(DAG(["B", "C"], [("B", "->", "C")]), expected_data)
    assert expected_bn == new_bn


# Test cancer dataset when no variables need removing
def test_bn_remove_single_valued_cancer_ok_1():
    bn = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = bn.generate_cases(100)
    new_bn, new_data, removed = remove_single_valued(bn, data)
    assert removed == []
    assert bn == new_bn
    assert not len(new_data.compare(data).columns)


# Test cancer dataset when Xray variable needs removing
def test_bn_remove_single_valued_cancer_ok_2():
    bn = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = bn.generate_cases(100).assign(Xray="negative")
    new_bn, new_data, removed = remove_single_valued(bn, data)
    assert removed == ["Xray"]
    expected_data = data.drop(labels=["Xray"], axis="columns").copy()
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    expected_bn = BN.fit(
        DAG(
            ["Smoker", "Cancer", "Pollution", "Dyspnoea"],
            [
                ("Smoker", "->", "Cancer"),
                ("Pollution", "->", "Cancer"),
                ("Cancer", "->", "Dyspnoea"),
            ],
        ),
        expected_data,
    )
    assert expected_bn == new_bn


# Test cancer dataset when Xray and Smoker variables need removing
def test_bn_remove_single_valued_cancer_ok_3():
    bn = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = bn.generate_cases(100).assign(Xray="negative", Smoker="False")
    new_bn, new_data, removed = remove_single_valued(bn, data)
    assert removed == ["Smoker", "Xray"]
    expected_data = data.drop(labels=["Xray", "Smoker"], axis="columns").copy()
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    expected_bn = BN.fit(
        DAG(
            ["Cancer", "Pollution", "Dyspnoea"],
            [("Pollution", "->", "Cancer"), ("Cancer", "->", "Dyspnoea")],
        ),
        expected_data,
    )
    assert expected_bn == new_bn


# Test Pathfinder dataset with 1000 rows for single-valued removal
@pytest.mark.slow
def test_bn_remove_single_valued_pathfinder_1K_ok():
    bn = read_bn(TESTDATA_DIR + "/discrete/verylarge/pathfinder.dsc")
    data = bn.generate_cases(1000)
    _, _, removed = remove_single_valued(bn, data)
    assert removed == ["F13", "F15", "F27", "F69", "F72", "F75"]


# Test Pathfinder dataset with 2000 rows for single-valued removal
@pytest.mark.slow
def test_bn_remove_single_valued_pathfinder_2K_ok():
    bn = read_bn(TESTDATA_DIR + "/discrete/verylarge/pathfinder.dsc")
    data = bn.generate_cases(2000)
    _, _, removed = remove_single_valued(bn, data)
    assert removed == ["F15", "F27", "F72", "F75"]
