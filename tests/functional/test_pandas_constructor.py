# Test Pandas constructor and initialization

import pytest
from numpy import nan
from pandas import DataFrame, read_csv

from causaliq_data import Data, Pandas
from tests.conftest import TESTDATA_DIR

# Constructor errors


# Cannot call abstract Data constructor directly
def test_data_type_error_1():
    """Cannot call constructor directly."""
    with pytest.raises(TypeError):
        Data()


# Pandas constructor requires df argument
def test_constructor_type_error_1():
    """No arguments specified."""
    with pytest.raises(TypeError):
        Pandas(df=None)


# Pandas constructor rejects non-DataFrame types
def test_constructor_type_error_2():
    """Bad df type."""
    with pytest.raises(TypeError):
        Pandas(df=None)
    with pytest.raises(TypeError):
        Pandas(df=2)
    with pytest.raises(TypeError):
        Pandas(df=False)
    with pytest.raises(TypeError):
        Pandas(df=12.7)
    with pytest.raises(TypeError):
        Pandas(df=[2])


# Missing data (NaN values) are not supported
def test_constructor_value_error_1():
    """Missing data unsupported."""
    df = DataFrame(
        {"A": ["0", nan, "1"], "B": ["0", "0", "1"]}, dtype="category"
    )
    with pytest.raises(ValueError):
        Pandas(df=df)


# Test constructor setting member variables correctly


# Check initializes all attributes correctly for 2-var categorical dataset
def test_constructor_ab3_ok_1():
    """A,B dataset with 3 rows."""
    df = read_csv(TESTDATA_DIR + "/simple/ab_3.csv", dtype="category")

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert data.order == (0, 1)
    assert data.nodes == ("A", "B")
    assert data.N == 3
    assert data.ext_to_orig == {"A": "A", "B": "B"}
    assert data.node_values == {"A": {"0": 1, "1": 2}, "B": {"0": 2, "1": 1}}
    assert data.node_types == {"A": "category", "B": "category"}
    assert data.dstype == "categorical"

    assert data.get_order() == ("A", "B")

    df = read_csv(TESTDATA_DIR + "/simple/ab_3.csv", dtype="category")
    assert (data.as_df() == df).all().all()


# Verify constructor with 3-var categorical dataset and value counts
def test_constructor_abc5_1_ok():
    """A,B,C dataset with 5 rows."""
    df = read_csv(TESTDATA_DIR + "/simple/abc_5.csv", dtype="category")

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2)
    assert data.nodes == ("A", "B", "C")
    assert data.N == 5

    assert data.ext_to_orig == {"A": "A", "B": "B", "C": "C"}
    assert data.orig_to_ext == {"A": "A", "B": "B", "C": "C"}
    assert data.node_values == {
        "A": {"0": 2, "1": 3},
        "B": {"0": 2, "1": 3},
        "C": {"0": 4, "1": 1},
    }
    assert data.node_types == {
        "A": "category",
        "B": "category",
        "C": "category",
    }
    assert data.dstype == "categorical"

    assert data.get_order() == ("A", "B", "C")
    assert (data.as_df() == df).all().all()


# Verify constructor with larger 3-var categorical dataset (36 rows)
def test_constructor_abc36_1_ok():
    """A,B,C dataset with 36 rows."""
    df = read_csv(TESTDATA_DIR + "/simple/abc_36.csv", dtype="category")

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2)
    assert data.nodes == ("A", "B", "C")
    assert data.N == 36

    assert data.ext_to_orig == {"A": "A", "B": "B", "C": "C"}
    assert data.orig_to_ext == {"A": "A", "B": "B", "C": "C"}
    assert data.node_values == {
        "A": {"0": 10, "1": 26},
        "B": {"0": 14, "1": 22},
        "C": {"0": 16, "1": 20},
    }
    assert data.node_types == {
        "A": "category",
        "B": "category",
        "C": "category",
    }
    assert data.dstype == "categorical"

    assert data.get_order() == ("A", "B", "C")
    assert (data.as_df() == df).all().all()


# Verify constructor with continuous dataset (float32 variables)
def test_constructor_xy3_1_ok():
    """XY with 3 continuous rows."""
    df = read_csv(TESTDATA_DIR + "/simple/xy_3.csv", dtype="float32")

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1)
    assert data.nodes == ("F1", "F2")
    assert data.N == 3

    assert data.ext_to_orig == {"F1": "F1", "F2": "F2"}
    assert data.orig_to_ext == {"F1": "F1", "F2": "F2"}
    assert data.node_values == {}
    assert data.node_types == {"F1": "float32", "F2": "float32"}
    assert data.dstype == "continuous"

    assert data.get_order() == ("F1", "F2")
    assert (data.as_df() == df).all().all()


# Verify constructor with Cancer benchmark dataset (5 variables)
def test_constructor_cancer_1_ok():
    """Cancer dataset with 10 rows."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/cancer.data.gz",
        dtype="category",
        nrows=10,
    )

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4)
    assert data.nodes == ("Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray")
    assert data.N == 10
    assert data.ext_to_orig == {
        "Cancer": "Cancer",
        "Dyspnoea": "Dyspnoea",
        "Pollution": "Pollution",
        "Smoker": "Smoker",
        "Xray": "Xray",
    }
    assert data.orig_to_ext == {
        "Cancer": "Cancer",
        "Dyspnoea": "Dyspnoea",
        "Pollution": "Pollution",
        "Smoker": "Smoker",
        "Xray": "Xray",
    }
    assert data.node_values == {
        "Cancer": {"False": 10},
        "Dyspnoea": {"False": 5, "True": 5},
        "Pollution": {"low": 7, "high": 3},
        "Smoker": {"False": 7, "True": 3},
        "Xray": {"negative": 8, "positive": 2},
    }
    assert data.node_types == {
        "Cancer": "category",
        "Dyspnoea": "category",
        "Pollution": "category",
        "Smoker": "category",
        "Xray": "category",
    }
    assert data.dstype == "categorical"

    assert data.get_order() == (
        "Cancer",
        "Dyspnoea",
        "Pollution",
        "Smoker",
        "Xray",
    )
    assert (data.sample == df).all().all()


# Verify constructor with Asia benchmark dataset (8 variables)
def test_constructor_asia_1_ok():
    """Asia dataset with 100 rows."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == (
        "asia",
        "bronc",
        "dysp",
        "either",
        "lung",
        "smoke",
        "tub",
        "xray",
    )
    assert data.N == 100
    assert data.ext_to_orig == {
        "asia": "asia",
        "bronc": "bronc",
        "dysp": "dysp",
        "either": "either",
        "lung": "lung",
        "smoke": "smoke",
        "tub": "tub",
        "xray": "xray",
    }
    assert data.orig_to_ext == {
        "asia": "asia",
        "bronc": "bronc",
        "dysp": "dysp",
        "either": "either",
        "lung": "lung",
        "smoke": "smoke",
        "tub": "tub",
        "xray": "xray",
    }
    assert data.node_values == {
        "asia": {"no": 97, "yes": 3},
        "bronc": {"no": 56, "yes": 44},
        "dysp": {"no": 56, "yes": 44},
        "either": {"no": 92, "yes": 8},
        "lung": {"no": 93, "yes": 7},
        "smoke": {"no": 57, "yes": 43},
        "tub": {"no": 99, "yes": 1},
        "xray": {"no": 90, "yes": 10},
    }
    assert data.node_types == {
        "asia": "category",
        "bronc": "category",
        "dysp": "category",
        "either": "category",
        "lung": "category",
        "smoke": "category",
        "tub": "category",
        "xray": "category",
    }
    assert data.dstype == "categorical"

    assert data.get_order() == (
        "asia",
        "bronc",
        "dysp",
        "either",
        "lung",
        "smoke",
        "tub",
        "xray",
    )
    assert (data.sample == df).all().all()
