"""
Tests for NumPy data adapter from_df class method functionality.

These tests validate DataFrame to NumPy conversion functionality
that require external test data files.
"""

import os

import pytest
from numpy import array

from causaliq_data import NumPy, Pandas

# Path to test data relative to this file
TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "functional"
)


@pytest.fixture(scope="module")  # categorical AB, 3 rows as DataFrame
def ab3_df():
    return Pandas.read(
        TESTDATA_DIR + "/simple/ab_3.csv", dstype="categorical"
    ).as_df()


@pytest.fixture(scope="module")  # continuous XY, 3 rows as DataFrame
def xy3_df():
    return Pandas.read(
        TESTDATA_DIR + "/simple/xy_3.csv", dstype="continuous"
    ).as_df()


# from_df error tests


# Test from_df() requires all arguments
def test_from_df_type_error_1_():
    with pytest.raises(TypeError):
        NumPy.from_df()


# Test from_df() requires df, dstype and keep_df arguments
def test_from_df_type_error_2_(ab3_df):
    with pytest.raises(TypeError):
        NumPy.from_df(ab3_df)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype="categorical")
    with pytest.raises(TypeError):
        NumPy.from_df(ab3_df, keep_df=True)


def test_from_df_type_error_3_():  # df bad type
    with pytest.raises(TypeError):
        NumPy.from_df(df=True, dstype="categorical", keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=None, dstype="categorical", keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=2, dstype="categorical", keep_df=False)


def test_from_df_type_error_4_(ab3_df):  # dstype bad type
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype="invalid", keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype=["categorical"], keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype=True, keep_df=False)


def test_from_df_type_error_5_(ab3_df):  # keep_df bad type
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype="categorical", keep_df=None)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype="categorical", keep_df=1)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype="categorical", keep_df={True})


def test_from_df_value_error_1_(ab3_df):  # too few rows in df
    with pytest.raises(ValueError):
        NumPy.from_df(df=ab3_df[:1], dstype="categorical", keep_df=True)


def test_from_df_value_error_2_(ab3_df):  # too few columns in df
    with pytest.raises(ValueError):
        NumPy.from_df(df=ab3_df[["A"]], dstype="categorical", keep_df=True)


def test_from_df_value_error_3_(ab3_df):  # type mismatch
    with pytest.raises(ValueError):
        NumPy.from_df(df=ab3_df, dstype="continuous", keep_df=True)


def test_from_df_value_error_4_(xy3_df):  # type mismatch
    with pytest.raises(ValueError):
        NumPy.from_df(df=xy3_df, dstype="categorical", keep_df=True)


# from_df successful conversion tests


def test_from_df_ab3_1_ok():  # AB 3 rows categorical data. keep_df = True
    dstype = "categorical"
    df = Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (data.data == array([[0, 0], [0, 1], [1, 1]])).all().all()
    assert data.nodes == ("A", "B")
    assert data.order == (0, 1)
    assert data.ext_to_orig == {"A": "A", "B": "B"}
    assert data.orig_to_ext == {"A": "A", "B": "B"}
    assert data.N == 3
    assert data.dstype == "categorical"
    assert (data.categories == (("1", "0"), ("1", "0"))).all().all()
    assert data.node_values == {"A": {"0": 1, "1": 2}, "B": {"0": 2, "1": 1}}
    assert data.node_types == {"A": "category", "B": "category"}

    assert (df == data.as_df()).all().all()


def test_from_df_ab3_2_ok():  # AB 3 rows categorical data. keep_df = False
    dstype = "categorical"
    df = Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=False)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (data.data == array([[0, 0], [0, 1], [1, 1]])).all().all()
    assert data.nodes == ("A", "B")
    assert data.order == (0, 1)
    assert data.ext_to_orig == {"A": "A", "B": "B"}
    assert data.orig_to_ext == {"A": "A", "B": "B"}
    assert data.N == 3
    assert data.dstype == "categorical"
    assert (data.categories == (("1", "0"), ("1", "0"))).all().all()
    assert data.node_values == {"A": {"0": 1, "1": 2}, "B": {"0": 2, "1": 1}}
    assert data.node_types == {"A": "category", "B": "category"}

    assert not (df == data.as_df()).all().all()


def test_from_df_abc36_1_ok():  # ABC 36 rows categorical data, keep_df = True
    dstype = "categorical"
    df = Pandas.read(
        TESTDATA_DIR + "/simple/abc_36.csv", dstype=dstype
    ).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (
        (
            array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            )
            == data.data
        )
        .all()
        .all()
    )
    assert data.nodes == ("A", "B", "C")
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {"A": "A", "B": "B", "C": "C"}
    assert data.orig_to_ext == {"A": "A", "B": "B", "C": "C"}
    assert data.N == 36
    assert data.dstype == "categorical"
    assert (
        (data.categories == (("0", "1"), ("0", "1"), ("0", "1"))).all().all()
    )
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

    assert (df == data.as_df()).all().all()


def test_from_df_abc36_2_ok():  # ABC 36 rows categorical data, keep_df=False
    dstype = "categorical"
    df = Pandas.read(
        TESTDATA_DIR + "/simple/abc_36.csv", dstype=dstype
    ).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=False)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (
        (
            array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            )
            == data.data
        )
        .all()
        .all()
    )
    assert data.nodes == ("A", "B", "C")
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {"A": "A", "B": "B", "C": "C"}
    assert data.orig_to_ext == {"A": "A", "B": "B", "C": "C"}
    assert data.N == 36
    assert data.dstype == "categorical"
    assert (
        (data.categories == (("0", "1"), ("0", "1"), ("0", "1"))).all().all()
    )
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

    assert not (df == data.as_df()).all().all()


def test_from_df_cancer_1_ok():  # Cancer dataset with 10 rows
    df = Pandas.read(
        TESTDATA_DIR + "/experiments/datasets/cancer.data.gz",
        dstype="categorical",
        N=10,
    ).as_df()
    data = NumPy.from_df(df=df, dstype="categorical", keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (
        (
            array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ]
            )
            == data.data
        )
        .all()
        .all()
    )
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

    assert (df == data.as_df()).all().all()


def test_from_df_asia_1_ok():  # Asia dataset with 100 rows
    df = Pandas.read(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dstype="categorical",
        N=100,
    ).as_df()
    data = NumPy.from_df(df=df, dstype="categorical", keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
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

    assert (df == data.as_df()).all().all()


def test_from_df_xy3_1_ok():  # XY 3 rows of continuous data, keep_df = True
    dstype = "continuous"
    df = Pandas.read(TESTDATA_DIR + "/simple/xy_3.csv", dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "float32"
    assert (
        (
            data.data
            == array(
                [[1.01, 1.21], [-0.45, 0.67], [1.22, -1.41]], dtype="float32"
            )
        )
        .all()
        .all()
    )
    assert data.nodes == ("F1", "F2")
    assert data.order == (0, 1)
    assert data.ext_to_orig == {"F1": "F1", "F2": "F2"}
    assert data.orig_to_ext == {"F1": "F1", "F2": "F2"}
    assert data.N == 3
    assert data.dstype == "continuous"
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {"F1": "float32", "F2": "float32"}

    assert data.as_df().map(lambda x: round(x, 2)).to_dict(orient="list") == {
        "F1": [-0.45, 1.01, 1.22],
        "F2": [0.67, 1.21, -1.41],
    }


def test_from_df_xy3_2_ok():  # XY 3 rows of continuous data, keep_df = False
    dstype = "continuous"
    df = Pandas.read(TESTDATA_DIR + "/simple/xy_3.csv", dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=False)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "float32"
    assert (
        (
            data.data
            == array(
                [[1.01, 1.21], [-0.45, 0.67], [1.22, -1.41]], dtype="float32"
            )
        )
        .all()
        .all()
    )
    assert data.nodes == ("F1", "F2")
    assert data.order == (0, 1)
    assert data.ext_to_orig == {"F1": "F1", "F2": "F2"}
    assert data.orig_to_ext == {"F1": "F1", "F2": "F2"}
    assert data.N == 3
    assert data.dstype == "continuous"
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {"F1": "float32", "F2": "float32"}

    assert data.as_df().map(lambda x: round(x, 2)).to_dict(orient="list") == {
        "F1": [-0.45, 1.01, 1.22],
        "F2": [0.67, 1.21, -1.41],
    }


def test_from_df_xyz10_1_ok():  # XYZ 10 rows of continuous data
    dstype = "continuous"
    df = Pandas.read(
        TESTDATA_DIR + "/simple/xyz_10.csv", dstype=dstype
    ).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == "float32"
    assert (
        (
            data.data
            == array(
                [
                    [1.1, 0.3, 0.3],
                    [0.0, 3.1, 4.0],
                    [0.2, 5.4, 1.7],
                    [4.4, 6.6, 1.9],
                    [0.6, 2.8, 9.9],
                    [4.0, 6.0, 9.0],
                    [2.2, 3.1, 0.8],
                    [0.1, 0.0, 2.2],
                    [7.1, 3.9, 1.4],
                    [6.0, 0.2, 0.5],
                ],
                dtype="float32",
            )
        )
        .all()
        .all()
    )
    assert data.nodes == ("X", "Y", "Z")
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {"X": "X", "Y": "Y", "Z": "Z"}
    assert data.orig_to_ext == {"X": "X", "Y": "Y", "Z": "Z"}
    assert data.N == 10
    assert data.dstype == "continuous"
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {"X": "float32", "Y": "float32", "Z": "float32"}

    assert data.as_df().map(lambda x: round(x, 2)).to_dict(orient="list") == {
        "X": [0.0, 0.1, 0.2, 0.6, 1.1, 2.2, 4.0, 4.4, 6.0, 7.1],
        "Y": [3.1, 0.0, 5.4, 2.8, 0.3, 3.1, 6.0, 6.6, 0.2, 3.9],
        "Z": [4.0, 2.2, 1.7, 9.9, 0.3, 0.8, 9.0, 1.9, 0.5, 1.4],
    }
