"""
Tests for NumPy data adapter constructor functionality.

These tests validate NumPy class instantiation, parameter validation, and
basic object setup without requiring external file dependencies.
"""

import pytest
from numpy import array

from causaliq_data import NumPy


@pytest.fixture(scope="module")  # categorical AB, 3 rows
def ab3():
    data = array([[1, 1], [1, 0], [0, 0]], dtype="uint8")
    dstype = "categorical"
    col_values = {"A": ("1", "0"), "B": ("1", "0")}
    return {"d": data, "t": dstype, "v": col_values}


@pytest.fixture(scope="module")  # categorical ABC, 5 rows
def abc5():
    data = array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype="uint8"
    )
    dstype = "categorical"
    col_values = {"A": ("0", "1"), "B": ("0", "1"), "C": ("0", "1")}
    return {"d": data, "t": dstype, "v": col_values}


@pytest.fixture(scope="module")  # categorical ABC, 36 rows
def abc36():
    data = array(
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
        ],
        dtype="uint8",
    )
    dstype = "categorical"
    col_values = {"A": ("0", "1"), "B": ("0", "1"), "C": ("0", "1")}
    return {"d": data, "t": dstype, "v": col_values}


@pytest.fixture(scope="module")  # continuous XY, 3 rows
def xy3():
    data = array([[1, 1], [1, 0], [0, 0]], dtype="float32")
    dstype = "continuous"
    col_values = {"X": None, "Y": None}
    return {"d": data, "t": dstype, "v": col_values}


# Constructor errors


# Test NumPy constructor requires all arguments
def test_constructor_type_error_1_():
    with pytest.raises(TypeError):
        NumPy()


# Test NumPy constructor requires data, dstype and col_values arguments
def test_constructor_type_error_2_(ab3):
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"])
    with pytest.raises(TypeError):
        NumPy(ab3["d"])


def test_constructor_type_error_3_(ab3):  # data not an ndarray
    with pytest.raises(TypeError):
        NumPy(None, ab3["t"], ab3["v"])
    with pytest.raises(TypeError):
        NumPy(True, ab3["t"], ab3["v"])
    with pytest.raises(TypeError):
        NumPy(1, ab3["t"], ab3["v"])
    with pytest.raises(TypeError):
        NumPy([[2, 3], [2, 3]], ab3["t"], ab3["v"])


def test_constructor_type_error_4_(ab3):  # dstype not string or DatasetType
    with pytest.raises(TypeError):
        NumPy(ab3["d"], None, ab3["v"])
    with pytest.raises(TypeError):
        NumPy(ab3["d"], False, ab3["v"])
    with pytest.raises(TypeError):
        NumPy(ab3["d"], "invalid", ab3["v"])
    with pytest.raises(TypeError):
        NumPy(ab3["d"], 2, ab3["v"])


def test_constructor_type_error_5_(ab3):  # col_types not dict
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], True)
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], None)
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], 31.2)
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], ["A", "B"])


def test_constructor_type_error_6_(ab3):  # col_types keys not all strings
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], {"A": ("Y", "N"), 2: ("0", "1")})


def test_constructor_type_error_7_(ab3):  # col_types values not all tuples
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], {"A": ["Y", "N"], "B": ["0", "1"]})


def test_constructor_type_error_8_(ab3):  # tuple values not all strings
    with pytest.raises(TypeError):
        NumPy(ab3["d"], ab3["t"], {"A": ("Y", "N"), "B": (0, 1)})


def test_constructor_type_error_9_(ab3):  # col_types values not None
    with pytest.raises(TypeError):
        NumPy(ab3["d"], "continuous", {"A": ("Y", "N"), "B": ("0", "1")})


def test_constructor_value_error_1_():  # less than two columns
    with pytest.raises(ValueError):
        NumPy(array([[1], [0]]), "categorical", {"A": ("Y", "N")})


def test_constructor_value_error_2_():  # less than two rows
    with pytest.raises(ValueError):
        NumPy(array([[0, 0]]), "categorical", {"A": ("Y",), "B": ("N",)})


def test_constructor_value_error_3_():  # data/col_values column count mismatch
    with pytest.raises(ValueError):
        NumPy(
            array([[0, 0], [1, 1]]),
            "categorical",
            {"A": ("Y",), "B": ("N",), "C": ("1",)},
        )


def test_constructor_value_error_4_(ab3):  # categorical, dtype not uint8
    with pytest.raises(ValueError):
        NumPy(ab3["d"], "continuous", {"A": None, "B": None})


def test_constructor_value_error_5_(ab3, xy3):  # continuous, dtype not float32
    with pytest.raises(ValueError):
        NumPy(xy3["d"], "categorical", ab3["v"])


# Test constructor setting member variables correctly


def test_constructor_ab3_1_ok(ab3):
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (data.data == array([[1, 1], [1, 0], [0, 0]])).all().all()
    assert data.nodes == ("A", "B")
    assert data.order == (0, 1)
    assert data.ext_to_orig == {"A": "A", "B": "B"}
    assert data.orig_to_ext == {"A": "A", "B": "B"}
    assert data.N == 3
    assert data.dstype == "categorical"
    assert (data.categories == (("1", "0"), ("1", "0"))).all().all()
    assert data.node_values == {"A": {"0": 2, "1": 1}, "B": {"0": 1, "1": 2}}
    assert data.node_types == {"A": "category", "B": "category"}


def test_constructor_abc5_1_ok(abc5):  # A,B,C dataset with 5 rows
    data = NumPy(abc5["d"], abc5["t"], abc5["v"])

    assert isinstance(data, NumPy)
    assert data.data.dtype == "uint8"
    assert (
        (
            data.data
            == array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
        )
        .all()
        .all()
    )
    assert data.nodes == ("A", "B", "C")
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {"A": "A", "B": "B", "C": "C"}
    assert data.orig_to_ext == {"A": "A", "B": "B", "C": "C"}
    assert data.N == 5
    assert data.dstype == "categorical"
    assert (
        (data.categories == (("0", "1"), ("0", "1"), ("0", "1"))).all().all()
    )
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


# Test NumPy constructor correctly initializes larger N=36 categorical dataset
def test_constructor_abc36_1_ok(abc36):
    data = NumPy(abc36["d"], abc36["t"], abc36["v"])

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
                ],
                dtype="uint8",
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


# Test NumPy constructor correctly initializes continuous dataset
def test_constructor_xy3_1_ok(xy3):
    data = NumPy(xy3["d"], xy3["t"], xy3["v"])

    assert isinstance(data, NumPy)
    assert data.data.dtype == "float32"
    assert (
        (data.data == array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])).all().all()
    )
    assert data.nodes == ("X", "Y")
    assert data.order == (0, 1)
    assert data.ext_to_orig == {"X": "X", "Y": "Y"}
    assert data.orig_to_ext == {"X": "X", "Y": "Y"}
    assert data.N == 3
    assert data.dstype == "continuous"
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {"X": "float32", "Y": "float32"}
