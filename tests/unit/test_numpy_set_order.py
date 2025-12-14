"""
Tests for NumPy data adapter set_order method.

These tests validate column reordering functionality and error handling
without requiring external file dependencies.
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


# Test set_order() requires order argument
def test_set_order_type_error_1_(ab3):
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])
    with pytest.raises(TypeError):
        data.set_order()


# Test set_order() order argument must be tuple
def test_set_order_type_error_2_(ab3):
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])
    with pytest.raises(TypeError):
        data.set_order(None)
    with pytest.raises(TypeError):
        data.set_order(True)
    with pytest.raises(TypeError):
        data.set_order(12)
    with pytest.raises(TypeError):
        data.set_order(list(data.nodes))
    with pytest.raises(TypeError):
        data.set_order(tuple([1, 2]))


def test_set_order_value_error_1_(ab3):  # AB bad arg values
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])
    with pytest.raises(ValueError):
        data.set_order(tuple())
    with pytest.raises(ValueError):
        data.set_order(("A",))
    with pytest.raises(ValueError):
        data.set_order(("B",))
    with pytest.raises(ValueError):
        data.set_order(("A", "B", "extra"))


def test_set_order_abc36_1_ok(abc36):  # ABC 36 rows
    data = NumPy(abc36["d"], abc36["t"], abc36["v"])
    print("\n\nOriginal ABC order:\n{}".format(data.as_df().head()))

    order = ("B", "C", "A")
    data.set_order(order)
    print("\n{} order:\n{}".format(order, data.as_df().head()))
    assert data.order == (1, 2, 0)
    assert data.get_order() == order
    assert data.node_values == {
        "A": {"0": 10, "1": 26},
        "B": {"0": 14, "1": 22},
        "C": {"0": 16, "1": 20},
    }

    order = ("C", "A", "B")
    data.set_order(order)
    print("\n{} order:\n{}".format(order, data.as_df().head()))
    assert data.order == (2, 0, 1)
    assert data.get_order() == order
    assert data.node_values == {
        "A": {"0": 10, "1": 26},
        "B": {"0": 14, "1": 22},
        "C": {"0": 16, "1": 20},
    }

    order = ("A", "B", "C")
    data.set_order(order)
    print("\n{} order:\n{}".format(order, data.as_df().head()))
    assert data.order == (0, 1, 2)
    assert data.get_order() == order
    assert data.node_values == {
        "A": {"0": 10, "1": 26},
        "B": {"0": 14, "1": 22},
        "C": {"0": 16, "1": 20},
    }

    assert (data.data == abc36["d"]).all().all()


# Test set_order() reordering columns for continuous data with 3 rows
def test_set_order_xy3_1_ok(xy3):
    data = NumPy(xy3["d"], xy3["t"], xy3["v"])
    print("\n\nOriginal XY order:\n{}".format(data.as_df().head()))

    order = ("Y", "X")
    data.set_order(order)
    print("\n{} order:\n{}".format(order, data.as_df().head()))
    assert data.order == (1, 0)
    assert data.get_order() == order
    assert data.node_values == {}

    order = ("X", "Y")
    data.set_order(order)
    print("\n{} order:\n{}".format(order, data.as_df().head()))
    assert data.order == (0, 1)
    assert data.get_order() == order
    assert data.node_values == {}

    assert (data.data == xy3["d"]).all().all()
