"""
Tests for NumPy data adapter set_N method error handling.

These tests validate parameter validation and error conditions for the set_N
method without requiring external file dependencies.
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


def test_set_N_type_error_1(ab3):  # AB 3 rows, no args
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(TypeError):
        data.set_N()


def test_set_N_type_error_2(ab3):  # Invalid type for N
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(TypeError):
        data.set_N(2.1)
    with pytest.raises(TypeError):
        data.set_N(True)
    with pytest.raises(TypeError):
        data.set_N(None)
    with pytest.raises(TypeError):
        data.set_N([2])


def test_set_N_type_error_3(ab3):  # Invalid type for seed
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(TypeError):
        data.set_N(N=3, seed=True)
    with pytest.raises(TypeError):
        data.set_N(N=2, seed=[1])
    with pytest.raises(TypeError):
        data.set_N(N=2, seed=2.1)


def test_set_N_type_error_4(ab3):  # random_selection not bool
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(TypeError):
        data.set_N(N=3, random_selection="bad")
    with pytest.raises(TypeError):
        data.set_N(N=3, random_selection=1)


def test_set_N_value_error_1(ab3):  # set non-positive N
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(ValueError):
        data.set_N(0)
    with pytest.raises(ValueError):
        data.set_N(-3)


def test_set_N_value_error_2(ab3):  # N larger than amount of data
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(ValueError):
        data.set_N(4)


# Test set_N() validates seed value range
def test_set_N_value_error_3(ab3):
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    with pytest.raises(ValueError):
        data.set_N(80, seed=-1)
    with pytest.raises(ValueError):
        data.set_N(80, seed=101)
