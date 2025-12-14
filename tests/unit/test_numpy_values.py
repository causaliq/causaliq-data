"""
Tests for NumPy data adapter values method.

These tests validate the values extraction functionality for continuous data
without requiring external file dependencies.
"""

import pytest
from numpy import array, ndarray

from causaliq_data import NumPy


@pytest.fixture(scope="module")  # continuous XYZ, 3 rows
def xyz3():
    data = array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype="float32"
    )
    dstype = "continuous"
    col_values = {"X": None, "Y": None, "Z": None}
    return NumPy(data, dstype, col_values)


# Test values() requires nodes argument
def test_values_type_error_1(xyz3):
    with pytest.raises(TypeError):
        xyz3.values()


# Test values() requires tuple of node names, not other types
def test_values_type_error_2(xyz3):
    with pytest.raises(TypeError):
        xyz3.values(False)
    with pytest.raises(TypeError):
        xyz3.values("X")
    with pytest.raises(TypeError):
        xyz3.values(["X"])
    with pytest.raises(TypeError):
        xyz3.values(12.7)


# Test values() rejects duplicate node names
def test_values_value_error_1(xyz3):
    with pytest.raises(ValueError):
        xyz3.values(("X", "X"))
    with pytest.raises(ValueError):
        xyz3.values(("Y", "X", "Y"))


# Test values() extracting single node X
def test_values_xyz3_1_(xyz3):
    nodes = ("X",)
    values = xyz3.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (3, 1)
    assert (values == [[1.0], [4.0], [7.0]]).all().all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Test values() extracting single node Z
def test_values_xyz3_2_(xyz3):
    nodes = ("Z",)
    values = xyz3.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (3, 1)
    assert (values == [[3.0], [6.0], [9.0]]).all().all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Test values() extracting nodes Z and Y in specific order
def test_values_xyz3_3_(xyz3):
    nodes = ("Z", "Y")
    values = xyz3.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (3, 2)
    assert (values == [[3.0, 2.0], [6.0, 5.0], [9.0, 8.0]]).all().all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Test values() extracting all nodes in different order Y, Z, X
def test_values_xyz3_4_(xyz3):
    nodes = ("Y", "Z", "X")
    values = xyz3.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (3, 3)
    assert (
        (values == [[2.0, 3.0, 1.0], [5.0, 6.0, 4.0], [8.0, 9.0, 7.0]])
        .all()
        .all()
    )
    print("\n\nData for {} is:\n{}".format(nodes, values))
