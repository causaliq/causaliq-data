# Test Pandas values() function

import pytest
from numpy import ndarray


# values() requires nodes argument
def test_values_type_error_1(xyz_data):
    """No argument specified."""
    with pytest.raises(TypeError):
        xyz_data.values()


# values() requires tuple of node names, not other types
def test_values_type_error_2(xyz_data):
    """Bad nodes argument type."""
    with pytest.raises(TypeError):
        xyz_data.values(False)
    with pytest.raises(TypeError):
        xyz_data.values("X")
    with pytest.raises(TypeError):
        xyz_data.values(["X"])
    with pytest.raises(TypeError):
        xyz_data.values(12.7)


# values() rejects duplicate node names
def test_values_value_error_1(xyz_data):
    """Duplicate node names."""
    with pytest.raises(ValueError):
        xyz_data.values(("X", "X"))
    with pytest.raises(ValueError):
        xyz_data.values(("Y", "X", "Y"))


# values() rejects non-existent node names
def test_values_value_error_2(xyz_data):
    """Nodes not in dataset."""
    with pytest.raises(ValueError):
        xyz_data.values(("X", "Y", "invalid"))
    with pytest.raises(ValueError):
        xyz_data.values(("badun",))


# Extract single continuous variable X as numpy array
def test_values_1_ok(xyz_data):
    """Extract X."""
    nodes = ("X",)
    values = xyz_data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values[:, 0] == xyz_data.sample["X"].values).all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Extract single continuous variable Y as numpy array
def test_values_2_ok(xyz_data):
    """Extract Y."""
    nodes = ("Y",)
    values = xyz_data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values[:, 0] == xyz_data.sample["Y"].values).all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Extract single continuous variable Z as numpy array
def test_values_3_ok(xyz_data):
    """Extract Z."""
    nodes = ("Z",)
    values = xyz_data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values[:, 0] == xyz_data.sample["Z"].values).all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Extract two continuous variables X, Y maintaining column order
def test_values_4_ok(xyz_data):
    """Extract X, Y."""
    nodes = ("X", "Y")
    values = xyz_data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 2)
    assert (values[:, 0] == xyz_data.sample["X"].values).all()
    assert (values[:, 1] == xyz_data.sample["Y"].values).all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Extract two continuous variables Y, X in reversed order
def test_values_5_ok(xyz_data):
    """Extract Y, X."""
    nodes = ("Y", "X")
    values = xyz_data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 2)
    assert (values[:, 0] == xyz_data.sample["Y"].values).all()
    assert (values[:, 1] == xyz_data.sample["X"].values).all()
    print("\n\nData for {} is:\n{}".format(nodes, values))


# Extract three continuous variables Y, Z, X in custom order
def test_values_6_ok(xyz_data):
    """Extract Y, Z, X."""
    nodes = ("Y", "Z", "X")
    values = xyz_data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 3)
    assert (values[:, 0] == xyz_data.sample["Y"].values).all()
    assert (values[:, 1] == xyz_data.sample["Z"].values).all()
    assert (values[:, 2] == xyz_data.sample["X"].values).all()
    print("\n\nData for {} is:\n{}".format(nodes, values))
